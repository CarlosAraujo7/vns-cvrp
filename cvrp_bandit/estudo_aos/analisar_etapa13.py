#!/usr/bin/env python3
"""
analisar_etapa13.py
===================
Analise do Estudo 13 (generalidade: TSP) conforme o PRE-REGISTRO (diario, Entrada 113). dif = gap(a) - gap(b); negativo = a melhor.
Teste de Wilcoxon bilateral pareado por instancia (media das seeds), IC95% bootstrap (10.000, semente 0), empate |dif| < 1e-9.

Regime VND (etapa13_eval_v.csv; cenarios TSPLIB E1, TSPLIB E2, SYN_VAL_T E1):
  Familia A (Holm m=21): {ucb1c, pmc, hgucbc, ts, exp3, eff, static} - U nos 3 cenarios
  Familia B (Holm m=3) : efeito do conjunto, U(P9) - U(P5), nos 3 cenarios
  Exploratorias        : C (TSPLIB E3 = 25 E1; 7 testes, Holm m=7), D ({eff, ts} - static nos 3 cenarios, Holm m=6)
Controle positivo (etapa13_eval_pc.csv, TSPLIB E1): 6 politicas - U no mesmo pool, pools D8 e D16 (Holm m=12); recuperacao da perda
  plantada (exploratoria) = (gap_U_pool - gap_pol_pool) / (gap_U_pool - gap_U_limpo).
Regime ALNS (etapa13_eval_a.csv, 62 instancias):
  Primario (sem correcao): RP - U (62 instancias)
  Secundarios (Holm m=12): {ucb1, pm, hgucb, ts, exp3, eff} - U em TSPLIB e em SYN_VAL_T
  Terciarios (Holm m=4)  : static - U, fixrep - U, RP - static, RP - fixrep (62 instancias)
  Estrutura (Holm m=6)   : ALNS-U - {VND U P9, VND U P5, VND static} ao mesmo orcamento (etapa13_eval_ref.csv), por conjunto
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
N_BOOT = 10000
POL_V = ["ucb1c", "pmc", "hgucbc", "ts", "exp3", "eff", "static"]
POL_PC = ["ucb1c", "pmc", "hgucbc", "ts", "exp3", "eff"]
POL_A_SEC = ["ucb1", "pm", "hgucb", "ts", "exp3", "eff"]


def holm(pvals, m):
    ordem = np.argsort(pvals)
    aj = np.empty(len(pvals))
    corrente = 0.0
    for rank, idx in enumerate(ordem):
        corrente = max(corrente, min(1.0, (m - rank) * pvals[idx]))
        aj[idx] = corrente
    return aj


def contraste(piv, cid, a, b, rng, grupo, **extra):
    dif = piv[a].values - piv[b].values
    dif = np.where(np.abs(dif) < 1e-9, 0.0, dif)
    idx = rng.integers(0, len(dif), size=(N_BOOT, len(dif)))
    med = dif[idx].mean(axis=1)
    lo, hi = np.percentile(med, [2.5, 97.5])
    lo90, hi90 = np.percentile(med, [5, 95])
    p = stats.wilcoxon(dif).pvalue if np.any(dif != 0) else 1.0
    return dict(grupo=grupo, id=cid, a=a, b=b, n_inst=len(dif), dif_pp=dif.mean(), ic_lo=lo, ic_hi=hi,
                vit_a=int((dif < 0).sum()), der_a=int((dif > 0).sum()), emp=int((dif == 0).sum()), p_bruto=p,
                equiv_010=bool(lo90 > -0.1 and hi90 < 0.1), mde_pp=2.8 * dif.std(ddof=1) / np.sqrt(len(dif)), **extra)


def pivo(d):
    t = d[d.viavel].groupby(["braco", "instancia"]).gap_pct.mean().unstack(0).dropna()
    return t


def com_holm(linhas, m):
    r = pd.DataFrame(linhas)
    r["p_holm"] = holm(r.p_bruto.values, m)
    r["signif_005"] = r.p_holm < 0.05
    return r


def analisa_v(rng):
    e = pd.read_csv(RES / "etapa13_eval_v.csv")
    print(f"eval VND: {len(e)} execucoes, inviaveis {int((~e.viavel).sum())}")
    cen = {"TSPLIB E1": e[(e.conjunto == "TSPLIB") & (e.nivel == 1)], "TSPLIB E2": e[(e.conjunto == "TSPLIB") & (e.nivel == 5)],
           "SYN_VAL_T E1": e[(e.conjunto == "SYN_VAL_T") & (e.nivel == 1)], "TSPLIB E3": e[(e.conjunto == "TSPLIB") & (e.nivel == 25)]}
    piv = {}
    for nome, d in cen.items():
        t = pivo(d)
        t.columns = [c.replace("s13v_", "") for c in t.columns]
        piv[nome] = t
        print(f"\n=== {nome}: {len(t)} instancias; gap medio ===")
        print(t.mean().round(3).sort_values().to_string())
    A = com_holm([contraste(piv[n], f"{a}-U@{n}", a, "U", rng, "familiaA", cenario=n) for n in ("TSPLIB E1", "TSPLIB E2", "SYN_VAL_T E1")
                  for a in POL_V], 21)
    B = com_holm([contraste(piv[n], f"U-P5@{n}", "U", "P5_U", rng, "familiaB", cenario=n)
                  for n in ("TSPLIB E1", "TSPLIB E2", "SYN_VAL_T E1")], 3)
    C = com_holm([contraste(piv["TSPLIB E3"], f"{a}-U@TSPLIB E3", a, "U", rng, "explor_C", cenario="TSPLIB E3") for a in POL_V], 7)
    D = com_holm([contraste(piv[n], f"{a}-static@{n}", a, "static", rng, "explor_D", cenario=n) for n in
                  ("TSPLIB E1", "TSPLIB E2", "SYN_VAL_T E1") for a in ("eff", "ts")], 6)
    r = pd.concat([A, B, C, D], ignore_index=True)
    pd.set_option("display.width", 240)
    print("\n=== VND: familias A (m=21), B (m=3) e exploratorias C (m=7), D (m=6) ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa13_v.csv", index=False)
    on = A[A.a != "static"]
    print(f"\nH54 (conjunto: P9 pior que P5, significativo em TSPLIB E1 e E2): "
          f"{int(((B.cenario.isin(['TSPLIB E1', 'TSPLIB E2'])) & (B.dif_pp > 0) & B.signif_005).sum())} de 2")
    print(f"H55 (eff significativamente melhor que U em algum cenario): {int(((A.a == 'eff') & (A.dif_pp < 0) & A.signif_005).sum())} de 3")
    print(f"H56 (ts significativamente melhor que U em algum cenario): {int(((A.a == 'ts') & (A.dif_pp < 0) & A.signif_005).sum())} de 3")
    print(f"H57 (ucb1c, pmc, hgucbc, exp3 melhores que U): {int((A.a.isin(['ucb1c', 'pmc', 'hgucbc', 'exp3']) & (A.dif_pp < 0) & A.signif_005).sum())}")
    print(f"H58 (static significativamente melhor que U em algum cenario): {int(((A.a == 'static') & (A.dif_pp < 0) & A.signif_005).sum())} de 3")
    print(f"H59 (eff ou ts significativamente melhores que static): {int(((D.dif_pp < 0) & D.signif_005).sum())} de 6")
    # POS-HOC (escrito depois de ver os resultados primarios; NAO pre-registrado; gerador proprio para nao alterar os IC dos pre-registrados)
    rng_ph = np.random.default_rng(1)
    # POS-HOC (idem): a melhor politica no pool completo contra
    # simplesmente usar o pool menor com a escolha uniforme (Holm m=9)
    P = com_holm([contraste(piv[n], f"{a}-P5_U@{n}", a, "P5_U", rng_ph, "posthoc_P", cenario=n)
                  for n in ("TSPLIB E1", "TSPLIB E2", "SYN_VAL_T E1") for a in ("eff", "ucb1c", "hgucbc")], 9)
    print("\n=== VND pos-hoc: politica no P9 - uniforme no P5 (Holm m=9) ===")
    print(P.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    P.to_csv(RES / "analise_etapa13_v_posthoc.csv", index=False)
    return piv


def analisa_pc(rng, piv_v):
    pc = pd.read_csv(RES / "etapa13_eval_pc.csv")
    print(f"\neval controle positivo: {len(pc)} execucoes, inviaveis {int((~pc.viavel).sum())}")
    linhas, rec = [], []
    limpo = piv_v["TSPLIB E1"]["U"]
    for tag in ("D8", "D16"):
        t = pivo(pc[pc.braco.str.startswith(f"s13pc_{tag}_")])
        t.columns = [c.replace(f"s13pc_{tag}_", "") for c in t.columns]
        print(f"\n=== pool {tag}: gap medio ===")
        print(t.mean().round(3).sort_values().to_string())
        perda = t["U"] - limpo.reindex(t.index)
        print(f"perda plantada (U no pool {tag} - U limpo): {perda.mean():.3f} pp")
        for a in POL_PC:
            linhas.append(contraste(t, f"{a}-U@{tag}", a, "U", rng, "controle", pool=tag))
            # recuperacao por instancia-media (razao das medias; IC bootstrap sobre instancias)
            idx = rng.integers(0, len(t), size=(N_BOOT, len(t)))
            num = (t["U"].values - t[a].values)
            den = perda.values
            rr = num[idx].mean(axis=1) / den[idx].mean(axis=1)
            rec.append(dict(pool=tag, politica=a, rec=num.mean() / den.mean(), rec_lo=np.percentile(rr, 2.5), rec_hi=np.percentile(rr, 97.5)))
    r = com_holm(linhas, 12)
    print("\n=== CONTROLE POSITIVO (Holm m=12) ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    print(pd.DataFrame(rec).round(3).to_string(index=False))
    r.to_csv(RES / "analise_etapa13_pc.csv", index=False)
    pd.DataFrame(rec).to_csv(RES / "analise_etapa13_recuperacao.csv", index=False)
    d16 = r[(r.pool == "D16") & (r.dif_pp < 0) & r.signif_005]
    print(f"H60 (>= 2 politicas significativamente melhores que U no pool D16): {len(d16)} ({list(d16.a)})")


def analisa_a(rng):
    e = pd.read_csv(RES / "etapa13_eval_a.csv")
    rf = pd.read_csv(RES / "etapa13_eval_ref.csv")
    print(f"\neval ALNS: {len(e)} execucoes, inviaveis {int((~e.viavel).sum())}; referencias VND: {len(rf)}")
    t = pivo(e)
    t.columns = [c.replace("s13a_", "") for c in t.columns]
    conj = e.drop_duplicates("instancia").set_index("instancia").conjunto.reindex(t.index)
    print(f"=== ALNS: {len(t)} instancias; gap medio ===")
    print(t.mean().round(3).sort_values().to_string())
    for c in ("TSPLIB", "SYN_VAL_T"):
        print(f"  {c}: " + ", ".join(f"{k} {v:.3f}" for k, v in t[conj == c].mean().sort_values().items()))
    prim = pd.DataFrame([contraste(t, "RP-U", "RP", "U", rng, "primario")])
    prim["p_holm"], prim["signif_005"] = prim.p_bruto, prim.p_bruto < 0.05
    sec = com_holm([contraste(t[conj == c], f"{a}-U@{c}", a, "U", rng, "secundario", conjunto=c)
                    for c in ("TSPLIB", "SYN_VAL_T") for a in POL_A_SEC], 12)
    ter = com_holm([contraste(t, "static-U", "static", "U", rng, "terciario"), contraste(t, "fixrep-U", "fixrep", "U", rng, "terciario"),
                    contraste(t, "RP-static", "RP", "static", rng, "terciario"), contraste(t, "RP-fixrep", "RP", "fixrep", rng, "terciario")], 4)
    # estrutura: ALNS-U contra os tres braços VND ao mesmo orcamento
    tr = pivo(rf)
    tr.columns = [c.replace("s13r_", "vnd_") for c in tr.columns]
    ju = t[["U"]].join(tr, how="inner").rename(columns={"U": "alns_U"})
    est = com_holm([contraste(ju[conj.reindex(ju.index) == c], f"alns_U-{v}@{c}", "alns_U", v, rng, "estrutura", conjunto=c)
                    for c in ("TSPLIB", "SYN_VAL_T") for v in ("vnd_U_P9", "vnd_U_P5", "vnd_static")], 6)
    # POS-HOC (nao pre-registrado): a politica adaptativa mais forte (eff) contra a mistura estatica ajustada e contra o reparo unico (Holm m=2)
    rng_ph = np.random.default_rng(2)
    ph = com_holm([contraste(t, "eff-static", "eff", "static", rng_ph, "posthoc_A"), contraste(t, "eff-fixrep", "eff", "fixrep", rng_ph, "posthoc_A")], 2)
    print("\n=== ALNS pos-hoc: eff - static, eff - fixrep (Holm m=2) ===")
    print(ph.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    ph.to_csv(RES / "analise_etapa13_a_posthoc.csv", index=False)
    r = pd.concat([prim, sec, ter, est], ignore_index=True)
    pd.set_option("display.width", 240)
    print("\n=== ALNS: primario, secundarios (m=12), terciarios (m=4), estrutura (m=6) ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa13_a.csv", index=False)
    print("\nVND ao mesmo orcamento (gap medio):", tr.mean().round(3).to_dict())
    print(f"H61 (RP - U nao significativo, alfa 0,05): {'CONFIRMADA' if not prim.signif_005.iloc[0] else 'nao confirmada'} "
          f"({prim.dif_pp.iloc[0]:+.3f} pp, p {prim.p_bruto.iloc[0]:.3f})")
    print(f"H62 (nenhuma politica secundaria melhor que U): {int(((sec.dif_pp < 0) & sec.signif_005).sum())} melhores")
    print(f"H63 (static nao melhor que U): {'CONFIRMADA' if not ((ter.id == 'static-U') & (ter.dif_pp < 0) & ter.signif_005).any() else 'nao confirmada'}")
    print(f"H65 (ALNS-U melhor que o VND U(P9), significativo, nos dois conjuntos): "
          f"{int(((est.b == 'vnd_U_P9') & (est.dif_pp < 0) & est.signif_005).sum())} de 2")
    print(f"H64 (ALNS-U melhor que cada um dos tres VND por >= 0,3 pp e significativo, nos dois conjuntos; 6 de 6): "
          f"{int(((est.dif_pp <= -0.3) & est.signif_005).sum())} de 6 contrastes")


def analisa_ordem():
    """POS-HOC (NAO pre-registrado; escrito depois de ver os resultados primarios e o uso quase uniforme de ucb1c/hgucbc):
    VND classico de ORDEM FIXA no P9 (indice crescente = custo nominal crescente; FOC_asc) e no sentido inverso (FOC_desc), sem
    aprendizado, contra U, P5_U, eff, ucb1c e hgucbc (Holm m=24: 4 cenarios x 6 contrastes)."""
    x = pd.read_csv(RES / "etapa13_explor_ordem.csv")
    e = pd.read_csv(RES / "etapa13_eval_v.csv")
    d = pd.concat([x, e], ignore_index=True)
    rng_ph = np.random.default_rng(3)
    linhas = []
    cen = {"TSPLIB E1": (d.conjunto == "TSPLIB") & (d.nivel == 1), "TSPLIB E2": (d.conjunto == "TSPLIB") & (d.nivel == 5),
           "SYN_VAL_T E1": (d.conjunto == "SYN_VAL_T") & (d.nivel == 1), "TSPLIB E3": (d.conjunto == "TSPLIB") & (d.nivel == 25)}
    print("\n=== POS-HOC: VND de ordem fixa (FOC) ===")
    for nome, msk in cen.items():
        t = pivo(d[msk])
        t.columns = [c.replace("s13v_", "").replace("s13x_", "") for c in t.columns]
        print(f"{nome}: " + ", ".join(f"{k} {v:.3f}" for k, v in t.mean().sort_values().items()))
        for a, b in (("FOC_asc", "U"), ("FOC_asc", "P5_U"), ("FOC_asc", "eff"), ("FOC_asc", "ucb1c"), ("FOC_asc", "hgucbc"), ("FOC_desc", "U")):
            linhas.append(contraste(t, f"{a}-{b}@{nome}", a, b, rng_ph, "posthoc_O", cenario=nome))
    r = com_holm(linhas, len(linhas))
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa13_ordem.csv", index=False)


def main():
    rng = np.random.default_rng(0)
    piv_v = analisa_v(rng)
    analisa_pc(rng, piv_v)
    analisa_a(rng)
    if (RES / "etapa13_explor_ordem.csv").exists():
        analisa_ordem()


if __name__ == "__main__":
    main()
