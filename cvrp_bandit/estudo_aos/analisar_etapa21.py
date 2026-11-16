#!/usr/bin/env python3
"""
analisar_etapa21.py
===================
Analise do Estudo 21 (tamanho do conjunto de operadores no ALNS do CVRP: P7, P12, P21) conforme o PRE-REGISTRO (diario, Entrada 136).
dif = gap(a) - gap(b); negativo = a melhor. Unidade = instancia (media das seeds viaveis; 51 instancias: X, 21, seeds 1000-1009, e SYN_VAL, 30, seeds 1000-1004);
Wilcoxon bilateral pareado; IC95% bootstrap (10.000, semente 0); empate |dif| < 1e-9; MDE = 2,8 DP / raiz(n); equivalencia = IC90% dentro de +-0,10 pp.
Bracos (sufixo @P7, @P12, @P21): U, RP (hiperparametros padrao), RPt (ajustado, 33 avaliacoes), S33 (mistura estatica, 33 avaliacoes), S65 (65 avaliacoes; P7 e P21),
HG (HG-UCB do Estudo 9; P7 e P21). Os bracos de P7 sao REUTILIZADOS dos Estudos 9, 12 e 20. Se o ajuste de RP nao achou configuracao melhor que a padrao,
RPt = RP; se a continuacao do ES nao achou melhora, S65 = S33 (colunas copiadas).

  Primarios (Holm m=2)      : H120 RP - S33 em P21 (negativo: a adaptacao sem ajuste supera a mistura ajustada);
                              H121 diferenca-em-diferencas [(RP - S33)@P21] - [(RP - S33)@P7] (negativo: a vantagem da adaptacao CRESCE com o conjunto)
  Secundarios (Holm m=5)    : H122 RPt - S33 em P21 (esforco de ajuste igual); H123 S65 - S33 em P21 (dobrar o ajuste ajuda); H124 RP - U em P21; H125 S33 - U em P21;
                              H126 HG - U em P21
  Terciarios (Holm m=3)     : H127 RP - S33 em P12; H128 RPt - S33 em P12; H129 [(RP - S33)@P12] - [(RP - S33)@P7]
  Exploratorios (rotular)   : todos os bracos contra U em cada conjunto; efeito do conjunto por braco (@P12 - @P7, @P21 - @P7); tendencia da diferenca RP - S33 em ln K
                              (inclinacao por instancia); robustez de H120 e H121 (t pareado, permutacao de sinais, bootstrap de dois niveis, sem cada conjunto de
                              instancias); uso de operadores (RP contra a mistura); curva de aprendizado do ajuste (seeds de validacao).
Uso: python estudo_aos/analisar_etapa21.py [--ensaio]   (--ensaio fabrica dados para exercitar o codigo; nada e gravado em results/)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo, com_holm  # noqa: E402
from analisar_etapa20 import boot2  # noqa: E402

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
COLS = ["braco", "conjunto", "instancia", "n_customers", "seed", "viavel", "gap_pct", "uso_d", "uso_r"]
POOLS = ("P12", "P21")
# ANALISE PARCIAL (logistica, nao analitica): enquanto o conjunto P12 nao foi executado (a fila o deixou por ultimo), as hipoteses de P21 (H120-H126) sao analisadas sem ele; as de P12
# (H127-H129), a tendencia em ln K e os contrastes de P12 so entram quando etapa21_eval_P12.csv existe. Os resultados de P21 nao mudam quando P12 chega (os dados de P21 estao completos).
PARCIAL = not (Path(__file__).resolve().parent.parent / "results" / "aos_study" / "etapa21_eval_P12.csv").exists()
K = {"P7": 7, "P12": 12, "P21": 21}
KD = {"P7": 4, "P12": 7, "P21": 12}
BRACOS = {"P7": ("U", "RP", "HG", "S33", "RPt", "S65"), "P12": ("U", "RP", "RPt", "S33"), "P21": ("U", "RP", "RPt", "S33", "S65", "HG")}


def referencias():
    e9, e12, e20 = (pd.read_csv(RES / f) for f in ("etapa9_eval.csv", "etapa12_eval.csv", "etapa20_eval_CVRP.csv"))
    sel = lambda d, b, novo: d[d.braco == b][COLS].assign(braco=novo)  # noqa: E731
    return pd.concat([sel(e9, "s9_U", "U@P7"), sel(e9, "s9_RP", "RP@P7"), sel(e9, "s9_hgucb", "HG@P7"), sel(e12, "s12_static", "S33@P7"),
                      sel(e20, "s20_CVRP_RP", "RPt@P7"), sel(e20, "s20_CVRP_static65", "S65@P7")], ignore_index=True)


def ler_novos(pool):
    d = pd.read_csv(RES / f"etapa21_eval_{pool}.csv")
    d["braco"] = d.braco.str.replace(f"s21_{pool}_", "", regex=False) + f"@{pool}"
    return d[COLS]


def fabricar(ref, rng):
    """ENSAIO: bracos novos fabricados a partir de S33@P7 (efeitos conhecidos + ruido) para exercitar o codigo."""
    base = ref[ref.braco == "S33@P7"]
    efeito = {"P12": {"U": 0.30, "RP": 0.05, "RPt": 0.08, "S33": 0.0}, "P21": {"U": 0.60, "RP": 0.10, "RPt": 0.15, "S33": 0.0, "S65": -0.10, "HG": 0.35}}
    out = []
    for p in POOLS:
        for a, e in efeito[p].items():
            d = base.assign(braco=f"{a}@{p}", gap_pct=base.gap_pct + e + rng.normal(0, 0.35, len(base)))
            d["uso_d"] = "|".join(["1"] * KD[p])
            d["uso_r"] = "|".join(["1"] * (K[p] - KD[p]))
            out.append(d)
    return pd.concat(out, ignore_index=True)


def completar(d):
    """RPt = RP se o ajuste nao achou melhor configuracao; S65 = S33 se a continuacao do ES nao achou melhora (P21)."""
    nomes = set(d.braco)
    extra = []
    for p in POOLS:
        if f"RPt@{p}" not in nomes:
            print(f"[{p}] RPt ausente: o ajuste nao achou configuracao melhor que a padrao (RPt = RP); coluna copiada")
            extra.append(d[d.braco == f"RP@{p}"].assign(braco=f"RPt@{p}"))
    if "S65@P21" not in nomes:
        print("[P21] S65 ausente: a continuacao do ES nao achou melhora (S65 = S33); coluna copiada")
        extra.append(d[d.braco == "S33@P21"].assign(braco="S65@P21"))
    return pd.concat([d] + extra, ignore_index=True) if extra else d


def uso_medio(d, braco, col):
    h = d[d.braco == braco]
    m = pd.DataFrame(h[col].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean()
    return (m / m.sum()).values


def main(ensaio=False):
    global K, BRACOS
    parcial = PARCIAL and not ensaio
    PU = ("P21",) if parcial else POOLS                                      # conjuntos novos analisados
    if parcial:
        K = {"P7": 7, "P21": 21}
        BRACOS = {p: BRACOS[p] for p in K}
        print("ANALISE PARCIAL: P12 ainda nao executado; H127-H129 e a tendencia em ln K ficam para depois")
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    ref = referencias()
    novos = fabricar(ref, np.random.default_rng(7)) if ensaio else pd.concat([ler_novos(p) for p in PU], ignore_index=True)
    dados = completar(pd.concat([ref, novos], ignore_index=True))
    t = pivo(dados)
    esperadas = {f"{a}@{p}" for p in K for a in BRACOS[p]}
    assert set(t.columns) == esperadas, (sorted(set(t.columns) ^ esperadas))
    conj = dados[dados.braco == "S33@P7"].groupby("instancia").conjunto.first().reindex(t.index)
    print(f"=== {len(t)} instancias (execucoes inviaveis nos bracos novos: {int((~novos.viavel).sum())}); gap medio (%) ===")
    print(t.mean().round(3).sort_values().to_string())
    # diferencas-em-diferencas (por instancia): inclinacao da diferenca RP - S33 com o tamanho do conjunto
    t = t.copy()
    for p in PU:
        t[f"DiD@{p}"] = (t[f"RP@{p}"] - t[f"S33@{p}"]) - (t["RP@P7"] - t["S33@P7"])
    t["zero"] = 0.0
    C = lambda cid, a, b, g, **kw: contraste(t, cid, a, b, rng, g, **kw)  # noqa: E731

    prim = com_holm([C("H120 RP-S33@P21", "RP@P21", "S33@P21", "primario"),
                     C("H121 DiD(RP-S33)@P21-P7", "DiD@P21", "zero", "primario")], 2)
    sec = com_holm([C("H122 RPt-S33@P21", "RPt@P21", "S33@P21", "secundario"), C("H123 S65-S33@P21", "S65@P21", "S33@P21", "secundario"),
                    C("H124 RP-U@P21", "RP@P21", "U@P21", "secundario"), C("H125 S33-U@P21", "S33@P21", "U@P21", "secundario"),
                    C("H126 HG-U@P21", "HG@P21", "U@P21", "secundario")], 5)
    ter = pd.DataFrame() if parcial else com_holm([C("H127 RP-S33@P12", "RP@P12", "S33@P12", "terciario"), C("H128 RPt-S33@P12", "RPt@P12", "S33@P12", "terciario"),
                                                   C("H129 DiD(RP-S33)@P12-P7", "DiD@P12", "zero", "terciario")], 3)
    expl = []
    for p in K:
        for a in BRACOS[p]:
            if a != "U":
                expl.append(C(f"{a}-U@{p}", f"{a}@{p}", f"U@{p}", "explor"))
    for p in K:                                                              # RP contra a mistura e contra RPt em cada conjunto
        expl.append(C(f"RP-S33@{p}", f"RP@{p}", f"S33@{p}", "explor"))
        expl.append(C(f"RPt-S33@{p}", f"RPt@{p}", f"S33@{p}", "explor"))
        expl.append(C(f"RPt-RP@{p}", f"RPt@{p}", f"RP@{p}", "explor"))
        if "S65" in BRACOS[p]:
            expl.append(C(f"S65-RP@{p}", f"S65@{p}", f"RP@{p}", "explor"))
            expl.append(C(f"S65-RPt@{p}", f"S65@{p}", f"RPt@{p}", "explor"))
        if "HG" in BRACOS[p]:
            expl.append(C(f"HG-RP@{p}", f"HG@{p}", f"RP@{p}", "explor"))
    for p in PU:                                                             # efeito do conjunto sobre cada braco (mesmas instancias, bracos de P7 reutilizados)
        for a in BRACOS[p]:
            expl.append(C(f"{a}@{p}-{a}@P7", f"{a}@{p}", f"{a}@P7", "explor_conjunto"))
    for p in K:
        for c in sorted(set(conj)):
            sub = t[(conj == c).values]
            for a, b in (("RP", "S33"), ("RPt", "S33")):
                expl.append(contraste(sub, f"{a}-{b}@{p}@{c}", f"{a}@{p}", f"{b}@{p}", rng, "explor_set", conjunto=c))
    ex = pd.DataFrame(expl)
    ex["p_holm"], ex["signif_005"] = ex.p_bruto, ex.p_bruto < 0.05
    r = pd.concat([prim, sec, ter, ex], ignore_index=True)
    print("\n=== primarios (Holm m=2), secundarios (Holm m=5), terciarios (Holm m=3) e exploratorios ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))

    # tendencia: inclinacao de (RP - S33) em ln K, por instancia (3 pontos)
    tend = None
    if not parcial:
        lk = np.log([K["P7"], K["P12"], K["P21"]])
        dl = np.column_stack([(t[f"RP@{p}"] - t[f"S33@{p}"]).values for p in ("P7", "P12", "P21")])
        incl = ((dl - dl.mean(axis=1, keepdims=True)) * (lk - lk.mean())).sum(axis=1) / ((lk - lk.mean()) ** 2).sum()
        tend = pd.DataFrame([dict(id="inclinacao de RP-S33 em ln K (pp por unidade de ln K)", n=len(incl), media=float(incl.mean()),
                                  ep=float(incl.std(ddof=1) / np.sqrt(len(incl))), p_wilcoxon=float(stats.wilcoxon(incl).pvalue),
                                  p_t=float(stats.ttest_1samp(incl, 0).pvalue), delta_P7=float(dl[:, 0].mean()), delta_P12=float(dl[:, 1].mean()),
                                  delta_P21=float(dl[:, 2].mean()))])
        print("\n=== exploratorio: tendencia de RP - S33 com o tamanho do conjunto ===")
        print(tend.round(4).to_string(index=False))

    # robustez de H120 e H121 (exploratorio): t pareado, permutacao de sinais, bootstrap de dois niveis, sem cada conjunto de instancias
    runs = {}
    for (arm, inst), gg in dados[dados.viavel].groupby(["braco", "instancia"]).gap_pct:
        runs.setdefault(inst, {})[arm] = gg.values
    rob = []                                                                 # (a DiD nao e um braco: o bootstrap de dois niveis cobre so os contrastes simples)
    for cid, a, b in (("H120", "RP@P21", "S33@P21"), ("H122", "RPt@P21", "S33@P21")) + (() if parcial else (("H127", "RP@P12", "S33@P12"),)):
        dif = np.where(np.abs((t[a] - t[b]).values) < 1e-9, 0.0, (t[a] - t[b]).values)
        sinais = rng.choice([-1.0, 1.0], size=(10000, len(dif)))
        perm = (sinais * dif).mean(axis=1)
        lo2, hi2 = boot2(runs, list(t.index), a, b, rng)
        rob.append(dict(grupo="robustez", id=f"{cid} {a}-{b}", n_inst=len(dif), dif_pp=dif.mean(),
                        p_wilcoxon=stats.wilcoxon(dif).pvalue if np.any(dif != 0) else 1.0, p_t=stats.ttest_1samp(dif, 0).pvalue,
                        p_perm=(1 + int((np.abs(perm) >= abs(dif.mean()) - 1e-12).sum())) / 10001, ic2_lo=lo2, ic2_hi=hi2))
    dif = np.where(np.abs(t["DiD@P21"].values) < 1e-9, 0.0, t["DiD@P21"].values)
    sinais = rng.choice([-1.0, 1.0], size=(10000, len(dif)))
    perm = (sinais * dif).mean(axis=1)
    rob.append(dict(grupo="robustez", id="H121 DiD@P21", n_inst=len(dif), dif_pp=dif.mean(), p_wilcoxon=stats.wilcoxon(dif).pvalue if np.any(dif != 0) else 1.0,
                    p_t=stats.ttest_1samp(dif, 0).pvalue, p_perm=(1 + int((np.abs(perm) >= abs(dif.mean()) - 1e-12).sum())) / 10001))
    for c in sorted(set(conj)):
        sub = t[(conj != c).values]
        for cid, a, b in (("H120", "RP@P21", "S33@P21"), ("H121", "DiD@P21", "zero")):
            x = contraste(sub, f"{cid} sem_{c}", a, b, rng, "robustez_sem_conjunto")
            rob.append(dict(grupo=x["grupo"], id=x["id"], n_inst=x["n_inst"], dif_pp=x["dif_pp"], p_wilcoxon=x["p_bruto"], ic_lo=x["ic_lo"], ic_hi=x["ic_hi"]))
    rob = pd.DataFrame(rob)
    print("\n=== exploratorio: robustez (t pareado, permutacao de sinais, bootstrap de dois niveis, sem cada conjunto) ===")
    print(rob.drop(columns=["grupo"]).round(4).to_string(index=False))

    g = lambda cid: r[r.id.str.startswith(cid + " ")].iloc[0]  # noqa: E731
    neg = lambda x: bool(x.p_holm < 0.05 and x.dif_pp < 0)  # noqa: E731
    H = {
        "H120 RP - S33 em P21 significativamente negativo (primario, Holm m=2)": neg(g("H120")),
        "H121 diferenca-em-diferencas P21 vs P7 de (RP - S33) significativamente negativa (primario, Holm m=2)": neg(g("H121")),
        "H122 RPt - S33 em P21 significativamente negativo (Holm m=5)": neg(g("H122")),
        "H123 S65 - S33 em P21 significativamente negativo (Holm m=5)": neg(g("H123")),
        "H124 RP - U em P21 significativamente negativo (Holm m=5)": neg(g("H124")),
        "H125 S33 - U em P21 significativamente negativo (Holm m=5)": neg(g("H125")),
        "H126 HG - U em P21 significativamente negativo (Holm m=5)": neg(g("H126")),
    }
    if not parcial:
        H.update({"H127 RP - S33 em P12 significativamente negativo (Holm m=3)": neg(g("H127")),
                  "H128 RPt - S33 em P12 significativamente negativo (Holm m=3)": neg(g("H128")),
                  "H129 diferenca-em-diferencas P12 vs P7 de (RP - S33) significativamente negativa (Holm m=3)": neg(g("H129"))})
    print("\n=== hipoteses (pre-registro, Entrada 136) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de {len(H)} (esperado no pre-registro: 4,5 de 10; analise parcial: H127-H129 pendentes)" if parcial else f"  confirmadas: {sum(H.values())} de 10 (esperado no pre-registro: 4,5)")

    print("\n=== exploratorio: uso medio de operadores (fracao de iteracoes), destruicoes | reparos ===")
    uso_rows = []
    for p in K:
        for a in ("RP", "S33", "S65", "RPt", "HG"):
            if f"{a}@{p}" not in set(dados.braco) or a not in BRACOS[p]:
                continue
            ud, ur = uso_medio(dados, f"{a}@{p}", "uso_d"), uso_medio(dados, f"{a}@{p}", "uso_r")
            print(f"  {p:4s} {a:4s} d {np.round(ud, 3).tolist()}  r {np.round(ur, 3).tolist()}")
            uso_rows.append(dict(pool=p, braco=a, uso_d="|".join(f"{x:.4f}" for x in ud), uso_r="|".join(f"{x:.4f}" for x in ur)))
    conc = []
    for p in K:
        rp = np.concatenate([uso_medio(dados, f"RP@{p}", "uso_d"), uso_medio(dados, f"RP@{p}", "uso_r")])
        st = np.concatenate([uso_medio(dados, f"S33@{p}", "uso_d"), uso_medio(dados, f"S33@{p}", "uso_r")])
        rho_, p_ = stats.spearmanr(rp, st)
        conc.append(dict(pool=p, K=K[p], spearman_RP_S33=float(rho_), p=float(p_), dist_variacao_total_d=float(0.5 * np.abs(
            uso_medio(dados, f"RP@{p}", "uso_d") - uso_medio(dados, f"S33@{p}", "uso_d")).sum()), dist_variacao_total_r=float(0.5 * np.abs(
            uso_medio(dados, f"RP@{p}", "uso_r") - uso_medio(dados, f"S33@{p}", "uso_r")).sum())))
    conc = pd.DataFrame(conc)
    print("concordancia entre o uso de RP e os pesos da mistura S33:\n" + conc.round(3).to_string(index=False))

    if not ensaio:
        curva = []
        print("\n=== exploratorio: validacao (instancias de ajuste, seeds 3003-3005) e curva de aprendizado do ajuste ===")
        for p in PU:
            v = pd.read_csv(RES / f"etapa21_validacao_{p}.csv")
            v = v[v.viavel]
            med = v.groupby("braco").gap_pct.mean()
            esc = json.loads((RES / f"etapa21_ajuste_escolhidas_{p}.json").read_text())
            rp_val = float(med[f"s21_{p}_RP"])
            print(f"  {p}: validacao {med.round(4).sort_values().to_dict()}")
            for tr in esc["trajetoria"]:
                if tr["geracao"] in (0, 2, 4, 8, 12, 16):
                    nome = f"val21_{p}_gen{tr['geracao']:02d}"
                    curva.append(dict(pool=p, geracao=tr["geracao"], n_avaliacoes=tr["n_avaliacoes"], gap_ajuste=tr["gap_ajuste"], gap_validacao=float(med[nome]),
                                      gap_validacao_RP=rp_val, gap_validacao_U=float(med[f"s21_{p}_U"])))
            print(f"     escolhidas: RPt mudou={esc['RPt']['mudou']} {esc['RPt']['params']} (gap de ajuste {esc['RPt']['gap_ajuste']:.4f}, padrao {esc['RPt']['gap_padrao']:.4f}); "
                  f"S65 mudou={esc['S65']['mudou']} (gap de ajuste {esc['S33']['gap_ajuste']:.4f} -> {esc['S65']['gap_ajuste']:.4f})")
        curva = pd.DataFrame(curva)
        print(curva.round(4).to_string(index=False))
        r.to_csv(RES / "analise_etapa21.csv", index=False)
        rob.to_csv(RES / "analise_etapa21_robustez.csv", index=False)
        if tend is not None:
            tend.to_csv(RES / "analise_etapa21_tendencia.csv", index=False)
        conc.to_csv(RES / "analise_etapa21_uso_concordancia.csv", index=False)
        pd.DataFrame(uso_rows).to_csv(RES / "analise_etapa21_uso.csv", index=False)
        curva.to_csv(RES / "analise_etapa21_curva.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa21_hipoteses.csv", index=False)


if __name__ == "__main__":
    main(ensaio="--ensaio" in sys.argv)
