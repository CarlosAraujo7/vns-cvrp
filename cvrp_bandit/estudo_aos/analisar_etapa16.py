#!/usr/bin/env python3
"""
analisar_etapa16.py
===================
Analise do Estudo 16 (regime ALNS no CVRPTW, 56 instancias de Solomon, 10 seeds) conforme o PRE-REGISTRO (diario, Entrada 121).
dif = gap(a) - gap(b); negativo = a melhor. Teste de Wilcoxon bilateral pareado por instancia (media das seeds), IC95% bootstrap
(10.000, semente 0), empate |dif| < 1e-9.

  Primario (sem correcao)  : RP - U
  Secundarios (Holm m=6)   : {ucb1, pm, hgucb, ts, exp3, eff} - U
  Terciarios (Holm m=4)    : static - U, fixrep - U, RP - static, RP - fixrep
  Melhor desenho nao adaptativo (Holm m=14): {RP, ucb1, pm, hgucb, ts, exp3, eff} - {static, fixrep}
  Estrutura (Holm m=4)     : ALNS-U - {VND U(P9), VND U(P5), VND ordem de custo crescente, VND static} ao mesmo orcamento
  Exploratorios (rotular)  : por classe de Solomon (C1, C2, R1, R2, RC1, RC2); equivalencia (IC90% em +-0,10pp); MDE; fracao do ganho
                             do RP sobre U capturada pela mistura estatica e pelo reparo fixo; uso dos reparos e das destruicoes.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
N_BOOT = 10000
ADAPT = ["RP", "ucb1", "pm", "hgucb", "ts", "exp3", "eff"]
POL_SEC = ["ucb1", "pm", "hgucb", "ts", "exp3", "eff"]
NAO_ADAPT = ["static", "fixrep"]


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
    return d[d.viavel].groupby(["braco", "instancia"]).gap_pct.mean().unstack(0).dropna()


def com_holm(linhas, m):
    r = pd.DataFrame(linhas)
    r["p_holm"] = holm(r.p_bruto.values, m)
    r["signif_005"] = r.p_holm < 0.05
    return r


def classe(nome):
    """Classe de Solomon: C1, C2, R1, R2, RC1, RC2 (letras do prefixo + primeiro digito)."""
    m = re.match(r"([A-Z]+)(\d)", nome)
    return m.group(1) + m.group(2)


def uso(e, braco, col):
    h = e[e.braco == braco]
    m = pd.DataFrame(h[col].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean()
    return (m / m.sum()).round(3).tolist()


def main():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 240)
    e = pd.read_csv(RES / "etapa16_eval_a.csv")
    rf = pd.read_csv(RES / "etapa16_eval_ref.csv")
    print(f"eval ALNS-TW: {len(e)} execucoes, inviaveis {int((~e.viavel).sum())}; referencias VND: {len(rf)}, "
          f"inviaveis {int((~rf.viavel).sum())}")
    t = pivo(e)
    t.columns = [c.replace("s16a_", "") for c in t.columns]
    print(f"=== ALNS-TW: {len(t)} instancias; gap medio (%) ===")
    print(t.mean().round(3).sort_values().to_string())
    prim = pd.DataFrame([contraste(t, "RP-U", "RP", "U", rng, "primario")])
    prim["p_holm"], prim["signif_005"] = prim.p_bruto, prim.p_bruto < 0.05
    sec = com_holm([contraste(t, f"{a}-U", a, "U", rng, "secundario") for a in POL_SEC], 6)
    ter = com_holm([contraste(t, "static-U", "static", "U", rng, "terciario"), contraste(t, "fixrep-U", "fixrep", "U", rng, "terciario"),
                    contraste(t, "RP-static", "RP", "static", rng, "terciario"), contraste(t, "RP-fixrep", "RP", "fixrep", rng, "terciario")], 4)
    mel = com_holm([contraste(t, f"{a}-{b}", a, b, rng, "melhor_nao_adaptativo") for a in ADAPT for b in NAO_ADAPT], 14)
    tr = pivo(rf)
    tr.columns = [c.replace("s16r_", "vnd_") for c in tr.columns]
    ju = t[["U"]].join(tr, how="inner").rename(columns={"U": "alns_U"})
    est = com_holm([contraste(ju, f"alns_U-{v}", "alns_U", v, rng, "estrutura")
                    for v in ("vnd_U_P9", "vnd_U_P5", "vnd_FOC_custo", "vnd_static")], 4)
    r = pd.concat([prim, sec, ter, mel, est], ignore_index=True)
    print("\n=== primario, secundarios (m=6), terciarios (m=4), melhor nao adaptativo (m=14), estrutura (m=4) ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa16_a.csv", index=False)
    print("\nVND ao mesmo orcamento (gap medio):", tr.mean().round(3).to_dict())

    # exploratorio: por classe de Solomon
    cl = pd.Series({n: classe(n) for n in t.index})
    linhas = []
    for c in sorted(cl.unique()):
        s = t[cl == c]
        for a, b in (("RP", "U"), ("static", "U"), ("fixrep", "U"), ("RP", "static")):
            linhas.append(contraste(s, f"{a}-{b}@{c}", a, b, rng, "explor_classe", classe=c))
    ex = pd.DataFrame(linhas)
    print("\n=== exploratorio: por classe de Solomon (sem correcao; rotular) ===")
    print(ex[["id", "n_inst", "dif_pp", "ic_lo", "ic_hi", "vit_a", "der_a", "p_bruto"]].round(4).to_string(index=False))
    ex.to_csv(RES / "analise_etapa16_classe.csv", index=False)

    # exploratorio: fracao do ganho do RP sobre U capturada pela estatica e pelo reparo fixo (rec = (U - x) / (U - RP))
    rec = []
    for x in ("static", "fixrep"):
        pt = ((t.U - t[x]).mean()) / ((t.U - t.RP).mean())
        bs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(t), len(t))
            den = (t.U.values[i] - t.RP.values[i]).mean()
            bs.append((t.U.values[i] - t[x].values[i]).mean() / den if abs(den) > 1e-9 else np.nan)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rec.append(dict(x=x, captura=pt, ic_lo=lo, ic_hi=hi))
    rec = pd.DataFrame(rec)
    print("\n=== exploratorio: fracao do ganho do RP sobre U capturada ===")
    print(rec.round(3).to_string(index=False))
    rec.to_csv(RES / "analise_etapa16_captura.csv", index=False)

    print("\nuso dos reparos (guloso, regret2, barato) e das destruicoes (aleatoria, pior, relacionada, rota), fracao media:")
    for a in ["U", "RP", "static", "fixrep", "eff", "ts"]:
        b = f"s16a_{a}"
        print(f"  {a:7s} reparos {uso(e, b, 'uso_r')}  destruicoes {uso(e, b, 'uso_d')}")
    print(f"iteracoes medias (U): {e[e.braco == 's16a_U'].iteracoes.mean():.0f}; rotas inviaveis por execucao (U): "
          f"{e[e.braco == 's16a_U'].inviaveis.mean():.3f}")


if __name__ == "__main__":
    main()
