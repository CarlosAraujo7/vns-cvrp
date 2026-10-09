#!/usr/bin/env python3
"""
analisar_etapa5.py
==================
Analise da Etapa 5 conforme o PRE-REGISTRO (diario, Entrada 89): atribuicao do
efeito do conjunto de operadores. Confirmatorio: 5 contrastes x 2 estruturas x
2 niveis = 20 testes (Holm m=20). dif = gap(a) - gap(b); negativo = a melhor.

  A1 P8c vs P5   A2 P7e vs P5   A3 P10 vs P8c   A4 P7c vs P10   A5 P7c vs P8c
Referencias: LO: P5 = s3_P5_LO_U, P10 = s3_LO_U; R5: P5 = s5_R5_P5, P10 = s3_R5_U
(bracos da Etapa 3 refeitos com a solucao inicial fixada, Entrada 88).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
N_BOOT = 10000
REF = {"LO": {"P5": "s3_P5_LO_U", "P10": "s3_LO_U", "P8c": "s5_LO_P8c", "P7e": "s5_LO_P7e", "P7c": "s5_LO_P7c"},
       "R5": {"P5": "s5_R5_P5", "P10": "s3_R5_U", "P8c": "s5_R5_P8c", "P7e": "s5_R5_P7e", "P7c": "s5_R5_P7c"}}
PARES = [("A1", "P8c", "P5"), ("A2", "P7e", "P5"), ("A3", "P10", "P8c"), ("A4", "P7c", "P10"), ("A5", "P7c", "P8c")]


def holm(p, m):
    o = np.argsort(p)
    aj = np.empty(len(p))
    c = 0.0
    for r, i in enumerate(o):
        c = max(c, min(1.0, (m - r) * p[i]))
        aj[i] = c
    return aj


def main():
    e5 = pd.read_csv(RES / "etapa5_full.csv")
    e3 = pd.read_csv(RES / "etapa3_full.csv")
    print(f"etapa5: {len(e5)} execucoes, inviaveis {int((~e5.viavel).sum())}")
    d = pd.concat([e5[["braco", "nivel", "instancia", "seed", "gap_pct", "viavel"]],
                   e3[e3.braco.isin(["s3_P5_LO_U", "s3_LO_U", "s3_R5_U"])][["braco", "nivel", "instancia", "seed", "gap_pct", "viavel"]]])
    d = d[d.viavel]
    rng = np.random.default_rng(0)
    linhas = []
    for nv in (1, 5):
        t = d[d.nivel == nv].groupby(["braco", "instancia"]).gap_pct.mean().unstack(0)
        for est in ("LO", "R5"):
            arms = list(REF[est].values())
            print(f"\n=== {est}, E{nv}: gap medio (74 inst.) ===")
            print(t[arms].mean().round(3).sort_values().to_string())
            fr = stats.friedmanchisquare(*[t[a].values for a in arms])
            print(f"Friedman: chi2={fr.statistic:.2f} p={fr.pvalue:.3g}")
            for cid, a, b in PARES:
                x, y = t[REF[est][a]].values, t[REF[est][b]].values
                dif = x - y
                dif = np.where(np.abs(dif) < 1e-9, 0.0, dif)   # empates: diferencas de ponto flutuante (~1e-15) nao sao vitorias/derrotas
                idx = rng.integers(0, len(dif), size=(N_BOOT, len(dif)))
                bm = dif[idx].mean(axis=1)
                lo, hi = np.percentile(bm, [2.5, 97.5])
                lo90, hi90 = np.percentile(bm, [5, 95])
                linhas.append(dict(estrutura=est, nivel=nv, id=cid, a=REF[est][a], b=REF[est][b], dif_pp=dif.mean(),
                                   ic_lo=lo, ic_hi=hi, vit_a=int((dif < 0).sum()), der_a=int((dif > 0).sum()),
                                   emp=int((dif == 0).sum()), p_bruto=stats.wilcoxon(dif).pvalue,
                                   equiv_010=bool(lo90 > -0.1 and hi90 < 0.1)))
    r = pd.DataFrame(linhas)
    r["p_holm"] = holm(r.p_bruto.values, 20)
    r["signif_005"] = r.p_holm < 0.05
    pd.set_option("display.width", 220)
    print("\n=== CONFIRMATORIO (dif = gap(a) - gap(b)) ===")
    print(r.round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa5_confirmatoria.csv", index=False)
    print("\n[exploratorio] fracao das avaliacoes de distancia por operador (indices do pool de cada braco) e shakes:")
    for (br, nv), g in e5.groupby(["braco", "nivel"]):
        arr = np.array([[int(v) for v in s.split("|")] for s in g.calls_por_operador])
        pct = 100 * arr.sum(0) / arr.sum()
        print(f"  {br:12s} E{nv}: shakes={g.shakes.mean():6.1f}  " + " ".join(f"{p:5.1f}" for p in pct))


if __name__ == "__main__":
    main()
