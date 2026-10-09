#!/usr/bin/env python3
"""
analise_robustez.py
===================
Analise de robustez estatistica dos contrastes "politica adaptativa vs.
baseline" das Etapas 1-4 (diario, Entradas 83 e 85). POS-HOC quanto a
escolha dos contrastes e das margens de equivalencia (declarado no
pre-registro da Etapa 4): os testes confirmatorios originais NAO sao
alterados; isto acrescenta o que um revisor pede para sustentar "nao ha
diferenca":

 1. EQUIVALENCIA: IC90% bootstrap (instancias) da diferenca media pareada
    contida em (-delta, +delta)  <=>  TOST a 5%. delta primario = 0,10pp;
    sensibilidade 0,05 / 0,15 / 0,20.
 2. IC95% em DOIS NIVEIS (reamostra instancias e, dentro delas, seeds,
    mantendo o pareamento por seed): corrige a pseudorreplicacao.
 3. Poder: efeito minimo detectavel (MDE) a 80% de poder, alfa=0,05
    bilateral: MDE = (1,96+0,84) * SD(dif entre instancias) / sqrt(n).
 4. Pressuposto: Shapiro-Wilk das diferencas pareadas (justifica o uso do
    Wilcoxon como teste primario) e assimetria.
 5. Decomposicao de variancia do gap por execucao (instancia vs. seed).

Uso: python analise_robustez.py [--etapa4]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
DELTAS = (0.05, 0.10, 0.15, 0.20)
R_BOOT = 5000
SEEDS = list(range(1000, 1010))


def carregar():
    d1 = pd.read_csv(RES / "full.csv").rename(columns={"politica": "arm"})
    d1["estudo"] = "E1"
    d1["nivel"] = d1.B
    d2 = pd.read_csv(RES / "etapa2_full.csv").rename(columns={"braco": "arm"})
    d2["estudo"] = "E2"
    d2["nivel"] = d2.B
    d3 = pd.read_csv(RES / "etapa3_full.csv").rename(columns={"braco": "arm"})
    d3["estudo"] = "E3"
    cols = ["estudo", "arm", "nivel", "instancia", "seed", "gap_pct", "viavel"]
    quadros = [d1[cols], d2[cols], d3[cols]]
    for est, arq in (("E4", "etapa4_eval74.csv"), ("E6", "etapa6_full.csv")):     # Estudos 4 e 6 (se existirem)
        if (RES / arq).exists():
            dx = pd.read_csv(RES / arq).rename(columns={"braco": "arm"})
            dx["estudo"] = est
            quadros.append(dx[cols])
    d = pd.concat(quadros, ignore_index=True)
    return d[d.viavel]


def matriz(d, estudo, nivel, arm):
    """(74, 10): gap por instancia x seed, ordenado."""
    sub = d[(d.nivel == nivel) & (d.arm == arm)]          # os nomes dos bracos sao unicos entre estudos
    if sub.empty:
        return pd.DataFrame()
    piv = sub.pivot(index="instancia", columns="seed", values="gap_pct")
    return piv.sort_index()[SEEDS]


def contrastes():
    c = []
    for B in (800, 4000):                                   # Etapa 1 (loop do Algoritmo 3)
        for o in ("ucb1", "eps_greedy", "prob_matching", "hgucb_norm"):
            c.append(("E1", B, o, "uniforme"))
        c.append(("E1", B, "hgucb_literal", "hgucb_norm"))
    for B in (800, 4000):                                   # Etapa 2 (AOS dentro do VND)
        for o in ("s2b_ucb1", "s2b_eps_greedy", "s2b_prob_matching", "s2b_hgucb_norm"):
            c.append(("E2", B, o, "s2a_U_best_LO"))
    for nv in (1, 5):                                       # Etapa 3 (10 operadores)
        for t in ("LO", "R5"):
            for o in ("ucb1c", "pmc", "hgucbc"):
                c.append(("E3", nv, f"s3_{t}_{o}", f"s3_{t}_U"))
        c.append(("E3", nv, "s3_R5_hgucbc", "s3_R5_hgucb_cego"))
        c.append(("E3", nv, "s3_R5_U", "s3_LO_U"))
    for nv in (1, 5):                                       # Etapa 4 (politicas ajustadas e mistura estatica vs uniforme)
        for o in ("ucb1t", "pmt", "hgucbt", "static"):
            c.append(("E4", nv, f"s4_R5_{o}", "s3_R5_U"))
    for nv in (1, 5):                                       # Etapa 6 (familias adicionais vs uniforme)
        for o in ("exp3", "ts", "swucb", "ap", "eff", "eff2", "ctxts", "ctxeff"):
            c.append(("E6", nv, f"s6_R5_{o}", "s3_R5_U"))
    return c


def analisar(dif_runs, rng):
    """dif_runs: (n_inst, n_seed) diferenca a-b por execucao pareada."""
    n_i, n_s = dif_runs.shape
    dif = dif_runs.mean(axis=1)
    dif = np.where(np.abs(dif) < 1e-9, 0.0, dif)   # empates: diferencas de ponto flutuante nao contam
    media = dif.mean()
    sd = dif.std(ddof=1)
    # IC por bootstrap de instancias
    idx = rng.integers(0, n_i, size=(R_BOOT, n_i))
    med_b = dif[idx].mean(axis=1)
    ic95 = np.percentile(med_b, [2.5, 97.5])
    ic90 = np.percentile(med_b, [5, 95])
    # IC em dois niveis
    ii = rng.integers(0, n_i, size=(R_BOOT, n_i))
    ss = rng.integers(0, n_s, size=(R_BOOT, n_i, n_s))
    med2 = dif_runs[ii[:, :, None], ss].mean(axis=(1, 2))
    ic95_2 = np.percentile(med2, [2.5, 97.5])
    equiv = {d: (ic90[0] > -d and ic90[1] < d) for d in DELTAS}
    mde = (1.96 + 0.84) * sd / np.sqrt(n_i)
    sw = stats.shapiro(dif)
    return dict(dif_pp=media, sd_inst=sd, ic95_lo=ic95[0], ic95_hi=ic95[1], ic90_lo=ic90[0], ic90_hi=ic90[1],
                ic95_2n_lo=ic95_2[0], ic95_2n_hi=ic95_2[1], mde_pp=mde, shapiro_p=sw.pvalue,
                assimetria=stats.skew(dif), p_wilcoxon=stats.wilcoxon(dif).pvalue,
                **{f"equiv_{d:.2f}": equiv[d] for d in DELTAS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa4", action="store_true", help="inclui os contrastes da Etapa 4 (se os CSVs existirem)")
    args = ap.parse_args()
    d = carregar()
    rng = np.random.default_rng(20261011)
    linhas = []
    for estudo, nv, a, b in contrastes():
        A, Bm = matriz(d, estudo, nv, a), matriz(d, estudo, nv, b)
        if A.shape != (74, 10) or Bm.shape != (74, 10) or A.isna().any().any() or Bm.isna().any().any():
            print(f"  (pulado: {estudo} {nv} {a} vs {b}: dados incompletos {A.shape}/{Bm.shape})")
            continue
        r = analisar((A - Bm).values, rng)
        linhas.append(dict(estudo=estudo, nivel=nv, a=a, b=b, **r))
    res = pd.DataFrame(linhas)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["estudo", "nivel", "a", "b", "dif_pp", "ic90_lo", "ic90_hi", "ic95_2n_lo", "ic95_2n_hi", "mde_pp",
            "shapiro_p", "equiv_0.05", "equiv_0.10", "equiv_0.15", "equiv_0.20"]
    print(res[cols].round(3).to_string(index=False))
    res.to_csv(RES / "analise_robustez.csv", index=False)
    print(f"\nEquivalentes em delta=0,10pp: {int(res['equiv_0.10'].sum())} de {len(res)} contrastes; "
          f"em 0,20pp: {int(res['equiv_0.20'].sum())}.")
    print(f"MDE mediano (80% de poder): {res.mde_pp.median():.3f}pp; faixa {res.mde_pp.min():.3f}-{res.mde_pp.max():.3f}.")

    # Decomposicao de variancia (gap por execucao): instancia vs seed, braco U do E3 (E1 e E2)
    print("\nDecomposicao da variancia do gap por execucao (braco s3_R5_U):")
    for nv in (1, 5):
        M = matriz(d, "E3", nv, "s3_R5_U").values
        n_s = M.shape[1]
        dentro = M.var(axis=1, ddof=1).mean()                       # variancia entre seeds (dentro da instancia)
        entre = M.mean(axis=1).var(ddof=1) - dentro / n_s            # componente verdadeiro entre instancias (ANOVA)
        total = entre + dentro
        print(f"  E{nv}: componentes de variancia do gap de UMA execucao: entre instancias {entre:.3f} "
              f"({100*entre/total:.1f}%), entre seeds dentro de instancia {dentro:.3f} ({100*dentro/total:.1f}%) "
              f"[ICC={entre/total:.3f}]; SD de seed por instancia (mediana) {np.median(M.std(axis=1, ddof=1)):.3f}pp; "
              f"erro-padrao da media de {n_s} seeds ~ {np.sqrt(dentro/n_s):.3f}pp")


if __name__ == "__main__":
    main()
