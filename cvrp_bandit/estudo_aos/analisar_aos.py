#!/usr/bin/env python3
"""
analisar_aos.py
===============
Analise do estudo AOS conforme o PRE-REGISTRO (diario, Entrada 64). Este
script implementa SOMENTE o que la esta fixado como confirmatorio; tudo o
mais e rotulado EXPLORATORIO na saida.

Confirmatorio (por orcamento B):
  - Friedman sobre as 7 politicas confirmatorias (blocos = instancias).
  - Wilcoxon signed-rank bilateral pareado por instancia (media das seeds):
      hgucb_norm vs {gvns_classico, uniforme, ucb1, eps_greedy, prob_matching}
      hgucb_literal vs hgucb_norm
    = 6 testes por B; familia nominal de 18 testes (3 B), correcao de Holm
    sobre a familia INTEIRA.
  - Reportado sempre: diferenca media (pp) com IC95% bootstrap sobre
    instancias (10.000 reamostragens, semente fixa), vitorias/derrotas/
    empates, p bruto e p ajustado.

Se nem todos os orcamentos terminaram, o resultado e marcado PARCIAL e o
ajuste de Holm continua usando m=18 (conservador, nunca mais frouxo que o
pre-registrado).

Uso: python analisar_aos.py [--csv ../results/aos_study/full.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CONFIRM = ["gvns_classico", "uniforme", "ucb1", "eps_greedy", "prob_matching",
           "hgucb_literal", "hgucb_norm"]
PARES = [("hgucb_norm", o) for o in ["gvns_classico", "uniforme", "ucb1", "eps_greedy", "prob_matching"]] \
        + [("hgucb_literal", "hgucb_norm")]
M_FAMILIA = 18
N_BOOT = 10000


def holm(pvals, m=M_FAMILIA):
    """Holm step-down. `pvals`: lista (so os testes disponiveis); o ajuste usa
    o tamanho nominal m da familia (conservador se faltam testes)."""
    ordem = np.argsort(pvals)
    aj = np.empty(len(pvals))
    corrente = 0.0
    for rank, idx in enumerate(ordem):
        corrente = max(corrente, min(1.0, (m - rank) * pvals[idx]))
        aj[idx] = corrente
    return aj


def boot_ic(dif, rng):
    idx = rng.integers(0, len(dif), size=(N_BOOT, len(dif)))
    medias = dif[idx].mean(axis=1)
    return np.percentile(medias, [2.5, 97.5])


def tabela_instancia(d, max_seed=None):
    if max_seed is not None:
        d = d[d.seed < 1000 + max_seed]
    return d.groupby(["politica", "B", "conjunto", "instancia"]).gap_pct.mean().reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results" / "aos_study" / "full.csv"))
    args = ap.parse_args()
    d = pd.read_csv(args.csv)
    print(f"Execucoes lidas: {len(d)}")
    inv = d[~d.viavel]
    print(f"Inviaveis: {len(inv)} ({inv.groupby(['politica','B']).size().to_dict() if len(inv) else 'nenhuma'})")
    d = d[d.viavel]
    orcamentos = sorted(d.B.unique())
    completos = []
    for B in orcamentos:
        sub = d[(d.B == B) & d.politica.isin(CONFIRM)]
        por_inst = sub.groupby("instancia").apply(
            lambda g: (g.politica.nunique() == len(CONFIRM)) and (g.groupby("politica").size().min() == 10),
            include_groups=False)
        if por_inst.all() and len(por_inst) == 74:
            completos.append(B)
    parcial = len(completos) < 3
    print(f"Orcamentos completos (74 inst x 10 seeds x 7 politicas): {completos}"
          f"{'  -> ANALISE PARCIAL' if parcial else ''}\n")

    rng = np.random.default_rng(0)
    linhas = []
    for B in completos:
        t = tabela_instancia(d[d.B == B])
        piv = t.pivot(index="instancia", columns="politica", values="gap_pct")
        print(f"=== B = {B} ===")
        print("Gap medio por politica (media sobre 74 instancias de gap medio por instancia):")
        print(piv[CONFIRM + [c for c in piv.columns if c not in CONFIRM]].mean().round(3).to_string())
        fr = stats.friedmanchisquare(*[piv[c].values for c in CONFIRM])
        print(f"Friedman (7 politicas, n=74): chi2={fr.statistic:.2f}  p={fr.pvalue:.3g}")
        for a, b in PARES:
            x, y = piv[a].values, piv[b].values
            dif = x - y
            w = stats.wilcoxon(x, y)
            lo, hi = boot_ic(dif, rng)
            linhas.append(dict(B=B, a=a, b=b, dif_media_pp=dif.mean(), ic_lo=lo, ic_hi=hi,
                               vit_a=int((dif < 0).sum()), der_a=int((dif > 0).sum()),
                               emp=int((dif == 0).sum()), p_bruto=w.pvalue))
        print()
    if linhas:
        res = pd.DataFrame(linhas)
        res["p_holm"] = holm(res.p_bruto.values)
        res["signif_holm_005"] = res.p_holm < 0.05
        pd.set_option("display.width", 200)
        print(f"=== CONFIRMATORIO {'(PARCIAL: Holm com m=18 mas so testes disponiveis)' if parcial else '(familia completa de 18)'} ===")
        print("dif_media_pp = gap(a) - gap(b); negativo = a melhor.")
        print(res.round(4).to_string(index=False))
        res.to_csv(Path(args.csv).with_name("analise_confirmatoria.csv"), index=False)

    # ---------------- EXPLORATORIO (rotulado) ----------------
    print("\n######## EXPLORATORIO (nao confirmatorio) ########")
    for B in orcamentos:
        t = tabela_instancia(d[d.B == B])
        print(f"\n-- B={B}: gap medio por conjunto --")
        print(t.groupby(["politica", "conjunto"]).gap_pct.mean().unstack().round(3).to_string())
    for B in completos:
        t = tabela_instancia(d[d.B == B])
        piv = t.pivot(index="instancia", columns="politica", values="gap_pct")
        if "hgucb_norm_a5" in piv:
            print(f"\n-- B={B}: isolando a normalizacao (hgucb_literal vs hgucb_norm_a5) e alpha (hgucb_norm_a5 vs hgucb_norm) --")
            for a, b in [("hgucb_literal", "hgucb_norm_a5"), ("hgucb_norm_a5", "hgucb_norm")]:
                dif = piv[a] - piv[b]
                print(f"  {a} - {b}: dif media {dif.mean():+.3f}pp  wilcoxon p={stats.wilcoxon(piv[a], piv[b]).pvalue:.3g}")
        print(f"\n-- B={B}: sensibilidade 5 vs 10 seeds (hgucb_norm vs uniforme) --")
        for ms in (5, 10):
            tt = tabela_instancia(d[d.B == B], max_seed=ms).pivot(index="instancia", columns="politica", values="gap_pct")
            dif = tt["hgucb_norm"] - tt["uniforme"]
            print(f"  {ms} seeds: dif media {dif.mean():+.4f}pp  p={stats.wilcoxon(tt['hgucb_norm'], tt['uniforme']).pvalue:.3g}")
    print("\n-- Uso medio de operadores (% das aplicacoes; ordem: 2-Opt|Move(1)|Swap(1,1)|Reloc(1,0)|Reloc(2,0)) --")
    for (pol, B), g in d.groupby(["politica", "B"]):
        uso = np.array([[int(v) for v in s.split("|")] for s in g.uso_operadores])
        pct = 100 * uso.sum(axis=0) / uso.sum()
        print(f"  {pol:14s} B={B:6d}: " + " ".join(f"{p:5.1f}" for p in pct))


if __name__ == "__main__":
    main()
