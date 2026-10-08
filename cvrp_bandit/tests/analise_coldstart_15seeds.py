#!/usr/bin/env python3
"""
analise_coldstart_15seeds.py
=============================
Combina os 5 seeds de hgucb_completo/sem_cold_start já existentes no CSV
do Diagnóstico 16 (results/diag16_ablacao_74completas.csv, seeds
1000-1004) com os 10 seeds novos do Diagnóstico 17
(results/diag17_coldstart_maisseeds.csv, seeds 1005-1014) -- total 15
seeds por instância, nas 74 instâncias completas -- e refaz o teste de
significância pareado (Wilcoxon + teste t, mesma metodologia das Entradas
25/30/31) para a comparação hgucb_completo vs. sem_cold_start.

Decisão do pesquisador que motivou esta análise (Entrada 31, respondendo
ao achado #24: cold-start pertinho da significância em n=74/5-seeds mas
sem cruzar -- teste t p=0,0708): "mais seeds só para cold-start" em vez de
encerrar a Direção 2 ou ampliar as outras trilhas.

Uso: python analise_coldstart_15seeds.py
"""
import csv
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon, ttest_rel

ROOT = Path(__file__).resolve().parent.parent

ARMS = ["hgucb_completo", "sem_cold_start"]


def carregar(csv_path, seeds_validos=None):
    """Le um CSV no formato braco,conjunto,instancia,seed,gap_pct (colunas
    extras sao ignoradas) e devolve dict[braco][instancia] = lista de gaps."""
    dados = defaultdict(lambda: defaultdict(list))
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            braco = row["braco"]
            if braco not in ARMS:
                continue
            seed = int(row["seed"])
            if seeds_validos is not None and seed not in seeds_validos:
                continue
            dados[braco][row["instancia"]].append(float(row["gap_pct"]))
    return dados


def main():
    d16 = carregar(ROOT / "results" / "diag16_ablacao_74completas.csv", seeds_validos=set(range(1000, 1005)))
    d17 = carregar(ROOT / "results" / "diag17_coldstart_maisseeds.csv", seeds_validos=set(range(1005, 1015)))

    combinado = defaultdict(lambda: defaultdict(list))
    for fonte in (d16, d17):
        for braco in ARMS:
            for inst, gaps in fonte[braco].items():
                combinado[braco][inst].extend(gaps)

    instancias = sorted(combinado["hgucb_completo"].keys())
    print(f"Total de instancias: {len(instancias)}")
    for inst in instancias:
        n_completo = len(combinado["hgucb_completo"][inst])
        n_semcold = len(combinado["sem_cold_start"][inst])
        if n_completo != 15 or n_semcold != 15:
            print(f"  AVISO: {inst} tem {n_completo}/{n_semcold} seeds (esperado 15/15)")

    medias_completo = [sum(combinado["hgucb_completo"][i]) / len(combinado["hgucb_completo"][i]) for i in instancias]
    medias_semcold = [sum(combinado["sem_cold_start"][i]) / len(combinado["sem_cold_start"][i]) for i in instancias]

    dif = [c - s for c, s in zip(medias_completo, medias_semcold)]
    dif_media = sum(dif) / len(dif)
    vitorias_completo = sum(1 for d in dif if d < 0)   # completo MELHOR (gap menor) que sem_cold_start
    vitorias_semcold = sum(1 for d in dif if d > 0)    # sem_cold_start melhor
    empates = sum(1 for d in dif if d == 0)

    stat_w, p_w = wilcoxon(medias_completo, medias_semcold)
    stat_t, p_t = ttest_rel(medias_completo, medias_semcold)

    print(f"\n=== hgucb_completo vs. sem_cold_start -- n=74 instancias, 15 seeds/instancia (5 do diag16 + 10 do diag17) ===")
    print(f"Diferenca media (completo - sem_cold_start): {dif_media:+.4f}pp")
    print(f"Vitorias sem_cold_start (gap menor que completo): {vitorias_semcold}")
    print(f"Vitorias hgucb_completo (gap menor que sem_cold_start): {vitorias_completo}")
    print(f"Empates: {empates}")
    print(f"Wilcoxon pareado: estatistica={stat_w:.4f}  p={p_w:.4f}")
    print(f"Teste t pareado:  estatistica={stat_t:.4f}  p={p_t:.4f}")
    print(f"\nComparacao com achado #24 (n=74, 5 seeds): dif. media era +0,107pp, Wilcoxon p=0,1004, teste t p=0,0708")


if __name__ == "__main__":
    main()
