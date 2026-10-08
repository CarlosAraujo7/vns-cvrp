#!/usr/bin/env python3
"""
diag14_significancia_74.py
============================
Diagnóstico #14, decidido pelo pesquisador via consulta neutra ao Gemini
(Entrada 24 -- mesma metodologia das Entradas 9/16/18/20/21/22): agora que
a validação completa nas 74 instâncias terminou (Entrada 24, achado real
#15) e a melhora do candidato vencedor (alpha=0,5/mii=160) sobre o default
atual (alpha=0,5/mii=40) é pequena (~3,2% relativo) e NÃO unânime (37/74
melhoram), aplica um teste de significância estatística formal e PAREADO
sobre o dataset completo -- lacuna que o Gemini apontou duas vezes antes
(Entradas 21, 24) e nunca tinha sido endereçada em nenhum ponto desta
sub-investigação (Entradas 10-24), nem em amostra, nem agora.

Sem NENHUMA computação nova -- só análise dos CSVs já produzidos (Entradas
12/13/23) e já sincronizados. Unidade de pareamento: instância (n=74),
usando a média-de-5-seeds por instância como a observação pareada -- é o
padrão usual na literatura de metaheurísticas para comparar duas variantes
num mesmo benchmark (trata a variância de seed como ruído a ser resumido
por instância antes do teste, não como réplicas independentes, já que as 5
seeds de uma mesma instância não são amostras independentes entre si no
sentido do teste - compartilham a mesma instância/mesmo ótimo/mesma
dificuldade estrutural).

Testes: Wilcoxon signed-rank (não-paramétrico, apropriado dado que gap% não
é necessariamente normal) e teste t pareado (paramétrico, para referência
cruzada). Duas comparações, as duas relevantes para as decisões em aberto:
  1. alpha=0,5/mii=40 (default atual) vs. alpha=0,5/mii=160 (melhor
     candidato) -- a decisão de mudar ou não o default de max_inner_iter.
  2. alpha=5,0 vs. alpha=0,5, pooled nos dois niveis de mii (148 pares) --
     fecha com rigor estatistico a escolha de alpha ja feita nas Entradas
     6/12, agora com o dataset completo.

Uso: python diag14_significancia_74.py
"""
import csv
import sys
from pathlib import Path

from scipy import stats

R = Path(__file__).resolve().parent.parent / "results"
SETS = ["A", "B", "CMT", "X"]

CONFIGS = {
    "alpha=0,5/mii=40 (default atual)": "bandit_vns_validation_progressivo_normalizado_alpha0.5_{s}.csv",
    "alpha=5,0/mii=40": "bandit_vns_validation_progressivo_normalizado_alpha5.0_{s}.csv",
    "alpha=5,0/mii=160": "bandit_vns_validation_progressivo_normalizado_alpha5.0_mii160_{s}.csv",
    "alpha=0,5/mii=160": "bandit_vns_validation_progressivo_normalizado_alpha0.5_mii160_{s}.csv",
}


def carregar_media_por_instancia(padrao):
    dados = {}
    for s in SETS:
        path = R / padrao.format(s=s)
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                dados.setdefault(row["instancia"], []).append(float(row["erro_pct"]))
    assert all(len(v) == 5 for v in dados.values()), "instancia com != 5 seeds"
    return {k: sum(v) / len(v) for k, v in dados.items()}


medias = {nome: carregar_media_por_instancia(padrao) for nome, padrao in CONFIGS.items()}
nomes_instancias = sorted(medias["alpha=0,5/mii=40 (default atual)"].keys())
assert len(nomes_instancias) == 74, f"esperava 74 instancias, achei {len(nomes_instancias)}"
for nome in CONFIGS:
    assert set(medias[nome].keys()) == set(nomes_instancias), f"{nome}: instancias nao batem"

print(f"=== Diagnóstico 14: significância estatística, {len(nomes_instancias)} instâncias (dataset completo) ===\n")


def testar_par(nome_a, nome_b, label):
    a = [medias[nome_a][i] for i in nomes_instancias]
    b = [medias[nome_b][i] for i in nomes_instancias]
    diffs = [x - y for x, y in zip(a, b)]
    n_melhorou = sum(1 for d in diffs if d < -1e-9)
    n_piorou = sum(1 for d in diffs if d > 1e-9)
    n_empatou = len(diffs) - n_melhorou - n_piorou
    media_diff = sum(diffs) / len(diffs)

    # Wilcoxon signed-rank (nao-parametrico, par a par) -- ignora empates exatos automaticamente
    try:
        w_stat, w_p = stats.wilcoxon(a, b, alternative="two-sided")
    except ValueError as e:
        w_stat, w_p = float("nan"), float("nan")
        print(f"  (Wilcoxon nao pode ser computado: {e})")

    # teste t pareado (parametrico, para referencia cruzada)
    t_stat, t_p = stats.ttest_rel(a, b)

    print(f"--- {label} ---")
    print(f"  n={len(diffs)} pares (instancias)  media({nome_b})-media({nome_a})... "
          f"na verdade diff = {nome_a} - {nome_b}")
    print(f"  {nome_a}: melhora em {n_melhorou}, piora em {n_piorou}, empate em {n_empatou}")
    print(f"  diferenca media (primeiro - segundo): {media_diff:+.4f}pp")
    print(f"  Wilcoxon signed-rank: estatistica={w_stat:.2f}  p-valor={w_p:.4f}  "
          f"{'SIGNIFICATIVO (p<0.05)' if w_p < 0.05 else 'NAO significativo (p>=0.05)'}")
    print(f"  teste t pareado:      estatistica={t_stat:.3f}  p-valor={t_p:.4f}  "
          f"{'SIGNIFICATIVO (p<0.05)' if t_p < 0.05 else 'NAO significativo (p>=0.05)'}")
    print()
    return {"w_p": w_p, "t_p": t_p, "media_diff": media_diff}


print("### Comparação 1: default atual (alpha=0,5/mii=40) vs. melhor candidato (alpha=0,5/mii=160) ###")
print("### -- a decisão de mudar ou não max_inner_iter, com alpha=0,5 fixo ###\n")
r1 = testar_par("alpha=0,5/mii=40 (default atual)", "alpha=0,5/mii=160", "alpha=0,5: mii=40 vs mii=160 (n=74)")

print("### Comparação 2: alpha=5,0 vs. alpha=0,5, POOLED nos dois níveis de mii (n=148 pares) ###")
print("### -- fecha com rigor estatístico a escolha de alpha já feita (achado #6, Entradas 6/12) ###\n")
a_pool = [medias["alpha=5,0/mii=40"][i] for i in nomes_instancias] + \
         [medias["alpha=5,0/mii=160"][i] for i in nomes_instancias]
b_pool = [medias["alpha=0,5/mii=40 (default atual)"][i] for i in nomes_instancias] + \
         [medias["alpha=0,5/mii=160"][i] for i in nomes_instancias]
diffs_pool = [x - y for x, y in zip(a_pool, b_pool)]
n_melhorou = sum(1 for d in diffs_pool if d > 1e-9)  # alpha=0,5 melhor (menor) que alpha=5,0
n_piorou = sum(1 for d in diffs_pool if d < -1e-9)
n_empatou = len(diffs_pool) - n_melhorou - n_piorou
w_stat, w_p = stats.wilcoxon(a_pool, b_pool, alternative="two-sided")
t_stat, t_p = stats.ttest_rel(a_pool, b_pool)
print(f"  n={len(diffs_pool)} pares (74 instancias x 2 niveis de mii)")
print(f"  alpha=0,5 melhor em {n_melhorou}, alpha=5,0 melhor em {n_piorou}, empate em {n_empatou}")
print(f"  diferenca media (alpha5,0 - alpha0,5): {sum(diffs_pool)/len(diffs_pool):+.4f}pp")
print(f"  Wilcoxon signed-rank: estatistica={w_stat:.2f}  p-valor={w_p:.6f}  "
      f"{'SIGNIFICATIVO (p<0.05)' if w_p < 0.05 else 'NAO significativo (p>=0.05)'}")
print(f"  teste t pareado:      estatistica={t_stat:.3f}  p-valor={t_p:.6f}  "
      f"{'SIGNIFICATIVO (p<0.05)' if t_p < 0.05 else 'NAO significativo (p>=0.05)'}")
print()

print("### Comparação 3 (contexto -- Entrada 22, achado #14): alpha=5,0 vs alpha=0,5 em mii=160, só CMT (n=4) ###")
print("### -- n pequeno demais para teste formal ter poder, incluído só para registro ###\n")
cmt = [i for i in nomes_instancias if i.startswith("CMT")]
a_cmt = [medias["alpha=5,0/mii=160"][i] for i in cmt]
b_cmt = [medias["alpha=0,5/mii=160"][i] for i in cmt]
print(f"  CMT (n=4): a5,0={a_cmt}  a0,5={b_cmt}")
try:
    w_stat, w_p = stats.wilcoxon(a_cmt, b_cmt, alternative="two-sided")
    print(f"  Wilcoxon (n=4, poder estatistico muito baixo): p-valor={w_p:.4f}")
except ValueError as e:
    print(f"  Wilcoxon nao aplicavel com n=4: {e}")
