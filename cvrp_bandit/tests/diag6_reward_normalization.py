#!/usr/bin/env python3
"""
diag6_reward_normalization.py
================================
Diagnóstico #6, pedido pelo pesquisador (via `AskUserQuestion`, respondendo
à Entrada 11 do diário): "Investigar normalização de recompensa primeiro".

Motivação (Entrada 11): a correção "alpha pequeno" (Entradas 9-10) funciona,
mas a faixa "segura" de alpha não é a mesma entre conjuntos de instâncias
(conjunto X colapsa softmax(H) em alphas bem menores que A/B) -- evidência
de que o problema mais profundo pode ser rs=|sol_g-sol_c| (Eq.12) estar em
unidades de custo ABSOLUTAS, não numa escala comparável entre instâncias.

Testa `reward_positive_normalized()` (bandit.py, NOVA, proposta nossa --
divide rs pelo custo da solução corrente, tornando a recompensa uma medida
de melhora RELATIVA/adimensional) via o novo hook `reward_positive_fn` de
`bandit_vns()` (vns.py, adição PURAMENTE ADITIVA -- default None preserva
exatamente o comportamento já validado; ver docstring de bandit_vns()).

Hipótese principal sendo testada: com a recompensa normalizada, o PRÓPRIO
alpha=5.0 do artigo (Tabela 8, sem alterar NENHUM hiperparâmetro) pode
deixar de causar colapso -- o que seria uma correção mais fiel ao artigo
do que reduzir alpha (Entradas 9-10), já que não muda nenhum valor
publicado, só corrige uma normalização não-documentada no pseudocódigo.

Grade testada, mesma amostra ampliada do diag5 (5 instâncias de A, +
B-n41-k6, B-n63-k10, X-n101-k25 -- 8 instâncias, 5 seeds, SEED_BASE=1000,
progressive_x_linha=True):

  ("original",   alpha=0.001)  -- melhor ponto já encontrado (Entradas 9-11)
  ("original",   alpha=5.000)  -- literal do artigo, référence catastrófica
  ("normalized", alpha=0.5/1.0/2.0/5.0/10.0/20.0/50.0) -- nova varredura,
      escala maior pq a recompensa normalizada é ~100x menor em magnitude
      que a original (rs relativo tipicamente 0,001-0,1 vs. rs absoluto
      tipicamente 1-166 -- ver Entrada 9)

Uso: python diag6_reward_normalization.py
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from vns import bandit_vns  # noqa: E402
from bandit import HGUCBBandit, reward_positive, reward_positive_normalized  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STEPS_MAX = 20
N_SEEDS = 5
SEED_BASE = 1000

CASOS = [
    ("A-n32-k5", "A", 0.1892),
    ("A-n34-k5", "A", 0.1404),
    ("A-n38-k5", "A", 0.6835),
    ("A-n55-k9", "A", 1.2720),
    ("A-n64-k9", "A", 4.0859),
    ("B-n41-k6", "B", 0.9000),
    ("B-n63-k10", "B", 4.0804),
    ("X-n101-k25", "X", 3.5419),
]

# (rótulo, modo_recompensa, alpha)
CONFIGURACOES = [
    ("original(alpha=0.001)", "original", 0.001),
    ("original(alpha=5.0)_ARTIGO", "original", 5.0),
    ("normalizado(alpha=0.5)", "normalized", 0.5),
    ("normalizado(alpha=1.0)", "normalized", 1.0),
    ("normalizado(alpha=2.0)", "normalized", 2.0),
    ("normalizado(alpha=5.0)_ARTIGO", "normalized", 5.0),
    ("normalizado(alpha=10.0)", "normalized", 10.0),
    ("normalizado(alpha=20.0)", "normalized", 20.0),
    ("normalizado(alpha=50.0)", "normalized", 50.0),
]


def rodar_uma_semente(inst, cws, modo_recompensa, alpha, seed):
    rng = random.Random(seed)
    bandit_obj = HGUCBBandit(n_operators=5, rng=rng, p_hat_mode="q_values", alpha=alpha)
    reward_fn = reward_positive if modo_recompensa == "original" else reward_positive_normalized

    contador = {"colapsadas": 0, "total": 0}

    def on_inner(step, inner_iter, escolhido, melhorou, reward, drift, _b=bandit_obj, _c=contador):
        _c["total"] += 1
        if max(_b.q_values()) > 0.99:
            _c["colapsadas"] += 1

    resultado = bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
        bandit=bandit_obj, max_inner_iter=40, progressive_x_linha=True,
        reward_positive_fn=reward_fn, on_inner=on_inner,
    )
    return resultado, contador["colapsadas"], contador["total"]


def main():
    print("=== Diagnóstico 6: normalização da recompensa (Eq.12) vs. alpha pequeno ===\n")
    t_inicio = time.perf_counter()

    resumo_gap = {rotulo: {} for rotulo, _, _ in CONFIGURACOES}
    resumo_colapso = {rotulo: {} for rotulo, _, _ in CONFIGURACOES}

    for name, set_name, alvo_artigo in CASOS:
        path = ROOT / "instances" / set_name / f"{name}.vrp"
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        print(f"--- {name} (conjunto {set_name}, artigo HG-UCB={alvo_artigo:.2f}%) ---")
        for rotulo, modo_recompensa, alpha in CONFIGURACOES:
            gaps = []
            colapsadas_total = 0
            iters_total = 0
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                resultado, colapsadas, total = rodar_uma_semente(inst, cws, modo_recompensa, alpha, seed)
                viavel, motivo = inst.is_feasible(resultado)
                if not viavel:
                    print(f"    ERRO: {rotulo} seed={seed} inviável: {motivo}")
                    sys.exit(1)
                custo = inst.solution_cost(resultado)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                gaps.append(gap)
                colapsadas_total += colapsadas
                iters_total += total

            media = sum(gaps) / len(gaps)
            pct_colapso = 100.0 * colapsadas_total / iters_total if iters_total else 0.0
            resumo_gap[rotulo].setdefault(set_name, []).append(media)
            resumo_colapso[rotulo].setdefault(set_name, []).append(pct_colapso)
            print(f"  {rotulo:32s}  gap_medio={media:6.2f}%  pct_iter_colapsada={pct_colapso:5.1f}%")
        print()

    dt_total = time.perf_counter() - t_inicio
    print(f"Tempo total de execução: {dt_total:.1f}s\n")

    print("=== Resumo GERAL (média sobre as 8 instâncias, todos os 3 conjuntos) ===")
    for rotulo, _, _ in CONFIGURACOES:
        todos_gaps = [g for lst in resumo_gap[rotulo].values() for g in lst]
        todos_colapsos = [c for lst in resumo_colapso[rotulo].values() for c in lst]
        media_geral = sum(todos_gaps) / len(todos_gaps)
        colapso_geral = sum(todos_colapsos) / len(todos_colapsos)
        print(f"  {rotulo:32s}  gap_medio={media_geral:6.2f}%  pct_iter_colapsada={colapso_geral:5.1f}%")

    print("\n=== Resumo POR CONJUNTO (A vs. B vs. X) ===")
    for rotulo, _, _ in CONFIGURACOES:
        linha = f"  {rotulo:32s}"
        for set_name in ("A", "B", "X"):
            lst = resumo_gap[rotulo].get(set_name, [])
            if lst:
                linha += f"  {set_name}={sum(lst)/len(lst):5.2f}%"
        print(linha)


if __name__ == "__main__":
    main()
