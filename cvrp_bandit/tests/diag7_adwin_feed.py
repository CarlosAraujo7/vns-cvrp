#!/usr/bin/env python3
"""
diag7_adwin_feed.py
======================
Diagnóstico #7, pedido pelo pesquisador (via `AskUserQuestion`, respondendo
à Entrada 14 do diário, achado real #8): "Testar a alternativa na amostra
barata primeiro".

Motivação (Entrada 14): a Seção 5.4 do artigo diz que é "Qvalues"
(=q_values()=softmax(H)) que deveria alimentar o ADWIN para detecção de
concept drift -- não a recompensa bruta da Eq. 12/13, que é o que
`bandit.py` fazia até esta entrada (`self.adwin.update(reward)`), uma
DISCREPÂNCIA real entre o que já documentávamos ter decidido (docstring do
módulo, decisão 2c, escrita em 01/09/2026) e o que o código de fato fazia.
Medimos a taxa de disparo do ADWIN com a alimentação antiga: 0,69% das
iterações (alpha=5,0 normalizado) vs. 0,04% (alpha=0,5 normalizado) -- um
mecanismo vivo e sensível a alpha, não um canto morto.

Testa o novo `adwin_feed_mode="q_value"` (alimenta o ADWIN com pi[chosen],
o q_value do operador ESCOLHIDO antes desta atualização) contra o
`adwin_feed_mode="reward"` (default, comportamento original), cruzado com
os dois candidatos completos da Entrada 13 (alpha=5,0 e alpha=0,5,
recompensa normalizada), na mesma amostra ampliada de 8 instâncias do
diag5/diag6 (5 de A, B-n41-k6, B-n63-k10, X-n101-k25), 5 seeds,
progressive_x_linha=True. Mede gap% E taxa de drift lado a lado -- a
pergunta não é só "isso muda o resultado", é "isso muda o resultado E a
taxa de drift na direção esperada".

Uso: python diag7_adwin_feed.py
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from vns import bandit_vns  # noqa: E402
from bandit import HGUCBBandit, reward_positive_normalized  # noqa: E402

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

# (rótulo, alpha, adwin_feed_mode)
CONFIGURACOES = [
    ("alpha=5,0 / ADWIN=reward (atual)", 5.0, "reward"),
    ("alpha=5,0 / ADWIN=q_value (novo)", 5.0, "q_value"),
    ("alpha=0,5 / ADWIN=reward (atual)", 0.5, "reward"),
    ("alpha=0,5 / ADWIN=q_value (novo)", 0.5, "q_value"),
]


def rodar_uma_semente(inst, cws, alpha, adwin_feed_mode, seed):
    rng = random.Random(seed)
    bandit_obj = HGUCBBandit(n_operators=5, rng=rng, p_hat_mode="q_values",
                              alpha=alpha, adwin_feed_mode=adwin_feed_mode)

    contador = {"drift": 0, "total": 0}

    def on_inner(step, inner_iter, escolhido, melhorou, reward, drift, _c=contador):
        _c["total"] += 1
        if drift:
            _c["drift"] += 1

    resultado = bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
        bandit=bandit_obj, max_inner_iter=40, progressive_x_linha=True,
        reward_positive_fn=reward_positive_normalized, on_inner=on_inner,
    )
    return resultado, contador["drift"], contador["total"]


def main():
    print("=== Diagnóstico 7: alvo do ADWIN (recompensa bruta vs. q_value) ===\n")
    t_inicio = time.perf_counter()

    resumo_gap = {rotulo: {} for rotulo, _, _ in CONFIGURACOES}
    resumo_drift = {rotulo: {"drift": 0, "total": 0} for rotulo, _, _ in CONFIGURACOES}

    for name, set_name, alvo_artigo in CASOS:
        path = ROOT / "instances" / set_name / f"{name}.vrp"
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        print(f"--- {name} (conjunto {set_name}, artigo HG-UCB={alvo_artigo:.2f}%) ---")
        for rotulo, alpha, adwin_feed_mode in CONFIGURACOES:
            gaps = []
            drift_total = 0
            iters_total = 0
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                resultado, drift_count, total = rodar_uma_semente(inst, cws, alpha, adwin_feed_mode, seed)
                viavel, motivo = inst.is_feasible(resultado)
                if not viavel:
                    print(f"    ERRO: {rotulo} seed={seed} inviável: {motivo}")
                    sys.exit(1)
                custo = inst.solution_cost(resultado)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                gaps.append(gap)
                drift_total += drift_count
                iters_total += total

            media = sum(gaps) / len(gaps)
            pct_drift = 100.0 * drift_total / iters_total if iters_total else 0.0
            resumo_gap[rotulo].setdefault(set_name, []).append(media)
            resumo_drift[rotulo]["drift"] += drift_total
            resumo_drift[rotulo]["total"] += iters_total
            print(f"  {rotulo:36s}  gap_medio={media:6.2f}%  taxa_drift={pct_drift:5.2f}%  "
                  f"drifts={drift_total:3d}/{iters_total}")
        print()

    dt_total = time.perf_counter() - t_inicio
    print(f"Tempo total de execução: {dt_total:.1f}s\n")

    print("=== Resumo GERAL (média sobre as 8 instâncias, todos os 3 conjuntos) ===")
    for rotulo, _, _ in CONFIGURACOES:
        todos_gaps = [g for lst in resumo_gap[rotulo].values() for g in lst]
        media_geral = sum(todos_gaps) / len(todos_gaps)
        d = resumo_drift[rotulo]
        pct_drift_geral = 100.0 * d["drift"] / d["total"] if d["total"] else 0.0
        print(f"  {rotulo:36s}  gap_medio={media_geral:6.2f}%  taxa_drift={pct_drift_geral:5.2f}%  "
              f"drifts={d['drift']:4d}/{d['total']}")

    print("\n=== Resumo POR CONJUNTO (A vs. B vs. X) ===")
    for rotulo, _, _ in CONFIGURACOES:
        linha = f"  {rotulo:36s}"
        for set_name in ("A", "B", "X"):
            lst = resumo_gap[rotulo].get(set_name, [])
            if lst:
                linha += f"  {set_name}={sum(lst)/len(lst):5.2f}%"
        print(linha)


if __name__ == "__main__":
    main()
