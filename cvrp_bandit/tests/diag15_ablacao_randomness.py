#!/usr/bin/env python3
"""
diag15_ablacao_randomness.py
==============================
Diagnóstico #15 -- Fase 1 da Direção 2 (Tarefa #179, diário Entrada 28,
06/09/2026): ablação dos 3 mecanismos de aleatoriedade do HG-UCB
(cold-start, ε-greedy, drift-reset via ADWIN), desenho aprovado pelo
pesquisador via `AskUserQuestion` na mesma data.

Filosofia de ablação (Gemini 2.5 Flash, consulta neutra + refinamento
próprio -- ver Entrada 28 e as docstrings de
`ablation_randomness_bandits.py` e de `drift_reset_enabled` em
src/vns.py): "B" -- remover a intervenção inteiramente, deixar o próximo
mecanismo da cadeia de precedência (cold-start > ε-greedy > UCB1) decidir
sozinho, em vez de substituir a aleatoriedade por uma escolha
determinística que preserva a função do mecanismo removido.

8 configurações, todas na amostra de 16 instâncias já validada no
Diagnóstico 12 (A=6/B=5/CMT=1/X=4 -- proporcional às 74 instâncias
completas), mesmo protocolo (steps_max=20, 5 seeds, SEED_BASE=1000):

  PASSADA PRINCIPAL (alpha=0,5 -- configuração de produção, fechamento
  da Tarefa #178), 6 braços:
    gvns_classico       -- sem bandit nenhum (âncora de referência).
    hgucb_completo      -- HGUCBBandit padrão, os 3 mecanismos ligados
                            (baseline desta ablação).
    sem_cold_start      -- NoColdStartBandit (mecanismo 1 removido).
    sem_epsilon_greedy  -- NoEpsilonGreedyBandit (mecanismo 3 removido).
    sem_drift_reset     -- HGUCBBandit + drift_reset_enabled=False
                            (mecanismo 2 removido).
    sem_nenhum          -- NoColdStartNoEpsilonBandit +
                            drift_reset_enabled=False (os 3 removidos
                            juntos -- ver nota mecânica sobre empates
                            deterministicos na docstring da classe).

  PASSADA SECUNDÁRIA (alpha=5,0 -- só para o mecanismo de drift-reset),
  2 braços:
    hgucb_completo_a5   -- HGUCBBandit em alpha=5,0 (baseline local).
    sem_drift_reset_a5  -- HGUCBBandit em alpha=5,0 + drift_reset_enabled=False.

  Motivo da passada secundária: a Entrada 14 já mediu que o ADWIN dispara
  em só ~0,04% das iterações internas em alpha=0,5 (~0,35 resets por
  execução) contra ~0,69% em alpha=5,0 (~16x mais) -- a passada principal
  sozinha pode não ter eventos suficientes para dizer algo com confiança
  sobre ESTE mecanismo especificamente (não é uma limitação dos outros
  dois, que disparam em toda execução -- cold-start sempre, ε-greedy com
  prob. 0,1 constante). Aprovado explicitamente pelo pesquisador rodar
  isso já nesta fase, em vez de guardar em reserva.

Reward, max_inner_iter e progressive_x_linha idênticos ao fechamento da
Tarefa #178 em todas as 8 configurações (reward_positive_normalized,
mii=40, progressive_x_linha=True) -- só os 3 mecanismos de aleatoriedade
(e, na passada secundária, alpha) variam entre braços, para isolar
exatamente o que está em disputa nesta ablação.

Uso: python diag15_ablacao_randomness.py
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from vns import bandit_vns, gvns  # noqa: E402
from bandit import HGUCBBandit, reward_positive_normalized  # noqa: E402
from ablation_randomness_bandits import (  # noqa: E402
    NoColdStartBandit, NoEpsilonGreedyBandit, NoColdStartNoEpsilonBandit,
)

ROOT = Path(__file__).resolve().parent.parent
STEPS_MAX = 20
N_SEEDS = 5
SEED_BASE = 1000

# Mesma amostra de 16 instâncias do Diagnóstico 12 (Entrada 21) -- ver lá
# para a justificativa da composição (proporcional às 74 completas).
CASOS = [
    ("A-n32-k5", "A", 0.1892),
    ("A-n34-k5", "A", 0.1404),
    ("A-n38-k5", "A", 0.6835),
    ("A-n55-k9", "A", 1.2720),
    ("A-n64-k9", "A", 4.0859),
    ("B-n41-k6", "B", 0.9000),
    ("B-n63-k10", "B", 4.0804),
    ("X-n101-k25", "X", 3.5419),
    ("A-n69-k9", "A", 3.4536),
    ("B-n31-k5", "B", 0.2592),
    ("B-n50-k8", "B", 1.1188),
    ("B-n78-k10", "B", 6.0025),
    ("CMT1", "CMT", 0.0),
    ("X-n106-k14", "X", 2.5488),
    ("X-n134-k13", "X", 7.3767),
    ("X-n157-k13", "X", 3.6606),
]

# (rótulo, classe_do_bandit [None = gvns clássico], alpha, drift_reset_enabled, grupo)
ARMS = [
    ("gvns_classico", None, None, True, "principal"),
    ("hgucb_completo", HGUCBBandit, 0.5, True, "principal"),
    ("sem_cold_start", NoColdStartBandit, 0.5, True, "principal"),
    ("sem_epsilon_greedy", NoEpsilonGreedyBandit, 0.5, True, "principal"),
    ("sem_drift_reset", HGUCBBandit, 0.5, False, "principal"),
    ("sem_nenhum", NoColdStartNoEpsilonBandit, 0.5, False, "principal"),
    ("hgucb_completo_a5", HGUCBBandit, 5.0, True, "stress"),
    ("sem_drift_reset_a5", HGUCBBandit, 5.0, False, "stress"),
]


def caminho_instancia(name, set_name):
    if set_name == "CMT":
        return ROOT / "instances" / "CMT" / f"{name}.vrp"
    return ROOT / "instances" / set_name / f"{name}.vrp"


def rodar(inst, cws, bandit_class, alpha, drift_reset_enabled, seed):
    rng = random.Random(seed)
    if bandit_class is None:
        resultado = gvns(inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX)
        return resultado, None

    bandit_obj = bandit_class(n_operators=5, rng=rng, p_hat_mode="q_values", alpha=alpha)
    contador = {"colapsadas": 0, "drift_sinalizado": 0, "total": 0}

    def on_inner(step, inner_iter, escolhido, melhorou, reward, drift, _b=bandit_obj, _c=contador):
        _c["total"] += 1
        if max(_b.q_values()) > 0.99:
            _c["colapsadas"] += 1
        if drift:
            # bruto do ADWIN -- vale mesmo com drift_reset_enabled=False,
            # ver docstring de drift_reset_enabled em vns.py.
            _c["drift_sinalizado"] += 1

    resultado = bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
        bandit=bandit_obj, max_inner_iter=40, progressive_x_linha=True,
        reward_positive_fn=reward_positive_normalized, on_inner=on_inner,
        drift_reset_enabled=drift_reset_enabled,
    )
    contador["nunca_testado"] = sum(1 for n in bandit_obj.n if n == 0)
    contador["resets_reais"] = contador["drift_sinalizado"] if drift_reset_enabled else 0
    return resultado, contador


def main():
    print("=== Diagnóstico 15: ablação dos 3 mecanismos de aleatoriedade do HG-UCB (Direção 2) ===\n")
    t_inicio = time.perf_counter()

    resumo_gap = {rotulo: {} for rotulo, *_ in ARMS}
    gap_por_instancia = {rotulo: {} for rotulo, *_ in ARMS}
    resumo_extra = {
        rotulo: {"colapsadas": 0, "drift_sinalizado": 0, "drift_reagido": 0,
                  "total": 0, "nunca_testado_soma": 0, "n_execucoes": 0}
        for rotulo, *_ in ARMS
    }

    for name, set_name, alvo_artigo in CASOS:
        path = caminho_instancia(name, set_name)
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        print(f"--- {name} (conjunto {set_name}, artigo HG-UCB={alvo_artigo:.2f}%) ---")
        for rotulo, bandit_class, alpha, drift_reset_enabled, grupo in ARMS:
            gaps = []
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                resultado, contador = rodar(inst, cws, bandit_class, alpha, drift_reset_enabled, seed)
                viavel, motivo = inst.is_feasible(resultado)
                if not viavel:
                    print(f"    ERRO: {rotulo} seed={seed} inviável: {motivo}")
                    sys.exit(1)
                custo = inst.solution_cost(resultado)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                gaps.append(gap)
                if contador is not None:
                    e = resumo_extra[rotulo]
                    e["colapsadas"] += contador["colapsadas"]
                    e["drift_sinalizado"] += contador["drift_sinalizado"]
                    e["drift_reagido"] += contador["resets_reais"]
                    e["total"] += contador["total"]
                    e["nunca_testado_soma"] += contador["nunca_testado"]
                    e["n_execucoes"] += 1

            media = sum(gaps) / len(gaps)
            resumo_gap[rotulo].setdefault(set_name, []).append(media)
            gap_por_instancia[rotulo][name] = media
            print(f"  [{rotulo:19s}] gap_medio={media:6.2f}%")
        print()

    dt_total = time.perf_counter() - t_inicio
    print(f"Tempo total de execução: {dt_total:.1f}s\n")

    print("=== Resumo GERAL — passada principal (alpha=0,5, média sobre as 16 instâncias) ===")
    for rotulo, _, _, _, grupo in ARMS:
        if grupo != "principal":
            continue
        todos = [g for lst in resumo_gap[rotulo].values() for g in lst]
        media_geral = sum(todos) / len(todos)
        print(f"  {rotulo:19s}  gap_medio={media_geral:6.2f}%")

    print("\n=== Resumo POR CONJUNTO (A/B/CMT/X) — passada principal ===")
    for rotulo, _, _, _, grupo in ARMS:
        if grupo != "principal":
            continue
        linha = f"  {rotulo:19s}"
        for set_name in ("A", "B", "CMT", "X"):
            lst = resumo_gap[rotulo].get(set_name, [])
            if lst:
                linha += f"  {set_name}={sum(lst)/len(lst):5.2f}%"
        print(linha)

    print("\n=== Diagnóstico dos mecanismos — passada principal (alpha=0,5) ===")
    for rotulo, _, _, _, grupo in ARMS:
        if grupo != "principal":
            continue
        e = resumo_extra[rotulo]
        if e["n_execucoes"] == 0:
            print(f"  {rotulo:19s}  (sem bandit -- sem diagnóstico de mecanismo)")
            continue
        pct_colapso = 100 * e["colapsadas"] / e["total"] if e["total"] else 0.0
        pct_drift_sinal = 100 * e["drift_sinalizado"] / e["total"] if e["total"] else 0.0
        pct_drift_reagido = 100 * e["drift_reagido"] / e["total"] if e["total"] else 0.0
        media_nunca_testado = e["nunca_testado_soma"] / e["n_execucoes"]
        print(
            f"  {rotulo:19s}  colapso={pct_colapso:5.2f}%  drift_sinalizado={pct_drift_sinal:6.3f}%  "
            f"drift_reagido(resets)={pct_drift_reagido:6.3f}%  media_operadores_nunca_testados={media_nunca_testado:.2f}  "
            f"(n_iter={e['total']}, n_execucoes={e['n_execucoes']})"
        )

    print("\n=== Passada secundária — stress-test drift-reset (alpha=5,0, mesma amostra) ===")
    for rotulo, _, _, _, grupo in ARMS:
        if grupo != "stress":
            continue
        todos = [g for lst in resumo_gap[rotulo].values() for g in lst]
        media_geral = sum(todos) / len(todos)
        e = resumo_extra[rotulo]
        pct_drift_sinal = 100 * e["drift_sinalizado"] / e["total"] if e["total"] else 0.0
        pct_drift_reagido = 100 * e["drift_reagido"] / e["total"] if e["total"] else 0.0
        print(
            f"  {rotulo:19s}  gap_medio={media_geral:6.2f}%  drift_sinalizado={pct_drift_sinal:6.3f}%  "
            f"drift_reagido(resets)={pct_drift_reagido:6.3f}%  (n_iter={e['total']})"
        )

    print("\n=== gap% POR INSTÂNCIA (todas as 16, todos os 8 braços) ===")
    for name, set_name, alvo_artigo in CASOS:
        linha = f"  {name:12s}"
        for rotulo, *_ in ARMS:
            linha += f"  {rotulo}={gap_por_instancia[rotulo][name]:5.2f}%"
        print(linha)


if __name__ == "__main__":
    main()
