#!/usr/bin/env python3
"""
diag2_bt_sensitivity.py
=========================
Diagnóstico #2 dos 3 sugeridos pelo Gemini (Entrada 8 do diário, Achado
Real #3): varia b_t (limiar do filtro de pool, hiperparâmetro reportado
como b_t=0.2 "melhor" no artigo -- Tabela 7/8) para checar se um valor
DIFERENTE do publicado muda o resultado de forma relevante -- o que
sugeriria erro tipográfico/interpretação diferente no artigo original --
ou se o resultado é insensível a b_t (apontando a causa da degradação
para OUTRO lugar, não o parâmetro b_t em si).

Note que b_t tem DOIS efeitos operacionais, não só a condição de parada já
provada matematicamente morta para b_t<=0.2/k=5 (vns.py, "ACHADO REAL"):
também restringe QUAIS operadores entram no pool elegível para
epsilon-greedy/UCB1 a cada seleção (unblocked_pool()). Um b_t maior pode
excluir mais operadores da seleção mesmo sem nunca disparar a parada
antecipada -- por isso testar sensibilidade a b_t continua informativo
mesmo sabendo que a condição de parada em si é inatingível para b_t<=0.2.

Usa progressive_x_linha=True (nossa melhor variante até agora, Entradas
7-8) e a mesma amostra reduzida de tests/compare_progressive_variant.py (5
instâncias do conjunto A, 5 seeds, SEED_BASE=1000 -- comparável seed-a-seed
com a validação completa da Tarefa #178) para manter o custo computacional
razoável (6 valores de b_t x 5 instâncias x 5 seeds = 150 execuções).

Uso: python diag2_bt_sensitivity.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from vns import bandit_vns  # noqa: E402
from bandit import HGUCBBandit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STEPS_MAX = 20
N_SEEDS = 5
SEED_BASE = 1000
VALORES_BT = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

CASOS = [
    ("A-n32-k5", "A", 0.1892),
    ("A-n34-k5", "A", 0.1404),
    ("A-n38-k5", "A", 0.6835),
    ("A-n55-k9", "A", 1.2720),
    ("A-n64-k9", "A", 4.0859),
]


def rodar_uma_semente(inst, cws, bt, seed):
    rng = random.Random(seed)
    # alpha=5.0 fixado explicitamente (Entrada 26): era o default da classe
    # quando este diagnóstico foi rodado/registrado; o default mudou para
    # 0.5 depois (Entradas 6/12/25), então precisa ficar explícito aqui para
    # preservar o comportamento original deste script (b_t não é sobre alpha).
    bandit_obj = HGUCBBandit(n_operators=5, rng=rng, p_hat_mode="q_values", b_t=bt, alpha=5.0)

    inner_por_step = []
    ultimo_inner = [0]

    def on_inner(step, inner_iter, escolhido, melhorou, reward, drift, _u=ultimo_inner):
        _u[0] = inner_iter

    def on_step(step, best_cost, _c=inner_por_step, _u=ultimo_inner):
        _c.append(_u[0])
        _u[0] = 0

    resultado = bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
        bandit=bandit_obj, max_inner_iter=40, progressive_x_linha=True,
        on_inner=on_inner, on_step=on_step,
    )
    return resultado, inner_por_step


def main():
    print("=== Diagnóstico 2: sensibilidade a b_t (progressive_x_linha=True) ===\n")
    resumo_gap = {bt: [] for bt in VALORES_BT}
    resumo_inner = {bt: [] for bt in VALORES_BT}

    for name, set_name, alvo_artigo in CASOS:
        path = ROOT / "instances" / set_name / f"{name}.vrp"
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        print(f"--- {name} (artigo HG-UCB={alvo_artigo:.2f}%) ---")
        for bt in VALORES_BT:
            gaps = []
            inner_iters_todos = []
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                resultado, inner_por_step = rodar_uma_semente(inst, cws, bt, seed)
                viavel, motivo = inst.is_feasible(resultado)
                if not viavel:
                    print(f"    ERRO: b_t={bt} seed={seed} inviável: {motivo}")
                    sys.exit(1)
                custo = inst.solution_cost(resultado)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                gaps.append(gap)
                inner_iters_todos.extend(inner_por_step)

            media = sum(gaps) / len(gaps)
            media_inner = sum(inner_iters_todos) / len(inner_iters_todos)
            resumo_gap[bt].append(media)
            resumo_inner[bt].append(media_inner)
            marcador = " <- artigo" if abs(bt - 0.20) < 1e-9 else ""
            print(f"  b_t={bt:.2f}  gap_medio={media:6.2f}%  inner_iter_medio_por_passo={media_inner:5.1f}{marcador}")
        print()

    print("=== Resumo: média sobre as 5 instâncias, por valor de b_t ===")
    for bt in VALORES_BT:
        media_geral = sum(resumo_gap[bt]) / len(resumo_gap[bt])
        media_inner_geral = sum(resumo_inner[bt]) / len(resumo_inner[bt])
        marcador = "  <- valor reportado no artigo" if abs(bt - 0.20) < 1e-9 else ""
        print(f"  b_t={bt:.2f}: gap_medio={media_geral:6.2f}%  inner_iter_medio={media_inner_geral:5.1f}{marcador}")


if __name__ == "__main__":
    main()
