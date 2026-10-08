#!/usr/bin/env python3
"""
diag4_alpha_sensitivity.py
=============================
Diagnóstico #4, pedido pelo pesquisador (via `AskUserQuestion`, respondendo
à Entrada 9 do diário) como continuação natural do Achado Real #4: os 3
diagnósticos do Gemini convergiram para uma causa mecânica específica --
`alpha=5.0` (Eq.11, hiperparâmetro "melhor" reportado no artigo, Tabela 8)
descasado com a escala real da recompensa (Eq.12 produz valores de 0,5 a
166 nas execuções reais, não normalizados) -- causando colapso quase
instantâneo e quase permanente de softmax(H) (diag1), independente do
valor de b_t testado (diag2), e isolado à peça "gradiente de preferência"
via ablação (diag3: softmax-only é a PIOR variante; UCB1-only, sem
H/softmax nenhum, é a ÚNICA que supera o GVNS clássico).

Este script testa se existe um valor de alpha MENOR, ainda defensável (não
escolhido para "consertar o resultado", mas varrido numa faixa ampla e
reportado por completo, positivo ou negativo), que evite o colapso SEM
alterar nenhuma fórmula do artigo (Eq. 11, 12, 13 permanecem exatamente
como estão -- só o valor do hiperparâmetro alpha muda, mesmo espírito do
diag2 para b_t). Mede, para cada alpha, tanto o gap% quanto a FRAÇÃO de
iterações internas com softmax colapsado (max(q_values)>0.99 logo após o
update) -- para conectar diretamente o resultado ao mecanismo já
identificado no diag1, não só reportar um número de saída.

Usa progressive_x_linha=True (nossa melhor variante -- Entradas 7-8) e a
mesma amostra reduzida de tests/compare_progressive_variant.py /
diag2_bt_sensitivity.py (5 instâncias do conjunto A, 5 seeds,
SEED_BASE=1000) para ficar diretamente comparável aos diagnósticos
anteriores.

Uso: python diag4_alpha_sensitivity.py
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
VALORES_ALPHA = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 5.0]

CASOS = [
    ("A-n32-k5", "A", 0.1892),
    ("A-n34-k5", "A", 0.1404),
    ("A-n38-k5", "A", 0.6835),
    ("A-n55-k9", "A", 1.2720),
    ("A-n64-k9", "A", 4.0859),
]


def rodar_uma_semente(inst, cws, alpha, seed):
    rng = random.Random(seed)
    bandit_obj = HGUCBBandit(n_operators=5, rng=rng, p_hat_mode="q_values", alpha=alpha)

    contador = {"colapsadas": 0, "total": 0}

    def on_inner(step, inner_iter, escolhido, melhorou, reward, drift, _b=bandit_obj, _c=contador):
        _c["total"] += 1
        if max(_b.q_values()) > 0.99:
            _c["colapsadas"] += 1

    resultado = bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
        bandit=bandit_obj, max_inner_iter=40, progressive_x_linha=True,
        on_inner=on_inner,
    )
    return resultado, contador["colapsadas"], contador["total"]


def main():
    print("=== Diagnóstico 4: sensibilidade a alpha (progressive_x_linha=True) ===\n")
    resumo_gap = {a: [] for a in VALORES_ALPHA}
    resumo_colapso = {a: [] for a in VALORES_ALPHA}

    for name, set_name, alvo_artigo in CASOS:
        path = ROOT / "instances" / set_name / f"{name}.vrp"
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        print(f"--- {name} (artigo HG-UCB={alvo_artigo:.2f}%) ---")
        for alpha in VALORES_ALPHA:
            gaps = []
            colapsadas_total = 0
            iters_total = 0
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                resultado, colapsadas, total = rodar_uma_semente(inst, cws, alpha, seed)
                viavel, motivo = inst.is_feasible(resultado)
                if not viavel:
                    print(f"    ERRO: alpha={alpha} seed={seed} inviável: {motivo}")
                    sys.exit(1)
                custo = inst.solution_cost(resultado)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                gaps.append(gap)
                colapsadas_total += colapsadas
                iters_total += total

            media = sum(gaps) / len(gaps)
            pct_colapso = 100.0 * colapsadas_total / iters_total if iters_total else 0.0
            resumo_gap[alpha].append(media)
            resumo_colapso[alpha].append(pct_colapso)
            marcador = " <- artigo" if abs(alpha - 5.0) < 1e-9 else ""
            print(f"  alpha={alpha:6.3f}  gap_medio={media:6.2f}%  pct_iter_colapsada(max_q>0.99)={pct_colapso:5.1f}%{marcador}")
        print()

    print("=== Resumo: média sobre as 5 instâncias, por valor de alpha ===")
    for alpha in VALORES_ALPHA:
        media_geral = sum(resumo_gap[alpha]) / len(resumo_gap[alpha])
        colapso_geral = sum(resumo_colapso[alpha]) / len(resumo_colapso[alpha])
        marcador = "  <- valor reportado no artigo (Tabela 8)" if abs(alpha - 5.0) < 1e-9 else ""
        print(f"  alpha={alpha:6.3f}: gap_medio={media_geral:6.2f}%  pct_iter_colapsada={colapso_geral:5.1f}%{marcador}")


if __name__ == "__main__":
    main()
