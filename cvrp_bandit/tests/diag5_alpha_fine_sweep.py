#!/usr/bin/env python3
"""
diag5_alpha_fine_sweep.py
============================
Diagnóstico #5, pedido pelo pesquisador (via `AskUserQuestion`, respondendo
à Entrada 10 do diário) como refinamento do diag4_alpha_sensitivity.py: o
diag4 mostrou um padrão monotônico limpo (alpha menor -> menos colapso de
softmax(H) -> gap menor), mas testado só em 5 instâncias do conjunto A --
risco real de overfitting a uma amostra pequena e homogênea. Este script:

  (a) refina a grade de alpha na faixa de interesse identificada no diag4
      (região sem colapso: 0,001-0,02; início do "cotovelo": ~0,05-0,1) --
      agora com 8 valores: 0,001; 0,002; 0,005; 0,008; 0,01; 0,015; 0,02; 0,03;
  (b) amplia a amostra para além do conjunto A: mantém as 5 instâncias de
      A já usadas (comparabilidade com diag2/diag4), acrescenta B-n41-k6
      (pior caso de toda a validação "progressivo", Entrada 8-9) e
      B-n63-k10 (instância maior/mais difícil de B), e X-n101-k25 (menor
      instância do conjunto X, que na comparação fixo/progressivo mostrou
      padrão DIFERENTE de A/B -- Entrada 8, "quase não mostrou o efeito").

Não escolhe um alpha "vencedor" sozinho -- reporta a grade completa,
inclusive por conjunto de instâncias, para o pesquisador decidir com mais
confiança antes de comprometer com uma validação completa de 74 instâncias
(mesmo espírito de todas as decisões anteriores desta linha de pesquisa:
Entradas 5, 7, 8, 9).

Uso: python diag5_alpha_fine_sweep.py
"""
import random
import sys
import time
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
VALORES_ALPHA = [0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03]

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

# GVNS clássico nestas MESMAS instâncias -- fontes: compare_progressive_variant.py
# (Entrada 7, para as 5 de A) e results/gvns_baseline_{B,X}.csv (Entrada 3,
# recuperado por instância para B-n41-k6/B-n63-k10/X-n101-k25) -- usado só
# para a linha de referência impressa ao final, não recomputado aqui.
GVNS_CLASSICO_REF = {
    "A-n32-k5": 5.48, "A-n34-k5": 3.47, "A-n38-k5": 3.23,
    "A-n55-k9": 2.27, "A-n64-k9": 3.73,
}


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
    print("=== Diagnóstico 5: varredura FINA de alpha, amostra AMPLIADA (A+B+X) ===\n")
    t_inicio = time.perf_counter()

    resumo_gap_por_conjunto = {}  # {set_name: {alpha: [medias por instancia]}}
    resumo_colapso_por_conjunto = {}

    for name, set_name, alvo_artigo in CASOS:
        path = ROOT / "instances" / set_name / f"{name}.vrp"
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        ref_classico = GVNS_CLASSICO_REF.get(name)
        ref_str = f"  GVNS_classico={ref_classico:.2f}%" if ref_classico is not None else ""
        print(f"--- {name} (conjunto {set_name}, artigo HG-UCB={alvo_artigo:.2f}%{ref_str}) ---")
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
            resumo_gap_por_conjunto.setdefault(set_name, {}).setdefault(alpha, []).append(media)
            resumo_colapso_por_conjunto.setdefault(set_name, {}).setdefault(alpha, []).append(pct_colapso)
            print(f"  alpha={alpha:6.3f}  gap_medio={media:6.2f}%  pct_iter_colapsada(max_q>0.99)={pct_colapso:5.1f}%")
        print()

    dt_total = time.perf_counter() - t_inicio
    print(f"Tempo total de execução: {dt_total:.1f}s\n")

    print("=== Resumo POR CONJUNTO (média das instâncias testadas daquele conjunto) ===")
    for set_name in resumo_gap_por_conjunto:
        print(f"\n-- Conjunto {set_name} --")
        for alpha in VALORES_ALPHA:
            gaps_conj = resumo_gap_por_conjunto[set_name][alpha]
            colapso_conj = resumo_colapso_por_conjunto[set_name][alpha]
            media_geral = sum(gaps_conj) / len(gaps_conj)
            colapso_geral = sum(colapso_conj) / len(colapso_conj)
            print(f"  alpha={alpha:6.3f}: gap_medio={media_geral:6.2f}%  pct_iter_colapsada={colapso_geral:5.1f}%")

    print("\n=== Resumo GERAL (média sobre TODAS as 8 instâncias, todos os 3 conjuntos) ===")
    for alpha in VALORES_ALPHA:
        todos_gaps = [g for set_name in resumo_gap_por_conjunto for g in resumo_gap_por_conjunto[set_name][alpha]]
        todos_colapsos = [c for set_name in resumo_colapso_por_conjunto for c in resumo_colapso_por_conjunto[set_name][alpha]]
        media_geral = sum(todos_gaps) / len(todos_gaps)
        colapso_geral = sum(todos_colapsos) / len(todos_colapsos)
        marcador = "  <- artigo usa 5,0 (fora desta grade fina)" if False else ""
        print(f"  alpha={alpha:6.3f}: gap_medio={media_geral:6.2f}%  pct_iter_colapsada={colapso_geral:5.1f}%{marcador}")


if __name__ == "__main__":
    main()
