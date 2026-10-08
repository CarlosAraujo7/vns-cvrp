#!/usr/bin/env python3
"""
diag11_colapso_max_inner_iter.py
===========================
Diagnóstico #11, aprovado pelo pesquisador (via `AskUserQuestion`,
respondendo à Entrada 19 -- achado real #11, MISTO): mede %colapso do
softmax (HG-UCB) por configuração de `max_inner_iter`, especificamente
para explicar POR QUE alpha=5,0/mii=160 produziu o menor gap% de todo o
Diagnóstico 10 (3,25%, superando o melhor resultado já visto de
alpha=0,5) -- pergunta deixada em aberto pela Entrada 19: será que o
colapso conhecido do alpha=5,0 (achados reais #4/#6) ainda está presente
em mii=160 e está sendo apenas COMPENSADO pelo volume maior de
tentativas dentro de cada passo, ou de fato diminui?

Escolha independente minha e do Gemini (consultado de forma neutra,
mesma metodologia das Entradas 16/18, sobre a Entrada 19): medir
%colapso é o próximo passo mais informativo por esforço computacional,
antes de considerar mii=160 um candidato de configuração ou de investir
em amostra maior / valores intermediários de mii.

Reusa EXATAMENTE a mesma amostra de 8 instâncias/5 seeds/SEED_BASE=1000
dos diagnósticos 5-10 e as MESMAS 8 configurações (2 alphas x 4 valores
de max_inner_iter) do diagnóstico 10 -- só ACRESCENTA a instrumentação
de %colapso (via on_inner, mesmo critério dos diagnósticos 6/9:
max(q_values) > 0,99) e %drift (mesmo critério do diagnóstico 9), para
permitir comparação DIRETA, linha a linha, com o gap% já registrado na
Entrada 19, sem introduzir nenhuma variável nova além da instrumentação
adicional. Se o gap% recomputado aqui não bater com o da Entrada 19,
isso por si só seria um sinal de alerta (checagem de consistência).

Uso: python diag11_colapso_max_inner_iter.py
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

ALPHAS = [5.0, 0.5]
MAX_INNER_ITER_VALUES = [20, 40, 80, 160]  # 40 = default atual (8*kmax), baseline

CONFIGURACOES = [
    (
        f"alpha={alpha:.1f} / max_inner_iter={mii}" + (" (baseline)" if mii == 40 else ""),
        alpha, mii,
    )
    for alpha in ALPHAS
    for mii in MAX_INNER_ITER_VALUES
]


def rodar_uma_semente(inst, cws, alpha, max_inner_iter, seed):
    rng = random.Random(seed)
    bandit_obj = HGUCBBandit(n_operators=5, rng=rng, p_hat_mode="q_values", alpha=alpha)

    contador = {"colapsadas": 0, "drift": 0, "total": 0}

    def on_inner(step, inner_iter, escolhido, melhorou, reward, drift, _b=bandit_obj, _c=contador):
        _c["total"] += 1
        if max(_b.q_values()) > 0.99:
            _c["colapsadas"] += 1
        if drift:
            _c["drift"] += 1

    resultado = bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
        bandit=bandit_obj, max_inner_iter=max_inner_iter, progressive_x_linha=True,
        reward_positive_fn=reward_positive_normalized, on_inner=on_inner,
    )
    return resultado, contador


def main():
    print("=== Diagnóstico 11: %colapso do softmax por max_inner_iter (segue Entrada 19) ===\n")
    t_inicio = time.perf_counter()

    resumo_gap = {rotulo: {} for rotulo, _, _ in CONFIGURACOES}
    resumo_extra = {rotulo: {"colapsadas": 0, "drift": 0, "total": 0}
                     for rotulo, _, _ in CONFIGURACOES}

    for name, set_name, alvo_artigo in CASOS:
        path = ROOT / "instances" / set_name / f"{name}.vrp"
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        print(f"--- {name} (conjunto {set_name}, artigo HG-UCB={alvo_artigo:.2f}%) ---")
        for rotulo, alpha, mii in CONFIGURACOES:
            gaps = []
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                resultado, contador = rodar_uma_semente(inst, cws, alpha, mii, seed)
                viavel, motivo = inst.is_feasible(resultado)
                if not viavel:
                    print(f"    ERRO: {rotulo} seed={seed} inviável: {motivo}")
                    sys.exit(1)
                custo = inst.solution_cost(resultado)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                gaps.append(gap)
                resumo_extra[rotulo]["colapsadas"] += contador["colapsadas"]
                resumo_extra[rotulo]["drift"] += contador["drift"]
                resumo_extra[rotulo]["total"] += contador["total"]

            media = sum(gaps) / len(gaps)
            resumo_gap[rotulo].setdefault(set_name, []).append(media)
            e = resumo_extra[rotulo]
            pct_col_ate_agora = 100 * e["colapsadas"] / e["total"] if e["total"] else 0.0
            print(f"  {rotulo:36s}  gap_medio={media:6.2f}%  pct_colapso_acumulado={pct_col_ate_agora:5.1f}%")
        print()

    dt_total = time.perf_counter() - t_inicio
    print(f"Tempo total de execução: {dt_total:.1f}s\n")

    print("=== Resumo GERAL (média sobre as 8 instâncias, todos os 3 conjuntos) ===")
    for rotulo, alpha, mii in CONFIGURACOES:
        todos_gaps = [g for lst in resumo_gap[rotulo].values() for g in lst]
        media_geral = sum(todos_gaps) / len(todos_gaps)
        e = resumo_extra[rotulo]
        pct_colapso = 100 * e["colapsadas"] / e["total"] if e["total"] else 0.0
        pct_drift = 100 * e["drift"] / e["total"] if e["total"] else 0.0
        print(f"  {rotulo:36s}  gap_medio={media_geral:6.2f}%  colapso={pct_colapso:5.1f}%  "
              f"drift={pct_drift:5.2f}%  (n={e['total']})")

    print("\n=== Resumo POR CONJUNTO (A vs. B vs. X) ===")
    for rotulo, alpha, mii in CONFIGURACOES:
        linha = f"  {rotulo:36s}"
        for set_name in ("A", "B", "X"):
            lst = resumo_gap[rotulo].get(set_name, [])
            if lst:
                linha += f"  {set_name}={sum(lst)/len(lst):5.2f}%"
        print(linha)


if __name__ == "__main__":
    main()
