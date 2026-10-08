#!/usr/bin/env python3
"""
diag10_max_inner_iter.py
===========================
Diagnóstico #10, aprovado pelo pesquisador (via `AskUserQuestion`, respondendo
à segunda opinião do Gemini sobre a Entrada 17/18 -- 5 frentes de investigação
sobre reward-shaping/ADWIN esgotadas): "max_inner_iter" -- recomendação de
maior prioridade tanto minha quanto do Gemini (consultado de novo, mesma
metodologia neutra, convergiu independentemente nas mesmas 2 dimensões que eu
já tinha identificado como genuinamente NOVAS: max_inner_iter e p_hat_mode
pós-correção -- e acrescentou C como 3ª).

Motivação: `max_inner_iter` (teto de segurança de iterações internas do loop
"while True" do Algoritmo 3) é um parâmetro NOSSO -- não está no artigo --
introduzido em 01/09/2026 (diário, Entrada 5) porque a condição de parada
ORIGINAL do pseudocódigo ("sem melhora & max(q_values)<b_t") é matematicamente
IMPOSSÍVEL de disparar com k=5 operadores e b_t=0.2 (softmax de 5 valores
sempre soma 1, então max(q)>=1/5=0.2=b_t sempre -- confirmado com 200 mil
amostras aleatórias de H). Consequência estrutural (nunca antes verificada
empiricamente de forma explícita, só deduzida): com a condição original
NUNCA disparando, `max_inner_iter` é a ÚNICA forma do loop interno terminar
-- ou seja, TODO passo externo roda exatamente `max_inner_iter` iterações
internas, sempre, para qualquer instância/seed/config. A docstring de
`bandit_vns()` (vns.py), ao introduzir este parâmetro, já registrava
explicitamente: "a ser calibrado/questionado na validação empírica da
Tarefa #178 se os resultados não forem competitivos" -- nunca foi de fato
calibrado até agora; o valor usado em TODAS as validações e diagnósticos
desta investigação (Entradas 7-18) foi sempre o default, 8*kmax=40, sem
nenhuma variação.

Hipótese: um teto pequeno demais poderia estar interrompendo a busca local
de um passo externo antes do bandit conseguir "compor" melhoras de operadores
diferentes (o mesmo mecanismo por trás do achado real #2, Entrada 8 -- x'
progressivo), ou um teto grande demais poderia estar gastando iterações
internas em rendimentos decrescentes sem valor real, sem nenhuma correção
sobre isso jamais ter sido tentada.

Testa `max_inner_iter` em {20 (metade do atual), 40 (ATUAL, baseline),
80 (2x), 160 (4x)}, cruzado com os dois alphas candidatos completos da
Entrada 13 (5,0 e 0,5), usando a recompensa RECOMENDADA atual (normalizada,
SEM reescala -- achados reais #6/#10 já estabeleceram que reescalar piora),
na mesma amostra de 8 instâncias dos diagnósticos 5-9 (5 de A, B-n41-k6,
B-n63-k10, X-n101-k25), 5 seeds, SEED_BASE=1000, progressive_x_linha=True.
Mede gap% (métrica principal), tempo de execução (custo computacional --
maior max_inner_iter custa proporcionalmente mais tempo, já que o loop
interno SEMPRE roda até o teto -- ver acima), e confirma empiricamente (via
contagem direta) que o número médio de iterações internas por passo externo
é de fato sempre exatamente igual a max_inner_iter, validando a dedução
estrutural do parágrafo acima em vez de só assumi-la.

Uso: python diag10_max_inner_iter.py
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

    contador = {"inner_total": 0, "outer_steps": set()}

    def on_inner(step, inner_iter, escolhido, melhorou, reward, drift, _c=contador):
        _c["inner_total"] += 1
        _c["outer_steps"].add(step)

    resultado = bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
        bandit=bandit_obj, max_inner_iter=max_inner_iter, progressive_x_linha=True,
        reward_positive_fn=reward_positive_normalized, on_inner=on_inner,
    )
    return resultado, contador["inner_total"], len(contador["outer_steps"])


def main():
    print("=== Diagnóstico 10: teto de iterações internas (max_inner_iter) ===\n")
    t_inicio = time.perf_counter()

    resumo_gap = {rotulo: {} for rotulo, _, _ in CONFIGURACOES}
    resumo_extra = {rotulo: {"inner_total": 0, "outer_steps_total": 0, "tempo_s": 0.0}
                     for rotulo, _, _ in CONFIGURACOES}

    for name, set_name, alvo_artigo in CASOS:
        path = ROOT / "instances" / set_name / f"{name}.vrp"
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        print(f"--- {name} (conjunto {set_name}, artigo HG-UCB={alvo_artigo:.2f}%) ---")
        for rotulo, alpha, mii in CONFIGURACOES:
            gaps = []
            t0_cfg = time.perf_counter()
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                resultado, inner_total, outer_steps = rodar_uma_semente(inst, cws, alpha, mii, seed)
                viavel, motivo = inst.is_feasible(resultado)
                if not viavel:
                    print(f"    ERRO: {rotulo} seed={seed} inviável: {motivo}")
                    sys.exit(1)
                custo = inst.solution_cost(resultado)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                gaps.append(gap)
                resumo_extra[rotulo]["inner_total"] += inner_total
                resumo_extra[rotulo]["outer_steps_total"] += outer_steps
            resumo_extra[rotulo]["tempo_s"] += time.perf_counter() - t0_cfg

            media = sum(gaps) / len(gaps)
            resumo_gap[rotulo].setdefault(set_name, []).append(media)
            print(f"  {rotulo:36s}  gap_medio={media:6.2f}%")
        print()

    dt_total = time.perf_counter() - t_inicio
    print(f"Tempo total de execução: {dt_total:.1f}s\n")

    print("=== Resumo GERAL (média sobre as 8 instâncias, todos os 3 conjuntos) ===")
    for rotulo, alpha, mii in CONFIGURACOES:
        todos_gaps = [g for lst in resumo_gap[rotulo].values() for g in lst]
        media_geral = sum(todos_gaps) / len(todos_gaps)
        e = resumo_extra[rotulo]
        media_inner_por_step = e["inner_total"] / e["outer_steps_total"] if e["outer_steps_total"] else 0.0
        print(f"  {rotulo:36s}  gap_medio={media_geral:6.2f}%  "
              f"inner_iter_medio_por_passo={media_inner_por_step:6.2f}  tempo={e['tempo_s']:7.1f}s")

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
