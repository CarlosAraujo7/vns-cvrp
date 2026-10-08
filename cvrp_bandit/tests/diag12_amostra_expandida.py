#!/usr/bin/env python3
"""
diag12_amostra_expandida.py
===========================
Diagnóstico #12, aprovado pelo pesquisador (via `AskUserQuestion`,
respondendo à Entrada 20 -- achado real #12): "mais uma rodada de
confirmação, em amostra maior, ainda barata" -- para reduzir ruído antes
de decidir entre validar os dois candidatos (alpha=0,5/mii=40 vs.
alpha=5,0/mii=160) na bateria completa de 74 instâncias, testar valores
intermediários de mii, ou manter o default atual.

Dobra a amostra de 8 para 16 instâncias, MANTENDO as 8 originais
(diagnósticos 5-11, permite comparação direta/extensão das tabelas já
publicadas nas Entradas 19-20) e ACRESCENTANDO 8 novas, escolhidas para
aproximar a composição proporcional do conjunto completo de 74 instâncias
(A=26/74=35%, B=23/74=31%, CMT=4/74=5%, X=21/74=28%) -- a amostra de 8
originais nunca incluiu NENHUMA instância CMT, um ponto cego desta
sub-investigação (Entradas 16-20) até agora:

  Originais (8, mantidas): A-n32-k5, A-n34-k5, A-n38-k5, A-n55-k9,
    A-n64-k9, B-n41-k6, B-n63-k10, X-n101-k25
  Novas (8, acrescentadas): A-n69-k9 (maior de A), B-n31-k5 (menor de B),
    B-n50-k8 (meio de B), B-n78-k10 (maior de B), CMT1 (1a instância CMT
    desta sub-investigação), X-n106-k14, X-n134-k13, X-n157-k13 (espalha
    a faixa de X além de n101, sem ir aos extremos mais caros ~172-195,
    para manter o custo computacional moderado)

Composição final: A=6, B=5, CMT=1, X=4 (16 no total) -- razoavelmente
proporcional ao conjunto completo. Valores de referência do artigo
(coluna HG-UCB, Tabelas 10-13) para as 8 novas instâncias extraídos do
dicionário já validado em `tests/run_bandit_vns_validation.py` (mesma
fonte usada desde o início da Tarefa #178, não retranscritos à mão).

Mede EXATAMENTE o mesmo que o diagnóstico 11 (gap%, %colapso do softmax,
%drift) nas MESMAS 8 configurações (2 alphas x 4 valores de
max_inner_iter), mesmos 5 seeds/SEED_BASE=1000 -- única mudança é o
tamanho/composição da amostra de instâncias. Custo estimado ~2,5x o do
diagnóstico 11 (~515s), pela combinação de 2x mais instâncias com um
tamanho médio um pouco maior nas 8 novas -- ainda ordens de grandeza mais
barato que a validação completa de 74 instâncias.

Uso: python diag12_amostra_expandida.py
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

# (nome, conjunto, artigo_HG-UCB%) -- 8 originais (diagnósticos 5-11) + 8 novas
CASOS = [
    # -- originais --
    ("A-n32-k5", "A", 0.1892),
    ("A-n34-k5", "A", 0.1404),
    ("A-n38-k5", "A", 0.6835),
    ("A-n55-k9", "A", 1.2720),
    ("A-n64-k9", "A", 4.0859),
    ("B-n41-k6", "B", 0.9000),
    ("B-n63-k10", "B", 4.0804),
    ("X-n101-k25", "X", 3.5419),
    # -- novas (valores do artigo extraídos de tests/run_bandit_vns_validation.py) --
    ("A-n69-k9", "A", 3.4536),
    ("B-n31-k5", "B", 0.2592),
    ("B-n50-k8", "B", 1.1188),
    ("B-n78-k10", "B", 6.0025),
    ("CMT1", "CMT", 0.0),
    ("X-n106-k14", "X", 2.5488),
    ("X-n134-k13", "X", 7.3767),
    ("X-n157-k13", "X", 3.6606),
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


def caminho_instancia(name, set_name):
    if set_name == "CMT":
        return ROOT / "instances" / "CMT" / f"{name}.vrp"
    return ROOT / "instances" / set_name / f"{name}.vrp"


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
    print("=== Diagnóstico 12: amostra expandida (8 -> 16 instâncias, segue Entrada 20) ===\n")
    t_inicio = time.perf_counter()

    resumo_gap = {rotulo: {} for rotulo, _, _ in CONFIGURACOES}
    resumo_extra = {rotulo: {"colapsadas": 0, "drift": 0, "total": 0}
                     for rotulo, _, _ in CONFIGURACOES}
    # também guardamos por-instância explicitamente (não só acumulado) --
    # aprendizado da nota metodológica da Entrada 20
    gap_por_instancia = {rotulo: {} for rotulo, _, _ in CONFIGURACOES}
    colapso_por_instancia = {rotulo: {} for rotulo, _, _ in CONFIGURACOES}

    for name, set_name, alvo_artigo in CASOS:
        path = caminho_instancia(name, set_name)
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        print(f"--- {name} (conjunto {set_name}, artigo HG-UCB={alvo_artigo:.2f}%) ---")
        for rotulo, alpha, mii in CONFIGURACOES:
            gaps = []
            colapsadas_inst, drift_inst, total_inst = 0, 0, 0
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
                colapsadas_inst += contador["colapsadas"]
                drift_inst += contador["drift"]
                total_inst += contador["total"]

            media = sum(gaps) / len(gaps)
            resumo_gap[rotulo].setdefault(set_name, []).append(media)
            gap_por_instancia[rotulo][name] = media
            pct_colapso_inst = 100 * colapsadas_inst / total_inst if total_inst else 0.0
            colapso_por_instancia[rotulo][name] = pct_colapso_inst
            print(f"  {rotulo:36s}  gap_medio={media:6.2f}%  pct_colapso_desta_instancia={pct_colapso_inst:5.1f}%")
        print()

    dt_total = time.perf_counter() - t_inicio
    print(f"Tempo total de execução: {dt_total:.1f}s\n")

    print("=== Resumo GERAL (média sobre as 16 instâncias, todos os 4 conjuntos) ===")
    for rotulo, alpha, mii in CONFIGURACOES:
        todos_gaps = [g for lst in resumo_gap[rotulo].values() for g in lst]
        media_geral = sum(todos_gaps) / len(todos_gaps)
        e = resumo_extra[rotulo]
        pct_colapso = 100 * e["colapsadas"] / e["total"] if e["total"] else 0.0
        pct_drift = 100 * e["drift"] / e["total"] if e["total"] else 0.0
        print(f"  {rotulo:36s}  gap_medio={media_geral:6.2f}%  colapso={pct_colapso:5.1f}%  "
              f"drift={pct_drift:5.2f}%  (n={e['total']})")

    print("\n=== Resumo GERAL restrito às 8 instâncias ORIGINAIS (comparação direta c/ Entradas 19-20) ===")
    originais = {"A-n32-k5", "A-n34-k5", "A-n38-k5", "A-n55-k9", "A-n64-k9",
                 "B-n41-k6", "B-n63-k10", "X-n101-k25"}
    for rotulo, alpha, mii in CONFIGURACOES:
        gaps_orig = [gap_por_instancia[rotulo][n] for n in originais]
        media_orig = sum(gaps_orig) / len(gaps_orig)
        print(f"  {rotulo:36s}  gap_medio_8_originais={media_orig:6.2f}%")

    print("\n=== Resumo POR CONJUNTO (A vs. B vs. CMT vs. X) ===")
    for rotulo, alpha, mii in CONFIGURACOES:
        linha = f"  {rotulo:36s}"
        for set_name in ("A", "B", "CMT", "X"):
            lst = resumo_gap[rotulo].get(set_name, [])
            if lst:
                linha += f"  {set_name}={sum(lst)/len(lst):5.2f}%"
        print(linha)

    print("\n=== gap% POR INSTÂNCIA (todas as 16, todas as 8 configs) ===")
    for name, set_name, alvo_artigo in CASOS:
        linha = f"  {name:12s}"
        for rotulo, alpha, mii in CONFIGURACOES:
            linha += f"  {rotulo.split('/')[0].strip():11s}mii{mii:<4d}={gap_por_instancia[rotulo][name]:5.2f}%"
        print(linha)


if __name__ == "__main__":
    main()
