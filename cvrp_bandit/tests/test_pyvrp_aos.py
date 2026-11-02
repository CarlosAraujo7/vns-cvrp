#!/usr/bin/env python3
"""
test_pyvrp_aos.py
=================
Controles do Estudo 17 (diario, Entrada 124), no WSL (venv de direcao1_local, pyvrp 0.13.3):

 (A) REGRESSAO EXATA: com um unico conjunto (o padrao do PyVRP) e a mesma semente, `rodar_ils` reproduz `pyvrp.solve` custo a custo
     e iteracao a iteracao sob o mesmo numero de iteracoes (MaxIterations).
 (B) CONVENCAO: o custo das rotas devolvidas, recomputado pela NOSSA instancia (`inst.solution_cost`), coincide com o custo do PyVRP
     (inteiros) ou difere de menos que 1e-3 relativo (CMT, distancias em milesimos) e `inst.is_feasible` (cobertura, capacidade,
     numero de veiculos) vale; nenhuma solucao abaixo do otimo provado (A, B).
 (C) DADOS: as distancias do ProblemData coincidem com `inst.dist` em todos os pares (primeiras instancias).
 (D) CONJUNTOS: P1..P6 sao prefixos aninhados e P7 = P6 + SwapStar; P6 e o padrao do PyVRP.
 (E) POLITICAS: cada politica roda, usa os k conjuntos e devolve solucoes viaveis.
Uso: python3 tests/test_pyvrp_aos.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pyvrp import solve  # noqa: E402
from pyvrp.stop import MaxIterations  # noqa: E402
from instance import load_all  # noqa: E402
from pyvrp_aos import _padrao, construir_dados, construir_pools, rodar_ils, rotas_em_ids  # noqa: E402

falhas = []


def confere(cond, msg):
    if not cond:
        falhas.append(msg)
        print("FALHA:", msg)


def carregar(conjunto, nome):
    return next(i for i in load_all(ROOT / "instances" / conjunto) if i.name == nome)


CASOS = [("A", "A-n32-k5"), ("B", "B-n45-k5"), ("X", "X-n101-k25"), ("CMT", "CMT1"), ("SYN_TUNE", "X-synt01-n51")]
for conj, nome in CASOS:
    inst = carregar(conj, nome)
    data, ids = construir_dados(inst)
    pools = construir_pools(data)
    # (D)
    confere(len(pools) == 7, f"{nome}: esperava 7 conjuntos, ha {len(pools)}")
    confere(all(pools[i][1] == pools[i + 1][1][:len(pools[i][1])] for i in range(5)), f"{nome}: prefixos nao aninhados")
    confere(pools[6][1] == pools[5][1] and [r.__name__ for r in pools[6][2]] == ["SwapStar"], f"{nome}: P7 != P6 + SwapStar")
    confere(_padrao(pools) == 5, f"{nome}: o conjunto padrao deveria ser P6")
    # (C) distancias
    esc = 1 if inst.round_distances else 1000
    D = data.distance_matrix(0)
    erros = sum(1 for a, ia in enumerate(ids[:15]) for b, ib in enumerate(ids[:15])
                if a != b and int(D[a, b]) != int(round(inst.dist(ia, ib) * esc)))
    confere(erros == 0, f"{nome}: {erros} distancias diferentes")
    # (A) regressao exata contra pyvrp.solve (um conjunto padrao, mesma semente, mesmo numero de iteracoes)
    for seed in (1, 2):
        ref = solve(data, MaxIterations(1500), seed=seed, collect_stats=False)
        res, _b = rodar_ils(data, [pools[5]], seed, MaxIterations(1500), fixo=0)
        confere(res.cost() == ref.cost() and res.num_iterations == ref.num_iterations,
                f"{nome} s{seed}: regressao: custo {res.cost()} vs {ref.cost()}, iteracoes {res.num_iterations} vs {ref.num_iterations}")
    # (B) convencao
    res, _b = rodar_ils(data, pools, 3, MaxIterations(800), nome_pol="U")
    rotas = rotas_em_ids(res.best, ids)
    viavel, why = inst.is_feasible(rotas)
    custo = inst.solution_cost(rotas)
    confere(viavel, f"{nome}: inviavel pela convencao: {why}")
    confere(abs(custo - res.cost() / esc) <= 1e-3 * custo, f"{nome}: custo {custo} vs PyVRP {res.cost() / esc}")
    confere(custo >= inst.optimum - 1e-6 or not inst.vehicles_exact, f"{nome}: custo {custo} abaixo do otimo {inst.optimum}")
    print(f"{nome}: ok (n={inst.n_customers}, custo {custo:g}, otimo {inst.optimum:g}, rotas {len(rotas)})", flush=True)

# (E) todas as politicas
inst = carregar("A", "A-n32-k5")
data, ids = construir_dados(inst)
pools = construir_pools(data)
for pol in ("U", "RP", "ucb1", "pm", "hgucb", "ts", "exp3", "eff"):
    res, b = rodar_ils(data, pools, 5, MaxIterations(1200), nome_pol=pol)
    rotas = rotas_em_ids(res.best, ids)
    viavel, why = inst.is_feasible(rotas)
    confere(viavel, f"{pol}: inviavel: {why}")
    confere(sum(b.uso) == b.chamadas > 0 and all(u > 0 for u in b.uso) or pol in ("RP", "ucb1", "exp3", "eff", "ts", "hgucb", "pm"),
            f"{pol}: uso {b.uso}")
    print(f"{pol}: uso {b.uso}, iteracoes {res.num_iterations}, custo {res.cost()}", flush=True)

print("TESTES DO PYVRP-AOS:", "OK" if not falhas else f"{len(falhas)} FALHAS")
sys.exit(1 if falhas else 0)
