"""
pyvrp_aos.py
============
Estudo 17 (diario, Entrada 124): selecao adaptativa do conjunto de operadores da busca local DENTRO de um solver do estado da
arte, a busca local iterada (ILS) do PyVRP 0.13 (a mesma maquinaria que `pyvrp.solve`).

Em cada iteracao do ILS (perturbacao + busca local granular) a politica escolhe UM de k conjuntos ANINHADOS de operadores
(prefixos da lista padrao do PyVRP para o CVRP, mais o SWAP* como operador de rota): P1 = {Exchange10}, P2 = P1 + Exchange20,
P3 = + Exchange11, P4 = + Exchange21, P5 = + Exchange22, P6 = + SwapTails (= o conjunto padrao do PyVRP para o CVRP),
P7 = P6 + SwapStar. A busca local do PyVRP intercala todos os operadores de um conjunto, entao a escolha e do CONJUNTO (forca e
custo da busca), nao da ordem dos operadores.

Recompensa (a mesma do ALNS dos Estudos 9, 12, 13 e 16): sigma1 = 33 se a candidata e a nova melhor solucao viavel, sigma2 = 9 se
melhora o custo penalizado da corrente, 0 caso contrario (sem sigma3: a aceitacao por late acceptance do ILS nao e visivel
a busca). O custo da aplicacao, para a politica de eficiencia, e o tempo de parede da chamada.

Controle de regressao exata: com UM unico conjunto (o padrao) e a mesma semente, `rodar_ils` reproduz `pyvrp.solve` custo a custo
(`tests/test_pyvrp_aos.py`).
"""
import random
import time

from pyvrp import (IteratedLocalSearch, IteratedLocalSearchParams, Model, PenaltyManager, PenaltyParams,
                   RandomNumberGenerator, Solution)
from pyvrp.search import (LocalSearch, NeighbourhoodParams, NODE_OPERATORS, PerturbationManager, PerturbationParams,
                          SwapStar, compute_neighbours)
from pyvrp.stop import MaxIterations, MaxRuntime

from alns_loop import SIGMA1, SIGMA2, construir_politica_alns
from bandit import reward_negative, reward_positive_normalized

ESCALA_FLOAT = 1000      # distancias reais (CMT) viram inteiros em milesimos para o PyVRP; o custo reportado e recomputado em float


def construir_dados(inst):
    """ProblemData do PyVRP a partir da nossa instancia (as MESMAS distancias da convencao: nint para A/B/X, real para CMT).
    A frota e limitada a n_vehicles_ref nas instancias de numero EXATO de veiculos (A, B) e livre nas demais."""
    ids = [inst.depot_id] + sorted(inst.customers)
    escala = 1 if inst.round_distances else ESCALA_FLOAT
    m = Model()
    nveic = inst.n_vehicles_ref if (inst.vehicles_exact and inst.n_vehicles_ref) else inst.n_customers
    dep = m.add_depot(x=inst.coords[ids[0]][0], y=inst.coords[ids[0]][1])
    m.add_vehicle_type(num_available=nveic, capacity=inst.capacity, start_depot=dep, end_depot=dep)
    loc = [dep] + [m.add_client(x=inst.coords[i][0], y=inst.coords[i][1], delivery=inst.demands[i]) for i in ids[1:]]
    for a, ia in enumerate(ids):
        for b, ib in enumerate(ids):
            if a != b:
                m.add_edge(loc[a], loc[b], distance=int(round(inst.dist(ia, ib) * escala)))
    return m.data(), ids


def construir_pools(data):
    """[(nome, operadores de no, operadores de rota)]: prefixos aninhados da lista padrao do PyVRP e P7 = padrao + SwapStar."""
    nos = [op for op in NODE_OPERATORS if op.supports(data)]
    pools = [(f"P{i + 1}", nos[:i + 1], []) for i in range(len(nos))]
    pools.append((f"P{len(nos) + 1}", nos, [SwapStar]))
    return pools


class BuscaAdaptativa:
    """SearchMethod do PyVRP: a cada iteracao nao exaustiva, a politica escolhe a busca local (conjunto de operadores)."""

    def __init__(self, lss, ref, pol=None, modo=None, fixo=None):
        self.lss, self.ref, self.pol, self.modo, self.fixo = lss, ref, pol, modo, fixo
        self.k = len(lss)
        self.todos = list(range(self.k))
        self.best_cost = float("inf")
        self.uso = [0] * self.k
        self.tempo = [0.0] * self.k
        self.chamadas = 0

    def __call__(self, solution, cost_evaluator, exhaustive=False):
        if exhaustive:      # busca exaustiva na nova melhor solucao: sempre o conjunto de referencia
            return self.lss[self.ref](solution, cost_evaluator, exhaustive=True)
        pol = self.pol
        j = self.fixo if pol is None else pol.select_masked(self.todos)
        t0 = time.perf_counter()
        cand = self.lss[j](solution, cost_evaluator)
        dt = time.perf_counter() - t0
        self.chamadas += 1
        self.uso[j] += 1
        self.tempo[j] += dt
        if pol is None:
            return cand
        cur_pen = cost_evaluator.penalised_cost(solution)
        cand_pen = cost_evaluator.penalised_cost(cand)
        cand_viavel = cost_evaluator.cost(cand)       # custo viavel (valor enorme se inviavel)
        sigma = 0.0
        if cand_viavel < self.best_cost:
            sigma, self.best_cost = SIGMA1, cand_viavel
        elif cand_pen < cur_pen:
            sigma = SIGMA2
        if self.modo == "raw":
            rec = sigma
        elif self.modo == "score":
            rec = sigma / SIGMA1
        else:   # hgucb: sucesso = melhora a corrente ou a melhor
            pool = len(pol.unblocked_pool(pol.q_values()))
            if sigma > 0:
                rec = reward_positive_normalized(cur_pen, cand_pen, pool, pol.n[j] + 1)
            else:
                rec = reward_negative(pol.q_values()[j], pool, pol.n[j] + 1)
        if getattr(pol, "usa_custo", False):
            pol.update_custo(j, rec, max(int(dt * 1e9), 1))
        else:
            pol.update(j, rec)
        return cand


def rodar_ils(data, pools, seed, parar, nome_pol=None, params=None, fixo=None, ref=None, coletar=False):
    """ILS do PyVRP com busca adaptativa. `nome_pol` (U, RP, ucb1, pm, hgucb, ts, exp3, eff, static) ou `fixo` (indice de um
    conjunto, politica ausente). `ref`: conjunto da solucao inicial e das buscas exaustivas (padrao: o ultimo prefixo SEM SwapStar,
    isto e, o conjunto padrao do PyVRP; para um braco fixo, o proprio conjunto). Devolve (resultado do PyVRP, busca)."""
    rng = RandomNumberGenerator(seed=seed)
    vizinhos = compute_neighbours(data, NeighbourhoodParams())
    lss = []
    for _nome, nos, rotas in pools:
        ls = LocalSearch(data, rng, vizinhos, PerturbationManager(PerturbationParams()))
        for op in nos:
            ls.add_node_operator(op(data))
        for op in rotas:
            ls.add_route_operator(op(data))
        lss.append(ls)
    if ref is None:
        ref = fixo if fixo is not None else _padrao(pools)
    pm = PenaltyManager.init_from(data, PenaltyParams())
    inicial = lss[ref](Solution.make_random(data, rng), pm.max_cost_evaluator(), exhaustive=True)
    pol, modo = (None, None)
    if nome_pol is not None:
        pol, modo = construir_politica_alns(nome_pol, len(pools), random.Random(seed * 31 + 7), params or {})
    busca = BuscaAdaptativa(lss, ref, pol, modo, fixo)
    busca.best_cost = pm.cost_evaluator().cost(inicial)
    ils = IteratedLocalSearch(data, pm, rng, busca, inicial, IteratedLocalSearchParams())
    res = ils.run(parar, collect_stats=coletar)
    return res, busca


def _padrao(pools):
    """indice do conjunto padrao do PyVRP: o maior prefixo de operadores de no, sem operadores de rota"""
    return max(i for i, p in enumerate(pools) if not p[2])


def rotas_em_ids(sol, ids):
    """rotas da solucao do PyVRP nos ids de cliente da nossa instancia"""
    return [[ids[v] for v in r.visits()] for r in sol.routes() if r.visits()]
