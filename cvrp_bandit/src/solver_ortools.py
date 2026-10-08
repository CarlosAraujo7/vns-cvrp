#!/usr/bin/env python3
"""
solver_ortools.py
==================
Wrapper OR-Tools (constraint_solver.routing) para o CVRP, usado como
comparador externo na Direção 1 (comparação justa mesma-máquina --
diario_pesquisa.md, Entradas 32-34).

Trata corretamente a mesma distinção de número de veículos (K) já
documentada e corrigida em instance.py (Entrada 3 do diário): quando
inst.vehicles_exact=True (conjuntos A/B), força EXATAMENTE
inst.n_vehicles_ref veículos com rota não-vazia; quando False (conjuntos
X/CMT), permite um teto maior de veículos (n_vehicles_ref + vehicle_slack)
mas força PELO MENOS n_vehicles_ref não-vazios -- a mesma checagem que
`instance.CVRPInstance.is_feasible` já exige na verificação pós-hoc, agora
imposta como restrição dentro do próprio modelo OR-Tools (não deixada só
para a checagem depois, que só rejeitaria a solução sem re-otimizar).

Custo de arco: OR-Tools exige custos inteiros. Para instâncias com
distância arredondada (A/B/X), usa-se `inst.dist()` diretamente (já
inteiro). Para CMT (distância float, ver instance.py), escala-se por 1000
e arredonda -- só para fins de busca interna; o custo final reportado por
quem chama deve vir de `inst.solution_cost(routes)` (não da função
objetivo interna do OR-Tools), preservando a precisão original.

Uso:
    from solver_ortools import solve_ortools
    routes, wall_time = solve_ortools(inst, time_limit_s=10.0, seed=0)
    viavel, motivo = inst.is_feasible(routes)
"""
import time

from ortools.constraint_solver import routing_enums_pb2, pywrapcp

ESCALA_FLOAT = 1000  # so para instancias com round_distances=False (CMT)


def solve_ortools(inst, time_limit_s, seed=0, vehicle_slack=5):
    """Resolve `inst` via OR-Tools (Guided Local Search), com limite de
    tempo de parede `time_limit_s` (segundos, aceita fração). `seed`
    alimenta o parametro nativo de semente aleatoria do solver (permite
    multiplos reinicios com o mesmo protocolo do nosso HG-UCB -- Entrada
    34 do diario). `vehicle_slack` só importa para instâncias
    vehicles_exact=False (X/CMT): quantos veículos além de n_vehicles_ref
    o modelo pode usar se for vantajoso.

    Retorna (routes, wall_time_s). `routes` é lista de listas de ids de
    cliente reais (mesma convenção de instance.py, sem o depósito nas
    pontas, rotas vazias já removidas). Levanta RuntimeError se o solver
    não achar nenhuma solução dentro do limite de tempo E do número de
    veículos configurado.
    """
    depot = inst.depot_id
    nodes = [depot] + sorted(inst.customers)  # indice OR-Tools 0..n-1 -> id real
    n = len(nodes)

    if inst.vehicles_exact:
        num_vehicles = inst.n_vehicles_ref
        min_used = num_vehicles  # todos os vehicles devem ser usados
    else:
        num_vehicles = inst.n_vehicles_ref + vehicle_slack
        min_used = inst.n_vehicles_ref

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = nodes[manager.IndexToNode(from_index)]
        to_node = nodes[manager.IndexToNode(to_index)]
        d = inst.dist(from_node, to_node)
        if inst.round_distances:
            return int(d)
        return int(round(d * ESCALA_FLOAT))

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        node = nodes[manager.IndexToNode(from_index)]
        return inst.demands[node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, [inst.capacity] * num_vehicles, True, "Capacity"
    )

    solver = routing.solver()

    if inst.vehicles_exact:
        # forca TODOS os `num_vehicles` veiculos a terem rota nao-vazia
        for v in range(num_vehicles):
            solver.Add(routing.NextVar(routing.Start(v)) != routing.End(v))
    else:
        # forca PELO MENOS min_used veiculos com rota nao-vazia (dos num_vehicles disponiveis)
        # IsEqualCstVar devolve um IntVar 0/1 (1 = rota vazia), sem metodo .Not() --
        # em vez de negar, soma-se as "vazias" e exige-se um teto (equivalente a
        # "pelo menos min_used usadas" entre os num_vehicles disponiveis).
        empty_bools = []
        for v in range(num_vehicles):
            end_index = routing.End(v)  # indice constante, nao IntVar -- por isso CstVar, nao Var
            is_empty = solver.IsEqualCstVar(routing.NextVar(routing.Start(v)), end_index)
            empty_bools.append(is_empty)
        solver.Add(solver.Sum(empty_bools) <= num_vehicles - min_used)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromMilliseconds(max(1, int(round(time_limit_s * 1000))))
    try:
        search_parameters.random_seed = seed
    except AttributeError:
        pass  # versao do OR-Tools sem esse campo -- segue sem semente fixa

    t0 = time.perf_counter()
    solution = routing.SolveWithParameters(search_parameters)
    wall_time = time.perf_counter() - t0

    if solution is None:
        raise RuntimeError(
            f"OR-Tools nao achou solucao para {inst.name} "
            f"(num_vehicles={num_vehicles}, min_used={min_used}, "
            f"time_limit={time_limit_s}s)"
        )

    routes = []
    for v in range(num_vehicles):
        index = routing.Start(v)
        route = []
        while not routing.IsEnd(index):
            node = nodes[manager.IndexToNode(index)]
            if node != depot:
                route.append(node)
            index = solution.Value(routing.NextVar(index))
        if route:
            routes.append(route)

    return routes, wall_time
