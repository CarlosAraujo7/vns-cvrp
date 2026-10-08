#!/usr/bin/env python3
"""
construction.py
================
Heurística construtiva Clarke & Wright Savings (CWS), conforme Seção 4.1
e Equação 3 de Kalatzantonakis et al. (2023):

    S_ij = dist(Depot, i) + dist(Depot, j) - dist(i, j)

Implementação "paralela" clássica (Clarke & Wright, 1964): começa com uma
rota depósito-cliente-depósito por cliente; calcula a economia (saving) de
unir cada par de clientes numa única rota; tenta os merges em ordem
decrescente de economia; só funde duas rotas quando (a) ambas ainda são
rotas distintas, (b) a demanda combinada cabe na capacidade, e (c) i e j
estão em posições de EXTREMIDADE de suas rotas (adjacentes ao depósito),
já que só assim o merge elimina de fato uma viagem ao depósito. Só
economias estritamente positivas são consideradas (convenção padrão).
"""
from instance import CVRPInstance  # noqa: F401 (para type hints/documentação)


def _demand(instance, route):
    return sum(instance.demands[c] for c in route)


def clarke_wright_savings(instance):
    """Retorna lista de rotas (cada rota = lista de ids de cliente, sem depósito)."""
    customers = instance.customers
    routes = {c: [c] for c in customers}
    route_of = {c: c for c in customers}

    savings = []
    for idx_i in range(len(customers)):
        for idx_j in range(idx_i + 1, len(customers)):
            i, j = customers[idx_i], customers[idx_j]
            s = (
                instance.dist(instance.depot_id, i)
                + instance.dist(instance.depot_id, j)
                - instance.dist(i, j)
            )
            if s > 0:
                savings.append((s, i, j))
    savings.sort(key=lambda t: t[0], reverse=True)

    next_rid = max(customers) + 1

    for s, i, j in savings:
        ri_id = route_of[i]
        rj_id = route_of[j]
        if ri_id == rj_id:
            continue
        ri = routes[ri_id]
        rj = routes[rj_id]
        if instance.route_demand(ri) + instance.route_demand(rj) > instance.capacity:
            continue

        if ri[-1] == i and rj[0] == j:
            merged = ri + rj
        elif rj[-1] == j and ri[0] == i:
            merged = rj + ri
        elif ri[-1] == i and rj[-1] == j:
            merged = ri + list(reversed(rj))
        elif ri[0] == i and rj[0] == j:
            merged = list(reversed(ri)) + rj
        else:
            # nem i nem j está numa extremidade mesclável -- não é um
            # merge válido de CWS (um deles já está "preso" no meio de
            # uma rota maior).
            continue

        new_rid = next_rid
        next_rid += 1
        routes[new_rid] = merged
        del routes[ri_id]
        del routes[rj_id]
        for c in merged:
            route_of[c] = new_rid

    return list(routes.values())


def _split_giant_tour_exact_k(instance, giant_tour, k_target):
    """Divide um "giant tour" (permutação de TODOS os clientes, ignorando
    capacidade) em EXATAMENTE k_target rotas contíguas, respeitando
    capacidade, minimizando o custo total -- via programação dinâmica
    (técnica clássica de Prins para split de giant tour, adaptada para
    forçar um número EXATO de segmentos em vez do mínimo natural).

    dp[k][i] = menor custo para cobrir os primeiros i clientes do giant
    tour usando EXATAMENTE k rotas. Transição: dp[k][i] = min sobre j<i,
    com demanda(giant_tour[j:i]) <= capacidade, de dp[k-1][j] + custo da
    rota giant_tour[j:i]. O custo da rota é mantido incrementalmente
    conforme j diminui (a rota cresce por um cliente à esquerda por vez),
    O(1) por passo -- então cada nível k custa O(n^2), total O(n^2 * k_target).

    Retorna lista de k_target rotas, ou None se nenhuma divisão em
    exatamente k_target rotas respeitando capacidade existir PARA ESTA
    ORDEM do giant tour (uma ordem diferente pode ou não ter solução;
    quem chama tenta mais de uma ordem antes de desistir)."""
    n = len(giant_tour)
    if k_target > n or k_target < 1:
        return None
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(k_target + 1)]
    parent = [[-1] * (n + 1) for _ in range(k_target + 1)]
    dp[0][0] = 0.0
    depot = instance.depot_id

    for k in range(1, k_target + 1):
        for i in range(1, n + 1):
            demanda = 0
            custo_rota = 0.0
            for j in range(i - 1, -1, -1):
                c = giant_tour[j]
                demanda += instance.demands[c]
                if demanda > instance.capacity + 1e-6:
                    break  # segmentos ainda mais longos só aumentam a demanda
                if j == i - 1:
                    custo_rota = instance.dist(depot, c) + instance.dist(c, depot)
                else:
                    prox = giant_tour[j + 1]
                    custo_rota += (
                        instance.dist(depot, c) + instance.dist(c, prox) - instance.dist(depot, prox)
                    )
                if dp[k - 1][j] < INF and dp[k - 1][j] + custo_rota < dp[k][i]:
                    dp[k][i] = dp[k - 1][j] + custo_rota
                    parent[k][i] = j

    if dp[k_target][n] == INF:
        return None

    rotas = []
    i, k = n, k_target
    while k > 0:
        j = parent[k][i]
        rotas.append(giant_tour[j:i])
        i, k = j, k - 1
    rotas.reverse()
    return rotas


def _nearest_neighbor_tour(instance):
    """Giant tour simples (vizinho mais próximo a partir do depósito,
    ignorando capacidade) -- usado como ordem alternativa quando o giant
    tour "achatado a partir do CWS" não admite divisão em exatamente K
    rotas (ver repair_route_count)."""
    restantes = set(instance.customers)
    atual = instance.depot_id
    tour = []
    while restantes:
        prox = min(restantes, key=lambda c: instance.dist(atual, c))
        tour.append(prox)
        restantes.discard(prox)
        atual = prox
    return tour


def _solve_gap_cpsat(instance, k_target, anchors, time_limit_s=12.0):
    """Um passo de "Problema de Atribuição Generalizada" (Generalized
    Assignment Problem, GAP): dadas k_target âncoras geográficas fixas
    (x,y), atribui cada cliente à âncora mais barata (distância Euclidiana),
    respeitando capacidade e exigindo toda rota não-vazia -- via CP-SAT
    (garante EXATIDÃO da atribuição, ao contrário de um k-means clássico
    "vizinho mais próximo" que ignoraria capacidade). Retorna k_target
    rotas (sem ordem interna definida) ou None se inviável dentro do
    tempo (nesse caso quem chama decide o que fazer)."""
    from ortools.sat.python import cp_model

    customers = instance.customers
    n = len(customers)
    demands = [instance.demands[c] for c in customers]
    capacity = instance.capacity
    custo = [
        [
            int(round(100 * ((instance.coords[c][0] - ax) ** 2 + (instance.coords[c][1] - ay) ** 2) ** 0.5))
            for (ax, ay) in anchors
        ]
        for c in customers
    ]

    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"x_{ci}_{k}") for k in range(k_target)] for ci in range(n)]
    for ci in range(n):
        model.AddExactlyOne(x[ci][k] for k in range(k_target))
    for k in range(k_target):
        model.Add(sum(x[ci][k] * demands[ci] for ci in range(n)) <= capacity)
        model.Add(sum(x[ci][k] for ci in range(n)) >= 1)  # rota nao-vazia
    model.Minimize(sum(custo[ci][k] * x[ci][k] for ci in range(n) for k in range(k_target)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = max(1, __import__("os").cpu_count() or 1)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    rotas = [[] for _ in range(k_target)]
    for ci in range(n):
        for k in range(k_target):
            if solver.Value(x[ci][k]):
                rotas[k].append(customers[ci])
                break
    return rotas


def _centroid(instance, rota):
    xs = [instance.coords[c][0] for c in rota]
    ys = [instance.coords[c][1] for c in rota]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _pack_exact_k_feasible(instance, k_target, giant_tour=None, n_restarts=6, n_rounds=4,
                            time_limit_s=12.0, rng_seed=0):
    """Fallback EXATO para quando _split_giant_tour_exact_k falha em todas
    as ordens de giant tour tentadas -- ver diario_pesquisa.md, Entrada 3,
    casos A-n38-k5, B-n51-k7, B-n57-k7 (empacotamento MUITO apertado, <=
    ~4% de folga, onde nenhuma ordem contígua de um único giant tour
    admite um corte em exatamente k_target segmentos respeitando
    capacidade -- mas uma partição viável EXISTE, provada pela própria
    solução de referência do benchmark).

    HISTÓRICO DA CORREÇÃO (importante -- duas versões anteriores deste
    fallback foram tentadas e descartadas por produzirem partições
    viáveis mas de QUALIDADE RUIM, confirmado rodando o VND completo a
    partir delas, não só inspecionando o custo bruto da construção):
    (1) CP-SAT com objetivo "min-max de posição no giant tour" (achava
    partições viáveis mas nunca provava otimalidade do objetivo dentro de
    60s -- para B-n57-k7, GVNS ficou preso em 26% de erro por 300+ passos,
    e um teste de controle mostrou que a mesma partição (cliente->rota) da
    SOLUÇÃO ÓTIMA, mesmo com a ORDEM interna de cada rota embaralhada,
    o VND recupera exatamente o ótimo -- ou seja, o gargalo é a PARTIÇÃO,
    não a busca local); (2) CP-SAT com uma única rodada de "problema de
    atribuição generalizada" (GAP) a âncoras de k-means padrão (resolve a
    OTIMALIDADE do GAP rapidamente, ~3s, mas ainda ficou preso em 32% de
    erro para B-n57-k7 -- k-means sem noção de capacidade escolhe âncoras
    que não correspondem a nenhuma partição capacitada boa).

    Solução final: "k-means CAPACITADO" (essencialmemte o método clássico
    de Fisher & Jaikumar (1981) para VRP) -- várias reinicializações
    aleatórias de âncoras, cada uma refinada por n_rounds rodadas de
    (resolver GAP via CP-SAT dadas as âncoras atuais -> recalcular cada
    âncora como o CENTROIDE da rota resultante -> repetir). Ao final de
    CADA trajetória de reinício, aplica-se um VND completo (vns.vnd) e
    mede-se o custo REAL pós-VND -- não o custo bruto da construção, que
    se mostrou um preditor ruim de qual reinício leva ao melhor resultado
    após a busca local (testado empiricamente: a reinicialização com
    MENOR custo de construção não foi a que deu o melhor resultado
    pós-VND). A trajetória com menor custo pós-VND é a retornada. Testado
    em B-n57-k7: reduziu o erro de 26.02% (versão 1) / 32.18% (versão 2)
    para 0.61% (esta versão, 6 reinícios x 4 rodadas)."""
    from vns import vnd  # import local -- evita import circular (vns importa neighborhoods, nao construction)

    customers = instance.customers
    rng = __import__("random").Random(rng_seed)
    todas_coords = [instance.coords[c] for c in customers]

    melhor_rotas = None
    melhor_custo = float("inf")
    for reinicio in range(n_restarts):
        anchors = list(rng.sample(todas_coords, k_target))
        rotas = None
        for _rodada in range(n_rounds):
            novas_rotas = _solve_gap_cpsat(instance, k_target, anchors, time_limit_s=time_limit_s)
            if novas_rotas is None:
                break
            rotas = novas_rotas
            anchors = [_centroid(instance, r) if r else anchors[i] for i, r in enumerate(rotas)]
        if rotas is None:
            continue
        rotas_vnd = vnd(instance, [list(r) for r in rotas])
        custo_vnd = instance.solution_cost(rotas_vnd)
        if custo_vnd < melhor_custo:
            melhor_custo = custo_vnd
            melhor_rotas = rotas  # guarda a partição PRE-VND -- o VND completo roda de novo depois em repair_route_count/gvns

    if melhor_rotas is None:
        return None
    if giant_tour is not None:
        pos_de = {c: i for i, c in enumerate(giant_tour)}
        for r in melhor_rotas:
            r.sort(key=lambda c: pos_de.get(c, 0))
    return melhor_rotas


def repair_route_count(instance, routes):
    """Ajusta o número de rotas não-vazias para bater com
    instance.n_vehicles_ref (exatamente, se vehicles_exact, ou pelo menos,
    caso contrário). Ver docstring de instance.py sobre a restrição de K
    -- bug real encontrado e corrigido em 29/08/2026 (diario_pesquisa.md,
    Entrada 3): sem este reparo, CWS podia entregar um número de rotas
    diferente do exigido pela definição formal do problema (Kalatzantonakis
    et al. 2023, Eq. 2c/2d), inflando artificialmente a qualidade da
    solução em instâncias onde "sobrar" um veículo ajuda.

    Para o conjunto X (vehicles_exact=False, restrição é só "PELO MENOS"
    n_vehicles_ref rotas -- ver instance.py), um bug relacionado foi
    identificado e corrigido aqui: a condição de saída antecipada original
    testava só "len(routes) == target", o que faria o código tentar reduzir
    à força o CWS para EXATAMENTE o limite inferior sempre que o CWS
    entregasse MAIS rotas que target -- errado para X, onde "mais rotas que
    o limite inferior" já é uma solução válida e não deve ser mexida.

    Três tentativas de reparo para o caso "reduzir contagem" (a direção
    difícil -- aumentar é sempre fácil, basta separar um cliente da maior
    rota em uma rota nova) já foram feitas e as duas primeiras descartadas
    por RODAREM e falharem em instâncias reais (não por inspeção de código
    -- ver Entrada 3): (1) fundir duas rotas inteiras -- falha quando
    nenhum par cabe junto na capacidade; (2) eliminar a rota de menor
    demanda e reinserir seus clientes um a um nas rotas restantes -- ainda
    falha em A-n33-k6 (nenhuma rota restante tem folga suficiente para o
    cliente isolado da rota eliminada); (3) split de giant tour (Prins)
    forçado a um número EXATO de segmentos via DP -- resolve A-n33-k6 (é
    capaz de re-otimizar a atribuição cliente-rota inteira, não só mover
    clientes isolados) mas ainda falha em A-n38-k5, um caso de
    empacotamento apertado (481/500 = 96.2% de utilização) onde NENHUMA
    ordem contígua testada admite um corte em exatamente 5 segmentos --
    ver _pack_exact_k_feasible. Reparo final: quando o split por DP falha
    para todas as ordens de giant tour tentadas, cai para o fallback exato
    via CP-SAT (_pack_exact_k_feasible), que resolve o bin-packing real
    (sem exigir contiguidade) e é garantido encontrar uma partição viável
    sempre que uma existir (dado tempo suficiente)."""
    routes = [list(r) for r in routes if len(r) > 0]
    target = instance.n_vehicles_ref
    n_atual = len(routes)

    if not target:
        return routes
    if instance.vehicles_exact:
        if n_atual == target:
            return routes
    else:
        if n_atual >= target:
            return routes

    giant_tours = [[c for r in routes for c in r], _nearest_neighbor_tour(instance)]
    for giant_tour in giant_tours:
        resultado = _split_giant_tour_exact_k(instance, giant_tour, target)
        if resultado is not None:
            return resultado

    resultado = _pack_exact_k_feasible(instance, target, giant_tours[0])
    if resultado is not None:
        return resultado

    raise RuntimeError(
        f"{instance.name}: nao foi possivel dividir em exatamente {target} rotas "
        f"respeitando a capacidade {instance.capacity} (tentado giant tour do CWS "
        f"achatado, vizinho-mais-proximo, e fallback exato via CP-SAT)"
    )
