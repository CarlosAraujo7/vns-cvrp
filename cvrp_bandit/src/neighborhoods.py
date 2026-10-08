#!/usr/bin/env python3
"""
neighborhoods.py
================
Os 5 operadores de vizinhança de Kalatzantonakis et al. (2023), Tabela 2 /
Figura 2:

  Intra-rota:
    2-Opt        -- remove duas arestas da MESMA rota e reconecta invertendo
                     o segmento entre elas.
    Move(1)      -- move um cliente para uma nova posição na MESMA rota.
  Inter-rota:
    Swap(1,1)    -- troca as posições de um cliente de cada uma de DUAS rotas.
    Relocate(1,0)-- realoca UM cliente de uma rota para outra.
    Relocate(2,0)-- realoca DOIS clientes CONSECUTIVOS de uma rota para outra.

Cada operador expõe duas funções, para os dois usos que o Algoritmo 1
(GVNS) e o Algoritmo 3 (Bandit VNS) do artigo fazem de uma vizinhança:

  best_move(instance, routes) -> (melhorou: bool, novas_rotas)
      Busca por MELHOR MELHORA (best-improvement) em toda a vizinhança --
      usado pelo VND (Algoritmo 2: "Search for x' in N_l(x) such that
      f(x') < f(x)"). O artigo não especifica first- vs best-improvement
      explicitamente; adotamos best-improvement (escolha documentada, não
      é uma cópia literal do artigo -- só a busca determinística "achar o
      melhor" é claramente compatível com Algoritmo 2 tal como escrito).

  random_move(instance, routes, rng) -> novas_rotas | None
      Aplica UM movimento aleatório válido (não necessariamente de melhora)
      -- usado pelo Shake do GVNS (Algoritmo 1, linha 5): "Given x, execute
      N_{l=rand(0,kmax)}", aceito mesmo se piorar a solução, desde que
      viável (página 8 do artigo). Retorna None se não existir nenhum
      movimento válido nessa vizinhança para a solução atual (ex.: só uma
      rota existe e o operador é inter-rota).

Todas as rotas são listas de ids de cliente (sem o depósito nas pontas).
Rotas vazias nunca são mantidas na solução (removidas ao ficarem vazias).
"""


def _routes_copy(routes):
    return [list(r) for r in routes]


def _demand(instance, route):
    return sum(instance.demands[c] for c in route)


def _emptying_route_allowed(instance, routes):
    """Diz se é permitido reduzir em 1 o número de rotas não-vazias agora
    (i.e., esvaziar totalmente uma rota) sem violar a restrição de número
    de veículos K da instância (ver instance.py, docstring do módulo, e
    diario_pesquisa.md Entrada 3 -- bug real de K não imposto, corrigido).
    Sem essa checagem, Relocate(1,0)/(2,0) podiam produzir soluções com
    MAIS rotas úteis do que a instância permite (K é teto E piso quando
    vehicles_exact=True), inflando artificialmente a qualidade."""
    if not instance.n_vehicles_ref:
        return True
    n_atual = sum(1 for r in routes if len(r) > 0)
    n_depois = n_atual - 1
    if instance.vehicles_exact:
        return n_depois >= instance.n_vehicles_ref  # falso quando n_atual==alvo (caso normal)
    return n_depois >= instance.n_vehicles_ref


# ---------------------------------------------------------------- 2-Opt --
def _two_opt_route_best(instance, route):
    """Melhor 2-opt dentro de UMA rota. Retorna (delta, nova_rota) ou None."""
    n = len(route)
    if n < 2:
        return None
    best_delta = 0.0
    best_route = None
    # arestas (a,b) e (c,d) com a=route[i-1 ou depot], ... -- forma clássica:
    # para 0<=i<j<n, reverte route[i:j+1]
    for i in range(n):
        prev_i = instance.depot_id if i == 0 else route[i - 1]
        for j in range(i + 1, n):
            next_j = instance.depot_id if j == n - 1 else route[j + 1]
            if j == i:
                continue
            old = instance.dist(prev_i, route[i]) + instance.dist(route[j], next_j)
            new = instance.dist(prev_i, route[j]) + instance.dist(route[i], next_j)
            delta = new - old
            if delta < best_delta - 1e-9:
                best_delta = delta
                best_route = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
    if best_route is None:
        return None
    return best_delta, best_route


def two_opt_best(instance, routes):
    best_delta = 0.0
    best_result = None
    for ridx, route in enumerate(routes):
        res = _two_opt_route_best(instance, route)
        if res is not None and res[0] < best_delta - 1e-9:
            best_delta, best_route = res
            best_result = (ridx, best_route)
    if best_result is None:
        return False, routes
    ridx, new_route = best_result
    new_routes = _routes_copy(routes)
    new_routes[ridx] = new_route
    return True, new_routes


def two_opt_random(instance, routes, rng):
    candidatos = [ridx for ridx, r in enumerate(routes) if len(r) >= 2]
    if not candidatos:
        return None
    ridx = rng.choice(candidatos)
    route = routes[ridx]
    n = len(route)
    i = rng.randrange(0, n)
    j = rng.randrange(0, n)
    if i > j:
        i, j = j, i
    if i == j:
        return None
    new_route = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
    new_routes = _routes_copy(routes)
    new_routes[ridx] = new_route
    return new_routes


# ------------------------------------------------------------- Move(1) --
def _move1_route_best(instance, route):
    """Melhor realocação de 1 cliente para outra posição da MESMA rota."""
    n = len(route)
    if n < 3:
        return None
    best_delta = 0.0
    best_route = None
    for p in range(n):
        c = route[p]
        prev_p = instance.depot_id if p == 0 else route[p - 1]
        next_p = instance.depot_id if p == n - 1 else route[p + 1]
        removal_gain = (
            instance.dist(prev_p, c) + instance.dist(c, next_p) - instance.dist(prev_p, next_p)
        )
        without_c = route[:p] + route[p + 1:]
        m = len(without_c)
        for q in range(m + 1):
            a = instance.depot_id if q == 0 else without_c[q - 1]
            b = instance.depot_id if q == m else without_c[q]
            if a == prev_p and b == next_p:
                continue  # mesma posição, sem mudança real
            insertion_cost = instance.dist(a, c) + instance.dist(c, b) - instance.dist(a, b)
            delta = insertion_cost - removal_gain
            if delta < best_delta - 1e-9:
                best_delta = delta
                best_route = without_c[:q] + [c] + without_c[q:]
    if best_route is None:
        return None
    return best_delta, best_route


def move1_best(instance, routes):
    best_delta = 0.0
    best_result = None
    for ridx, route in enumerate(routes):
        res = _move1_route_best(instance, route)
        if res is not None and res[0] < best_delta - 1e-9:
            best_delta, best_route = res
            best_result = (ridx, best_route)
    if best_result is None:
        return False, routes
    ridx, new_route = best_result
    new_routes = _routes_copy(routes)
    new_routes[ridx] = new_route
    return True, new_routes


def move1_random(instance, routes, rng):
    candidatos = [ridx for ridx, r in enumerate(routes) if len(r) >= 2]
    if not candidatos:
        return None
    ridx = rng.choice(candidatos)
    route = list(routes[ridx])
    n = len(route)
    p = rng.randrange(0, n)
    c = route.pop(p)
    q = rng.randrange(0, len(route) + 1)
    route.insert(q, c)
    new_routes = _routes_copy(routes)
    new_routes[ridx] = route
    return new_routes


# ------------------------------------------------------------ Swap(1,1) --
def swap11_best(instance, routes):
    best_delta = 0.0
    best_result = None
    n_routes = len(routes)
    for ra in range(n_routes):
        route_a = routes[ra]
        for pa in range(len(route_a)):
            ca = route_a[pa]
            prev_a = instance.depot_id if pa == 0 else route_a[pa - 1]
            next_a = instance.depot_id if pa == len(route_a) - 1 else route_a[pa + 1]
            remove_a = instance.dist(prev_a, ca) + instance.dist(ca, next_a) - instance.dist(prev_a, next_a)
            for rb in range(ra + 1, n_routes):
                route_b = routes[rb]
                for pb in range(len(route_b)):
                    cb = route_b[pb]
                    prev_b = instance.depot_id if pb == 0 else route_b[pb - 1]
                    next_b = instance.depot_id if pb == len(route_b) - 1 else route_b[pb + 1]
                    remove_b = instance.dist(prev_b, cb) + instance.dist(cb, next_b) - instance.dist(prev_b, next_b)

                    demand_a_new = _demand(instance, route_a) - instance.demands[ca] + instance.demands[cb]
                    demand_b_new = _demand(instance, route_b) - instance.demands[cb] + instance.demands[ca]
                    if demand_a_new > instance.capacity + 1e-6 or demand_b_new > instance.capacity + 1e-6:
                        continue

                    insert_a = instance.dist(prev_a, cb) + instance.dist(cb, next_a) - instance.dist(prev_a, next_a)
                    insert_b = instance.dist(prev_b, ca) + instance.dist(ca, next_b) - instance.dist(prev_b, next_b)
                    delta = (insert_a - remove_a) + (insert_b - remove_b)
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_result = (ra, pa, rb, pb)
    if best_result is None:
        return False, routes
    ra, pa, rb, pb = best_result
    new_routes = _routes_copy(routes)
    new_routes[ra][pa], new_routes[rb][pb] = new_routes[rb][pb], new_routes[ra][pa]
    return True, new_routes


def swap11_random(instance, routes, rng):
    n_routes = len(routes)
    candidatos = [r for r in range(n_routes) if len(routes[r]) >= 1]
    if len(candidatos) < 2:
        return None
    ra, rb = rng.sample(candidatos, 2)
    pa = rng.randrange(0, len(routes[ra]))
    pb = rng.randrange(0, len(routes[rb]))
    ca, cb = routes[ra][pa], routes[rb][pb]
    demand_a_new = _demand(instance, routes[ra]) - instance.demands[ca] + instance.demands[cb]
    demand_b_new = _demand(instance, routes[rb]) - instance.demands[cb] + instance.demands[ca]
    if demand_a_new > instance.capacity + 1e-6 or demand_b_new > instance.capacity + 1e-6:
        return None
    new_routes = _routes_copy(routes)
    new_routes[ra][pa], new_routes[rb][pb] = cb, ca
    return new_routes


# -------------------------------------------------------- Relocate(1,0) --
def relocate10_best(instance, routes):
    best_delta = 0.0
    best_result = None
    n_routes = len(routes)
    for ra in range(n_routes):
        route_a = routes[ra]
        if len(route_a) == 1 and not _emptying_route_allowed(instance, routes):
            continue  # esvaziar route_a violaria o numero de veiculos exigido
        for pa in range(len(route_a)):
            ca = route_a[pa]
            prev_a = instance.depot_id if pa == 0 else route_a[pa - 1]
            next_a = instance.depot_id if pa == len(route_a) - 1 else route_a[pa + 1]
            removal_gain = instance.dist(prev_a, ca) + instance.dist(ca, next_a) - instance.dist(prev_a, next_a)
            demand_a_after = _demand(instance, route_a) - instance.demands[ca]
            for rb in range(n_routes):
                if rb == ra:
                    continue
                route_b = routes[rb]
                if _demand(instance, route_b) + instance.demands[ca] > instance.capacity + 1e-6:
                    continue
                m = len(route_b)
                for q in range(m + 1):
                    a = instance.depot_id if q == 0 else route_b[q - 1]
                    b = instance.depot_id if q == m else route_b[q]
                    insertion_cost = instance.dist(a, ca) + instance.dist(ca, b) - instance.dist(a, b)
                    delta = insertion_cost - removal_gain
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_result = (ra, pa, rb, q, demand_a_after)
    if best_result is None:
        return False, routes
    ra, pa, rb, q, _ = best_result
    new_routes = _routes_copy(routes)
    ca = new_routes[ra].pop(pa)
    new_routes[rb].insert(q, ca)  # ra != rb sempre aqui (relocate é inter-rota)
    new_routes = [r for r in new_routes if len(r) > 0]
    return True, new_routes


def relocate10_random(instance, routes, rng):
    n_routes = len(routes)
    origem_candidatos = [
        r for r in range(n_routes)
        if len(routes[r]) >= 1
        and not (len(routes[r]) == 1 and not _emptying_route_allowed(instance, routes))
    ]
    if not origem_candidatos or n_routes < 2:
        return None
    ra = rng.choice(origem_candidatos)
    pa = rng.randrange(0, len(routes[ra]))
    ca = routes[ra][pa]
    destino_candidatos = [r for r in range(n_routes) if r != ra]
    rng.shuffle(destino_candidatos)
    for rb in destino_candidatos:
        if _demand(instance, routes[rb]) + instance.demands[ca] <= instance.capacity + 1e-6:
            new_routes = _routes_copy(routes)
            new_routes[ra].pop(pa)
            q = rng.randrange(0, len(new_routes[rb]) + 1)
            new_routes[rb].insert(q, ca)
            new_routes = [r for r in new_routes if len(r) > 0]
            return new_routes
    return None


# -------------------------------------------------------- Relocate(2,0) --
def relocate20_best(instance, routes):
    best_delta = 0.0
    best_result = None
    n_routes = len(routes)
    for ra in range(n_routes):
        route_a = routes[ra]
        if len(route_a) < 2:
            continue
        if len(route_a) == 2 and not _emptying_route_allowed(instance, routes):
            continue  # esvaziar route_a violaria o numero de veiculos exigido
        for pa in range(len(route_a) - 1):
            c1, c2 = route_a[pa], route_a[pa + 1]
            prev_a = instance.depot_id if pa == 0 else route_a[pa - 1]
            next_a = instance.depot_id if pa + 1 == len(route_a) - 1 else route_a[pa + 2]
            # As 3 arestas removidas de A são prev-c1, c1-c2 (interna do
            # bloco) e c2-next; a única aresta nova é prev-next. A aresta
            # interna c1-c2 precisa entrar aqui (mesmo saindo de A por
            # inteiro) para cancelar corretamente com o termo equivalente
            # em insertion_cost, que a inclui do lado da rota destino --
            # sem isso, o delta final fica errado por +dist(c1,c2) (bug
            # real encontrado e corrigido em 29/08/2026 por verificação à
            # mão -- ver diario_pesquisa.md, Entrada 3).
            removal_gain = (
                instance.dist(prev_a, c1)
                + instance.dist(c1, c2)
                + instance.dist(c2, next_a)
                - instance.dist(prev_a, next_a)
            )
            demanda_bloco = instance.demands[c1] + instance.demands[c2]
            for rb in range(n_routes):
                if rb == ra:
                    continue
                route_b = routes[rb]
                if _demand(instance, route_b) + demanda_bloco > instance.capacity + 1e-6:
                    continue
                m = len(route_b)
                for q in range(m + 1):
                    a = instance.depot_id if q == 0 else route_b[q - 1]
                    b = instance.depot_id if q == m else route_b[q]
                    # tenta as duas orientações do bloco (c1,c2) e (c2,c1)
                    for bloco in ((c1, c2), (c2, c1)):
                        insertion_cost = (
                            instance.dist(a, bloco[0])
                            + instance.dist(bloco[0], bloco[1])
                            + instance.dist(bloco[1], b)
                            - instance.dist(a, b)
                        )
                        delta = insertion_cost - removal_gain
                        if delta < best_delta - 1e-9:
                            best_delta = delta
                            best_result = (ra, pa, rb, q, bloco)
    if best_result is None:
        return False, routes
    ra, pa, rb, q, bloco = best_result
    new_routes = _routes_copy(routes)
    del new_routes[ra][pa:pa + 2]
    new_routes[rb][q:q] = list(bloco)
    new_routes = [r for r in new_routes if len(r) > 0]
    return True, new_routes


def relocate20_random(instance, routes, rng):
    n_routes = len(routes)
    origem_candidatos = [
        r for r in range(n_routes)
        if len(routes[r]) >= 2
        and not (len(routes[r]) == 2 and not _emptying_route_allowed(instance, routes))
    ]
    if not origem_candidatos or n_routes < 2:
        return None
    ra = rng.choice(origem_candidatos)
    pa = rng.randrange(0, len(routes[ra]) - 1)
    c1, c2 = routes[ra][pa], routes[ra][pa + 1]
    demanda_bloco = instance.demands[c1] + instance.demands[c2]
    destino_candidatos = [r for r in range(n_routes) if r != ra]
    rng.shuffle(destino_candidatos)
    for rb in destino_candidatos:
        if _demand(instance, routes[rb]) + demanda_bloco <= instance.capacity + 1e-6:
            new_routes = _routes_copy(routes)
            del new_routes[ra][pa:pa + 2]
            bloco = [c1, c2] if rng.random() < 0.5 else [c2, c1]
            q = rng.randrange(0, len(new_routes[rb]) + 1)
            new_routes[rb][q:q] = bloco
            new_routes = [r for r in new_routes if len(r) > 0]
            return new_routes
    return None


# ---------------------------------------------------------- Registro ----
# Ordem conforme Tabela 2 do artigo (2-Opt, Move(1), Swap(1,1),
# Relocate(1,0), Relocate(2,0)) -- usada como ordem fixa do VND baseline
# (sem bandit). O artigo não fixa esta ordem explicitamente para o GVNS
# "clássico"; adotamos a ordem em que os operadores são apresentados na
# Tabela 2, do mais simples/barato (intra-rota) ao mais caro (blocos de 2,
# inter-rota) -- coerente com o comentário da página 8 de que "a magnitude
# da complexidade computacional... geralmente dita a ordem de execução".
NEIGHBORHOODS = [
    ("2-Opt", two_opt_best, two_opt_random),
    ("Move(1)", move1_best, move1_random),
    ("Swap(1,1)", swap11_best, swap11_random),
    ("Relocate(1,0)", relocate10_best, relocate10_random),
    ("Relocate(2,0)", relocate20_best, relocate20_random),
]
