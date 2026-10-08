"""
neighborhoods_ext.py
====================
Operadores ADICIONAIS de vizinhanca para a Etapa 3 da Direcao A (diario,
Entradas 75 e seguintes): um conjunto maior e HETEROGENEO (custo e
eficacia bem diferentes), para testar se "selecao adaptativa nao ajuda"
(Etapas 1-2) e uma propriedade do conjunto de 5 operadores parecidos de
Kalatzantonakis et al. (2023) ou da adaptacao em si.

Mesma interface de neighborhoods.py: (nome, best_fn, random_fn), com
  best_fn(instance, routes) -> (melhorou, novas_rotas)   [melhor melhora]
  random_fn(instance, routes, rng) -> None               [NAO usado: o shake
      da Etapa 3 usa so os 5 operadores originais, para que a perturbacao
      seja identica a da Etapa 2; ver aos_vnd.vnd_generico(neighborhoods_shake)]

Operadores novos (todos respeitam capacidade e, quando algum esvazia uma
rota, a restricao de numero de veiculos via `_emptying_route_allowed`):
  2-opt*          inter-rota: troca as CAUDAS de duas rotas (O(na*nb))
  Or-opt(2-3)     intra-rota: move um segmento de 2 ou 3 clientes, com ou
                  sem inversao, para outra posicao da mesma rota
  Swap-intra      intra-rota: troca dois clientes da mesma rota
  CROSS(<=3)      inter-rota: troca um segmento de la clientes de A por um
                  de lb clientes de B, la,lb in {1,2,3}, max(la,lb)>=2
  SWAP*           inter-rota (Vidal 2022): troca u de A por v de B, cada um
                  reinserido na MELHOR posicao da rota do outro (nao
                  necessariamente no lugar do outro). Versao ingenua
                  (sem restricao por setor polar nem top-3 de insercao).

Distancias simetricas assumidas (todas as instancias do projeto o sao).
Verificacao de corretude: tests/validate_neighborhoods_ext.py (enumeracao
independente por forca bruta).
"""
from neighborhoods import _routes_copy, _demand, _emptying_route_allowed

EPS = 1e-9


def _nao_usado_random(instance, routes, rng):
    return None


# --------------------------------------------------------------- 2-opt* ---
def two_opt_star_best(instance, routes):
    d = instance.dist
    dep = instance.depot_id
    Q = instance.capacity + 1e-6
    best_delta = 0.0
    best = None
    cargas = [[0] for _ in routes]  # prefixos de demanda por rota
    for ridx, r in enumerate(routes):
        acc = 0
        for c in r:
            acc += instance.demands[c]
            cargas[ridx].append(acc)
    n = len(routes)
    for ra in range(n):
        a = routes[ra]
        la = len(a)
        for rb in range(ra + 1, n):
            b = routes[rb]
            lb = len(b)
            for i in range(la + 1):
                pa = dep if i == 0 else a[i - 1]
                na = dep if i == la else a[i]
                head_a = cargas[ra][i]
                tail_a = cargas[ra][la] - head_a
                for j in range(lb + 1):
                    if (i == 0 and j == 0) or (i == la and j == lb):
                        continue  # troca de rotas inteiras / nada muda
                    head_b = cargas[rb][j]
                    tail_b = cargas[rb][lb] - head_b
                    if head_a + tail_b > Q or head_b + tail_a > Q:
                        continue
                    if (i == 0 and j == lb) or (i == la and j == 0):
                        # uma das rotas fica vazia
                        if not _emptying_route_allowed(instance, routes):
                            continue
                    pb = dep if j == 0 else b[j - 1]
                    nb = dep if j == lb else b[j]
                    delta = d(pa, nb) + d(pb, na) - d(pa, na) - d(pb, nb)
                    if delta < best_delta - EPS:
                        best_delta = delta
                        best = (ra, rb, i, j)
    if best is None:
        return False, routes
    ra, rb, i, j = best
    a, b = routes[ra], routes[rb]
    new = _routes_copy(routes)
    new[ra] = a[:i] + b[j:]
    new[rb] = b[:j] + a[i:]
    new = [r for r in new if len(r) > 0]
    return True, new


# ------------------------------------------------------------ Or-opt(2-3) --
def or_opt_best(instance, routes):
    d = instance.dist
    dep = instance.depot_id
    best_delta = 0.0
    best = None
    for ridx, r in enumerate(routes):
        n = len(r)
        for L in (2, 3):
            if n < L + 1:
                continue
            for p in range(n - L + 1):
                seg = r[p:p + L]
                prev = dep if p == 0 else r[p - 1]
                nxt = dep if p + L == n else r[p + L]
                rest = r[:p] + r[p + L:]
                m = len(rest)
                remocao = d(prev, seg[0]) + d(seg[-1], nxt) - d(prev, nxt)
                for q in range(m + 1):
                    a = dep if q == 0 else rest[q - 1]
                    b = dep if q == m else rest[q]
                    for inv in (False, True):
                        if not inv and a == prev and b == nxt and q == p:
                            continue  # mesma posicao, mesma orientacao
                        s0, s1 = (seg[-1], seg[0]) if inv else (seg[0], seg[-1])
                        delta = d(a, s0) + d(s1, b) - d(a, b) - remocao
                        if delta < best_delta - EPS:
                            best_delta = delta
                            best = (ridx, p, L, q, inv)
    if best is None:
        return False, routes
    ridx, p, L, q, inv = best
    r = routes[ridx]
    seg = r[p:p + L]
    if inv:
        seg = seg[::-1]
    rest = r[:p] + r[p + L:]
    new = _routes_copy(routes)
    new[ridx] = rest[:q] + seg + rest[q:]
    return True, new


# ------------------------------------------------------------ Swap-intra ---
def swap_intra_best(instance, routes):
    d = instance.dist
    dep = instance.depot_id
    best_delta = 0.0
    best = None
    for ridx, r in enumerate(routes):
        n = len(r)
        for i in range(n):
            pi = dep if i == 0 else r[i - 1]
            ni = dep if i == n - 1 else r[i + 1]
            for j in range(i + 1, n):
                pj = dep if j == 0 else r[j - 1]
                nj = dep if j == n - 1 else r[j + 1]
                ri, rj = r[i], r[j]
                if j == i + 1:
                    delta = d(pi, rj) + d(ri, nj) - d(pi, ri) - d(rj, nj)
                else:
                    delta = (d(pi, rj) + d(rj, ni) + d(pj, ri) + d(ri, nj)
                             - d(pi, ri) - d(ri, ni) - d(pj, rj) - d(rj, nj))
                if delta < best_delta - EPS:
                    best_delta = delta
                    best = (ridx, i, j)
    if best is None:
        return False, routes
    ridx, i, j = best
    new = _routes_copy(routes)
    new[ridx][i], new[ridx][j] = new[ridx][j], new[ridx][i]
    return True, new


# --------------------------------------------------------- CROSS(<=3) ------
_COMBOS_CROSS = [(la, lb) for la in (1, 2, 3) for lb in (1, 2, 3) if max(la, lb) >= 2]


def cross_best(instance, routes):
    d = instance.dist
    dep = instance.depot_id
    Q = instance.capacity + 1e-6
    best_delta = 0.0
    best = None
    cargas = [_demand(instance, r) for r in routes]
    n = len(routes)
    for ra in range(n):
        a = routes[ra]
        for rb in range(ra + 1, n):
            b = routes[rb]
            for la, lb in _COMBOS_CROSS:
                if len(a) < la or len(b) < lb:
                    continue
                for pa in range(len(a) - la + 1):
                    sa = a[pa:pa + la]
                    qa = sum(instance.demands[c] for c in sa)
                    prevA = dep if pa == 0 else a[pa - 1]
                    nextA = dep if pa + la == len(a) else a[pa + la]
                    for pb in range(len(b) - lb + 1):
                        sb = b[pb:pb + lb]
                        qb = sum(instance.demands[c] for c in sb)
                        if cargas[ra] - qa + qb > Q or cargas[rb] - qb + qa > Q:
                            continue
                        prevB = dep if pb == 0 else b[pb - 1]
                        nextB = dep if pb + lb == len(b) else b[pb + lb]
                        delta = (d(prevA, sb[0]) + d(sb[-1], nextA) - d(prevA, sa[0]) - d(sa[-1], nextA)
                                 + d(prevB, sa[0]) + d(sa[-1], nextB) - d(prevB, sb[0]) - d(sb[-1], nextB))
                        if delta < best_delta - EPS:
                            best_delta = delta
                            best = (ra, rb, pa, la, pb, lb)
    if best is None:
        return False, routes
    ra, rb, pa, la, pb, lb = best
    a, b = routes[ra], routes[rb]
    new = _routes_copy(routes)
    new[ra] = a[:pa] + b[pb:pb + lb] + a[pa + la:]
    new[rb] = b[:pb] + a[pa:pa + la] + b[pb + lb:]
    return True, new


# ---------------------------------------------------------------- SWAP* ---
def _melhor_insercao(instance, rota, c):
    """(custo_minimo_de_inserir_c_em_rota, posicao). rota pode ser vazia."""
    d = instance.dist
    dep = instance.depot_id
    m = len(rota)
    melhor, pos = float("inf"), 0
    for q in range(m + 1):
        x = dep if q == 0 else rota[q - 1]
        y = dep if q == m else rota[q]
        custo = d(x, c) + d(c, y) - d(x, y)
        if custo < melhor:
            melhor, pos = custo, q
    return melhor, pos


def swap_star_best(instance, routes):
    d = instance.dist
    dep = instance.depot_id
    Q = instance.capacity + 1e-6
    best_delta = 0.0
    best = None
    cargas = [_demand(instance, r) for r in routes]
    n = len(routes)
    for ra in range(n):
        a = routes[ra]
        for rb in range(ra + 1, n):
            b = routes[rb]
            for iu, u in enumerate(a):
                qu = instance.demands[u]
                pu = dep if iu == 0 else a[iu - 1]
                nu = dep if iu == len(a) - 1 else a[iu + 1]
                ganho_u = d(pu, u) + d(u, nu) - d(pu, nu)
                a_sem_u = a[:iu] + a[iu + 1:]
                for iv, v in enumerate(b):
                    qv = instance.demands[v]
                    if cargas[ra] - qu + qv > Q or cargas[rb] - qv + qu > Q:
                        continue
                    pv = dep if iv == 0 else b[iv - 1]
                    nv = dep if iv == len(b) - 1 else b[iv + 1]
                    ganho_v = d(pv, v) + d(v, nv) - d(pv, nv)
                    b_sem_v = b[:iv] + b[iv + 1:]
                    ins_v, pos_v = _melhor_insercao(instance, a_sem_u, v)
                    ins_u, pos_u = _melhor_insercao(instance, b_sem_v, u)
                    delta = ins_v - ganho_u + ins_u - ganho_v
                    if delta < best_delta - EPS:
                        best_delta = delta
                        best = (ra, rb, iu, iv, pos_v, pos_u)
    if best is None:
        return False, routes
    ra, rb, iu, iv, pos_v, pos_u = best
    a, b = routes[ra], routes[rb]
    u, v = a[iu], b[iv]
    a_sem_u = a[:iu] + a[iu + 1:]
    b_sem_v = b[:iv] + b[iv + 1:]
    new = _routes_copy(routes)
    new[ra] = a_sem_u[:pos_v] + [v] + a_sem_u[pos_v:]
    new[rb] = b_sem_v[:pos_u] + [u] + b_sem_v[pos_u:]
    return True, new


NEIGHBORHOODS_EXT_NOVOS = [
    ("2-opt*", two_opt_star_best, _nao_usado_random),
    ("Or-opt(2-3)", or_opt_best, _nao_usado_random),
    ("Swap-intra", swap_intra_best, _nao_usado_random),
    ("CROSS(<=3)", cross_best, _nao_usado_random),
    ("SWAP*", swap_star_best, _nao_usado_random),
]
