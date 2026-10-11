"""
neighborhoods_fast.py
=====================
SWAP* EFICIENTE (Vidal, 2022): mesma vizinhanca e mesma melhor melhora do `swap_star_best` ingenuo de
neighborhoods_ext.py, mas com a pre-computacao das 3 melhores posicoes de insercao de cada cliente na rota
oposta (e a opcao "no lugar do removido"), de modo que cada par (u, v) custa O(1) em vez de O(|A| + |B|).
O custo cai por um fator de ~ (|A| + |B|) e o crescimento com n deixa de ser n^2 x tamanho de rota.

Por que existe (Entrada 100/104 do diario): o SWAP* ingenuo e muito mais caro que o do HGS-CVRP e o seu custo
cresce com o tamanho da instancia; isso pode inflar o valor de politicas de selecao em instancias grandes e
pesar na tese "operadores caros valem mais do que a eficiencia sugere". O Estudo 11 repete o nucleo dos
resultados com este operador.

Contabilidade: toda chamada a `instance.dist` conta (arestas das rotas, ganhos de remocao, custos de insercao).
Corretude: tests/test_swap_star_fast.py (mesmo melhor delta que a versao ingenua em instancias aleatorias; o
movimento devolvido reproduz o delta; viabilidade).
"""
from neighborhoods import _routes_copy, _demand

EPS = 1e-9
INF = float("inf")


def _top3(d, dep, nos, arest, c):
    """3 menores custos de inserir c entre nos consecutivos de `nos` (depositos nas pontas):
    lista de (custo, p) com p = posicao na LISTA da rota (0 = antes do 1o cliente), em ordem crescente."""
    ult = len(nos) - 1
    dcd = d(c, dep)
    prev_d = dcd
    t = []
    for p in range(1, ult + 1):
        dv = dcd if p == ult else d(c, nos[p])
        t.append((prev_d + dv - arest[p - 1], p - 1))
        prev_d = dv
    t.sort()
    return t[:3]


def swap_star_fast_best(instance, routes):
    d = instance.dist
    dep = instance.depot_id
    Q = instance.capacity + 1e-6
    dem = instance.demands
    nr = len(routes)
    cargas = [_demand(instance, r) for r in routes]
    nos = [[dep] + r + [dep] for r in routes]
    arest = [[d(nos[i][k], nos[i][k + 1]) for k in range(len(nos[i]) - 1)] for i in range(nr)]
    # ganho de remover o k-esimo cliente da rota i, e custo do "contorno" d(prev, next)
    ganho, contorno = [], []
    for i in range(nr):
        g, c_ = [], []
        for k in range(len(routes[i])):
            dn = d(nos[i][k], nos[i][k + 2])
            c_.append(dn)
            g.append(arest[i][k] + arest[i][k + 1] - dn)
        ganho.append(g)
        contorno.append(c_)
    best_delta = 0.0
    best = None
    for ra in range(nr):
        A = routes[ra]
        for rb in range(ra + 1, nr):
            B = routes[rb]
            # 3 melhores posicoes de cada u de A em B e de cada v de B em A
            top_u_em_B = [_top3(d, dep, nos[rb], arest[rb], u) for u in A]
            top_v_em_A = [_top3(d, dep, nos[ra], arest[ra], v) for v in B]
            for iu, u in enumerate(A):
                qu = dem[u]
                pu, nu = nos[ra][iu], nos[ra][iu + 2]
                for iv, v in enumerate(B):
                    qv = dem[v]
                    if cargas[ra] - qu + qv > Q or cargas[rb] - qv + qu > Q:
                        continue
                    pv, nv = nos[rb][iv], nos[rb][iv + 2]
                    # v em A sem u: no lugar de u, ou uma das 3 melhores posicoes nao adjacentes a u
                    ins_v, pos_v = d(pu, v) + d(v, nu) - contorno[ra][iu], iu
                    for custo, p in top_v_em_A[iv]:
                        if p != iu and p != iu + 1:
                            if custo < ins_v:
                                ins_v, pos_v = custo, (p if p < iu else p - 1)
                            break
                    ins_u, pos_u = d(pv, u) + d(u, nv) - contorno[rb][iv], iv
                    for custo, p in top_u_em_B[iu]:
                        if p != iv and p != iv + 1:
                            if custo < ins_u:
                                ins_u, pos_u = custo, (p if p < iv else p - 1)
                            break
                    delta = ins_v - ganho[ra][iu] + ins_u - ganho[rb][iv]
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
