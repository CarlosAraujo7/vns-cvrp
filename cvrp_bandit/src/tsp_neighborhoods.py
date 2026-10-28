"""
tsp_neighborhoods.py
====================
Operadores de vizinhanca do TSP para o Estudo 13 (mesma interface de neighborhoods.py: (nome, best_fn, random_fn)).

  best_fn(inst, routes)   -> (melhorou, novas_rotas)   melhor movimento de melhora da vizinhanca inteira (varredura
                                                       completa; desempate: o primeiro na ordem de varredura)
  random_fn(inst, routes, rng) -> novas_rotas | None   um movimento aleatorio (para o shake; so os 5 primeiros)

Solucao: routes = [tour], tour = clientes 1..N-1; o no 0 e o ponto fixo de partida. Com t = [0] + tour + [0] (posicoes
0..N), a aresta e liga t[e] a t[e+1] (e = 0..N-1) e `edge[e]` e seu comprimento.

Os operadores sao VETORIZADOS com numpy; para manter a medida de custo do projeto (avaliacoes de distancia), cada um soma
a `inst.chamadas` o numero de consultas d(.,.) que uma implementacao direta em lacos faria, com as arestas existentes
lidas sem contar (como nos operadores do CVRP). A corretude do melhor movimento E da contagem e verificada contra uma
enumeracao independente por forca bruta em tests/validate_tsp_neighborhoods.py.

Pool P9 (indice: operador; custo aproximado em avaliacoes, N = numero de nos):
  0 2-opt              inverte o trecho entre duas arestas                      ~ N^2
  1 Or-opt(1)          move 1 no para outra aresta                              ~ N^2
  2 Swap(1,1)          troca dois nos                                           ~ 2 N^2
  3 Or-opt(2)          move um segmento de 2 nos, com ou sem inversao           ~ 2 N^2
  4 Or-opt(3)          idem, 3 nos                                              ~ 2 N^2
  5 Cross(<=3)         troca dois segmentos disjuntos de la, lb in {1,2,3}      ~ 16 N^2
  6 Or-opt(4-10)       move segmento de 4 a 10 nos, com ou sem inversao         ~ 14 N^2
  7 3-opt A            troca de dois trechos adjacentes quaisquer (A C B)       ~ N^3 / 2
  8 3-opt B            inversao dos dois trechos (A B' C'), trechos de >= 2 nos ~ N^3 / 2
P5 = os cinco primeiros (analogos dos cinco operadores originais do VNS para o CVRP).
"""
import numpy as np

EPS = 1e-9


def _t(routes):
    return np.array([0] + list(routes[0]) + [0], dtype=np.int64)


def _saida(t):
    return [[int(v) for v in t[1:-1]]]


# ----------------------------------------------------------------------------------------------- 2-opt
def _best_2opt(inst, routes):
    t, N, D = _t(routes), inst.N, inst.Dn
    a, b = t[:N], t[1:N + 1]
    edge = D[a, b]
    delta = D[np.ix_(a, a)] + D[np.ix_(b, b)] - edge[:, None] - edge[None, :]
    delta = np.where(np.triu(np.ones((N, N), bool), 2), delta, np.inf)
    inst.chamadas += 2 * ((N - 2) * (N - 1) // 2)
    k = int(np.argmin(delta))
    i, j = divmod(k, N)
    if delta[i, j] < -EPS:
        return True, _saida(np.concatenate([t[:i + 1], t[i + 1:j + 1][::-1], t[j + 1:]]))
    return False, routes


# ----------------------------------------------------------------------------------------------- Or-opt
def _best_or(inst, routes, Ls):
    t, N, D = _t(routes), inst.N, inst.Dn
    edge = D[t[:-1], t[1:]]
    tk, tk1 = t[:N], t[1:N + 1]
    kk = np.arange(N)
    melhor, mov = np.inf, None
    for L in Ls:
        ns = N - L
        if ns < 1:
            continue
        S = np.arange(1, N - L + 1)
        first, last = t[S], t[S + L - 1]
        rem = D[t[S - 1], t[S + L]] - edge[S - 1] - edge[S + L - 1]
        df = D[first[:, None], tk[None, :]] + D[last[:, None], tk1[None, :]] - edge[None, :]
        validas = (kk[None, :] <= S[:, None] - 2) | (kk[None, :] >= S[:, None] + L)
        if L == 1:
            cand = np.where(validas, rem[:, None] + df, np.inf)[:, :, None]
        else:
            dr = D[last[:, None], tk[None, :]] + D[first[:, None], tk1[None, :]] - edge[None, :]
            cand = np.stack([np.where(validas, rem[:, None] + df, np.inf),
                             np.where(validas, rem[:, None] + dr, np.inf)], axis=2)
        inst.chamadas += ns + ns * (N - L - 1) * (2 if L == 1 else 4)
        flat = int(np.argmin(cand))
        si, ki, o = np.unravel_index(flat, cand.shape)
        if cand[si, ki, o] < melhor:
            melhor, mov = float(cand[si, ki, o]), (L, int(S[si]), int(ki), int(o))
    if mov is not None and melhor < -EPS:
        L, s, k, o = mov
        seg = list(t[s:s + L])
        if o == 1:
            seg = seg[::-1]
        rest = list(t[:s]) + list(t[s + L:])
        pos = k if k < s else k - L
        novo = rest[:pos + 1] + seg + rest[pos + 1:]
        return True, _saida(np.array(novo))
    return False, routes


def _best_or1(inst, routes):
    return _best_or(inst, routes, (1,))


def _best_or2(inst, routes):
    return _best_or(inst, routes, (2,))


def _best_or3(inst, routes):
    return _best_or(inst, routes, (3,))


def _best_or4_10(inst, routes):
    return _best_or(inst, routes, tuple(range(4, 11)))


# ----------------------------------------------------------------------------------------------- Swap(1,1)
def _best_swap(inst, routes):
    t, N, D = _t(routes), inst.N, inst.Dn
    m = N - 1
    if m < 2:
        return False, routes
    edge = D[t[:-1], t[1:]]
    P = np.arange(1, N)                                    # posicoes dos clientes (linhas e colunas)
    pm, pp = t[P - 1], t[P + 1]                            # vizinhos de cada posicao
    tp = t[P]
    # q >= p+2: vizinhos distintos
    qm1 = t[np.maximum(P - 1, 0)]
    nova = (D[pm[:, None], tp[None, :]] + D[tp[None, :], pp[:, None]]          # t[p-1]-t[q], t[q]-t[p+1]
            + D[qm1[None, :], tp[:, None]] + D[tp[:, None], pp[None, :]])      # t[q-1]-t[p], t[p]-t[q+1]
    velha = (edge[P - 1][:, None] + edge[P][:, None] + edge[P - 1][None, :] + edge[P][None, :])
    delta = nova - velha
    pq = P[:, None] < P[None, :] - 1                       # q >= p+2
    delta = np.where(pq, delta, np.inf)
    # q = p+1 (adjacentes): so 2 arestas novas; sem a aresta do meio
    if m >= 2:
        pa = P[:-1]
        d_adj = D[t[pa - 1], t[pa + 1]] + D[t[pa], t[pa + 2]] - edge[pa - 1] - edge[pa + 1]
        for idx, p in enumerate(pa):
            delta[p - 1, p] = d_adj[idx]                   # linha p-1, coluna (p+1)-1 = p
    npares = m * (m - 1) // 2
    inst.chamadas += 4 * (npares - (m - 1)) + 2 * (m - 1)
    k = int(np.argmin(delta))
    pi, qi = divmod(k, m)
    if delta[pi, qi] < -EPS:
        t2 = t.copy()
        t2[pi + 1], t2[qi + 1] = t[qi + 1], t[pi + 1]
        return True, _saida(t2)
    return False, routes


# ----------------------------------------------------------------------------------------------- Cross(<=3)
def _best_cross(inst, routes):
    t, N, D = _t(routes), inst.N, inst.Dn
    edge = D[t[:-1], t[1:]]
    melhor, mov = np.inf, None
    for la in (1, 2, 3):
        for lb in (1, 2, 3):
            if la == 1 and lb == 1:
                continue
            P = np.arange(1, N - la - lb + 1)              # p: 1 .. N-1-(la+lb-1)
            if len(P) == 0:
                continue
            Q = np.arange(1, N - lb + 1)                   # q: 1 .. N-lb
            pm, pf, pl, pn = t[P - 1], t[P], t[P + la - 1], t[P + la]
            qm, qf, ql, qn = t[Q - 1], t[Q], t[Q + lb - 1], t[Q + lb]
            nao_adj = (D[pm[:, None], qf[None, :]] + D[ql[None, :], pn[:, None]]
                       + D[qm[None, :], pf[:, None]] + D[pl[:, None], qn[None, :]])
            velha_n = (edge[P - 1] + edge[P + la - 1])[:, None] + (edge[Q - 1] + edge[Q + lb - 1])[None, :]
            d_na = nao_adj - velha_n
            adj = (D[pm[:, None], qf[None, :]] + D[ql[None, :], pf[:, None]] + D[pl[:, None], qn[None, :]]
                   - (edge[P - 1] + edge[P + la - 1])[:, None] - edge[Q + lb - 1][None, :])
            q_ge = Q[None, :] >= P[:, None] + la
            e_adj = Q[None, :] == P[:, None] + la
            delta = np.where(q_ge, np.where(e_adj, adj, d_na), np.inf)
            nadj = int(e_adj.sum())
            nvalidos = int(q_ge.sum())
            inst.chamadas += 4 * (nvalidos - nadj) + 3 * nadj
            k = int(np.argmin(delta))
            pi, qi = divmod(k, len(Q))
            if delta[pi, qi] < melhor:
                melhor, mov = float(delta[pi, qi]), (la, lb, int(P[pi]), int(Q[qi]))
    if mov is not None and melhor < -EPS:
        la, lb, p, q = mov
        S1, S2 = list(t[p:p + la]), list(t[q:q + lb])
        novo = list(t[:p]) + S2 + list(t[p + la:q]) + S1 + list(t[q + lb:])
        return True, _saida(np.array(novo))
    return False, routes


# ----------------------------------------------------------------------------------------------- 3-opt A: A C B
def _best_3opt_a(inst, routes):
    t, N, D = _t(routes), inst.N, inst.Dn
    edge = D[t[:-1], t[1:]]
    melhor, mov = np.inf, None
    for i in range(0, N - 2):
        J = np.arange(i + 1, N - 1)                        # j: i+1 .. N-2
        K = np.arange(i + 2, N)                            # k: i+2 .. N-1
        novas = (D[t[i], t[J + 1]][:, None] + D[t[K][None, :], t[i + 1]]
                 + D[t[J][:, None], t[K + 1][None, :]])
        delta = novas - edge[i] - edge[J][:, None] - edge[K][None, :]
        delta = np.where(K[None, :] > J[:, None], delta, np.inf)
        flat = int(np.argmin(delta))
        ji, ki = divmod(flat, len(K))
        if delta[ji, ki] < melhor:
            melhor, mov = float(delta[ji, ki]), (i, int(J[ji]), int(K[ki]))
    inst.chamadas += 3 * (N * (N - 1) * (N - 2) // 6)
    if mov is not None and melhor < -EPS:
        i, j, k = mov
        novo = np.concatenate([t[:i + 1], t[j + 1:k + 1], t[i + 1:j + 1], t[k + 1:]])
        return True, _saida(novo)
    return False, routes


# ----------------------------------------------------------------------------------------------- 3-opt B: A B' C'
def _best_3opt_b(inst, routes):
    t, N, D = _t(routes), inst.N, inst.Dn
    edge = D[t[:-1], t[1:]]
    melhor, mov = np.inf, None
    for i in range(0, N - 4):
        J = np.arange(i + 2, N - 2)                        # j: i+2 .. N-3  (B tem >= 2 nos)
        K = np.arange(i + 4, N)                            # k: i+4 .. N-1
        novas = (D[t[i], t[J]][:, None] + D[t[i + 1], t[K]][None, :] + D[t[J + 1][:, None], t[K + 1][None, :]])
        delta = novas - edge[i] - edge[J][:, None] - edge[K][None, :]
        delta = np.where(K[None, :] >= J[:, None] + 2, delta, np.inf)
        flat = int(np.argmin(delta))
        ji, ki = divmod(flat, len(K))
        if delta[ji, ki] < melhor:
            melhor, mov = float(delta[ji, ki]), (i, int(J[ji]), int(K[ki]))
    inst.chamadas += 3 * ((N - 2) * (N - 3) * (N - 4) // 6) if N >= 5 else 0
    if mov is not None and melhor < -EPS:
        i, j, k = mov
        novo = np.concatenate([t[:i + 1], t[i + 1:j + 1][::-1], t[j + 1:k + 1][::-1], t[k + 1:]])
        return True, _saida(novo)
    return False, routes


# ----------------------------------------------------------------------------------------------- movimentos aleatorios (shake)
def _rand_2opt(inst, routes, rng):
    m = inst.N - 1
    if m < 3:
        return None
    i = rng.randint(1, m - 1)
    j = rng.randint(i + 1, m)
    tour = list(routes[0])
    tour[i - 1:j] = tour[i - 1:j][::-1]
    return [tour]


def _rand_or(L):
    def f(inst, routes, rng):
        m = inst.N - 1
        if m < L + 2:
            return None
        tour = list(routes[0])
        s = rng.randint(0, m - L)
        seg = tour[s:s + L]
        rest = tour[:s] + tour[s + L:]
        pos = rng.randint(0, len(rest))
        if pos == s:
            pos = (pos + 1) % (len(rest) + 1)
        if L > 1 and rng.random() < 0.5:
            seg = seg[::-1]
        return [rest[:pos] + seg + rest[pos:]]
    return f


def _rand_swap(inst, routes, rng):
    m = inst.N - 1
    if m < 2:
        return None
    p, q = rng.sample(range(m), 2)
    tour = list(routes[0])
    tour[p], tour[q] = tour[q], tour[p]
    return [tour]


def _nao_usado(inst, routes, rng):
    return None


P9 = [("2-opt", _best_2opt, _rand_2opt),
      ("Or-opt(1)", _best_or1, _rand_or(1)),
      ("Swap(1,1)", _best_swap, _rand_swap),
      ("Or-opt(2)", _best_or2, _rand_or(2)),
      ("Or-opt(3)", _best_or3, _rand_or(3)),
      ("Cross(<=3)", _best_cross, _nao_usado),
      ("Or-opt(4-10)", _best_or4_10, _nao_usado),
      ("3-opt A", _best_3opt_a, _nao_usado),
      ("3-opt B", _best_3opt_b, _nao_usado)]
P5 = P9[:5]
