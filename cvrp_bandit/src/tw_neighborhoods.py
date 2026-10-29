"""
tw_neighborhoods.py
===================
Vizinhancas de MELHOR melhora do CVRPTW (Estudo 14), em numba, com viabilidade de janelas de tempo em O(1) por candidato por
concatenacao de segmentos (Vidal et al. 2013): cada segmento carrega (D = duracao minima, E = inicio mais cedo, L = inicio mais
tarde) e a concatenacao de dois segmentos e viavel sse E1 + D1 + d(ultimo1, primeiro2) <= L2. Prefixos e sufixos de cada rota sao
precomputados a cada aplicacao; trechos curtos (<= 5 nos) sao compostos na hora; trechos intra-rota intermediarios, incrementalmente.

Todas as grandezas sao inteiros em decimos (ver tw_inst.py). Cada operador examina TODA a vizinhanca em uma ordem de varredura
fixa e aplica o movimento de menor delta (o primeiro da varredura em caso de empate) se delta < 0.

CONVENCAO DO ORCAMENTO (avaliacoes de distancia, como nos Estudos 3-13): cada candidato estrutural que passa no teste de
capacidade custa o numero de ARCOS NOVOS do movimento (a viabilidade temporal e uma aritmetica O(1) e nao e contada); candidatos
inviaveis na capacidade custam 0. Arcos por candidato: Relocate(1,0) 3; Relocate(2,0) 6 (as duas orientacoes); Swap(1,1) 4;
2-opt* 2; Move(1) 3; Or-opt(2-3) 6 (as duas orientacoes); Swap-intra 4; CROSS 4. A contagem e analitica e foi validada contra
uma enumeracao independente (tests/validate_tw_neighborhoods.py).

Indices de posicao: a rota r tem Ln[r] clientes em R[r,1..Ln[r]], deposito em R[r,0] e R[r,Ln[r]+1]. Um movimento "inserir apos a
posicao j" coloca o(s) no(s) entre R[r,j] e R[r,j+1].
"""
import numpy as np
from numba import njit

EPS = 0


# --------------------------------------------------------------------------------------------- segmentos
@njit(cache=True)
def cat(A, B, Dm):
    """Concatena dois segmentos A=(d,e,l,primeiro,ultimo) e B. Devolve (segmento, viavel)."""
    dl = Dm[A[4], B[3]]
    dd = A[0] + dl
    wt = B[1] - dd - A[2]
    if wt < 0:
        wt = 0
    ok = A[1] + dd <= B[2]
    e = max(B[1] - dd, A[1]) - wt
    l = min(B[2] - dd, A[2])
    return (A[0] + B[0] + dl + wt, e, l, A[3], B[4]), ok


@njit(cache=True)
def no(u, e, l, s):
    return (s[u], e[u], l[u], u, u)


@njit(cache=True)
def pre(PD, PE, PL, R, r, p):
    return (PD[r, p], PE[r, p], PL[r, p], R[r, 0], R[r, p])


@njit(cache=True)
def suf(SD, SE, SL, R, Ln, r, p):
    return (SD[r, p], SE[r, p], SL[r, p], R[r, p], R[r, Ln[r] + 1])


@njit(cache=True)
def prepara(R, Ln, Dm, dem, e, l, s):
    m, W = R.shape
    PD = np.zeros((m, W), np.int64)
    PE = np.zeros((m, W), np.int64)
    PL = np.zeros((m, W), np.int64)
    SD = np.zeros((m, W), np.int64)
    SE = np.zeros((m, W), np.int64)
    SL = np.zeros((m, W), np.int64)
    CL = np.zeros((m, W), np.int64)
    for r in range(m):
        n = Ln[r]
        A = no(0, e, l, s)
        PD[r, 0], PE[r, 0], PL[r, 0] = A[0], A[1], A[2]
        for p in range(1, n + 2):
            A, _ = cat(A, no(R[r, p], e, l, s), Dm)
            PD[r, p], PE[r, p], PL[r, p] = A[0], A[1], A[2]
            CL[r, p] = CL[r, p - 1] + dem[R[r, p]]
        B = no(0, e, l, s)
        SD[r, n + 1], SE[r, n + 1], SL[r, n + 1] = B[0], B[1], B[2]
        for p in range(n, -1, -1):
            B, _ = cat(no(R[r, p], e, l, s), B, Dm)
            SD[r, p], SE[r, p], SL[r, p] = B[0], B[1], B[2]
    return PD, PE, PL, SD, SE, SL, CL


@njit(cache=True)
def seg_nos(R, r, i, k, rev, Dm, e, l, s):
    """Segmento dos k nos R[r,i..i+k-1] (invertido se rev). Devolve (segmento, viavel)."""
    if rev:
        S = no(R[r, i + k - 1], e, l, s)
        for q in range(i + k - 2, i - 1, -1):
            S, ok = cat(S, no(R[r, q], e, l, s), Dm)
            if not ok:
                return S, False
    else:
        S = no(R[r, i], e, l, s)
        for q in range(i + 1, i + k):
            S, ok = cat(S, no(R[r, q], e, l, s), Dm)
            if not ok:
                return S, False
    return S, True


# --------------------------------------------------------------------------------------------- 1. Relocate(1,0)
@njit(cache=True)
def k_reloc1(R, Ln, PD, PE, PL, SD, SE, SL, CL, Dm, dem, e, l, s, Q):
    m = R.shape[0]
    best, ba, bi, bb, bj, cnt = 0, -1, -1, -1, -1, 0
    for a in range(m):
        La = Ln[a]
        for i in range(1, La + 1):
            u = R[a, i]
            p = R[a, i - 1]
            sx = R[a, i + 1]
            _, okA = cat(pre(PD, PE, PL, R, a, i - 1), suf(SD, SE, SL, R, Ln, a, i + 1), Dm)
            rem = Dm[p, sx] - Dm[p, u] - Dm[u, sx]
            for b in range(m):
                if b == a:
                    continue
                if CL[b, Ln[b] + 1] + dem[u] > Q:
                    continue
                Lb = Ln[b]
                for j in range(0, Lb + 1):
                    cnt += 3
                    if not okA:
                        continue
                    x = R[b, j]
                    y = R[b, j + 1]
                    delta = rem + Dm[x, u] + Dm[u, y] - Dm[x, y]
                    if delta >= best:
                        continue
                    S1, ok1 = cat(pre(PD, PE, PL, R, b, j), no(u, e, l, s), Dm)
                    if not ok1:
                        continue
                    _, ok2 = cat(S1, suf(SD, SE, SL, R, Ln, b, j + 1), Dm)
                    if ok2:
                        best, ba, bi, bb, bj = delta, a, i, b, j
    return best, ba, bi, bb, bj, 0, cnt


# --------------------------------------------------------------------------------------------- 2. Relocate(2,0)
@njit(cache=True)
def k_reloc2(R, Ln, PD, PE, PL, SD, SE, SL, CL, Dm, dem, e, l, s, Q):
    m = R.shape[0]
    best, ba, bi, bb, bj, bo, cnt = 0, -1, -1, -1, -1, -1, 0
    for a in range(m):
        La = Ln[a]
        for i in range(1, La):
            u1 = R[a, i]
            u2 = R[a, i + 1]
            p = R[a, i - 1]
            sx = R[a, i + 2]
            _, okA = cat(pre(PD, PE, PL, R, a, i - 1), suf(SD, SE, SL, R, Ln, a, i + 2), Dm)
            remd = Dm[p, sx] - Dm[p, u1] - Dm[u2, sx]
            for b in range(m):
                if b == a:
                    continue
                if CL[b, Ln[b] + 1] + dem[u1] + dem[u2] > Q:
                    continue
                Lb = Ln[b]
                for j in range(0, Lb + 1):
                    cnt += 6
                    if not okA:
                        continue
                    x = R[b, j]
                    y = R[b, j + 1]
                    for o in range(2):
                        if o == 0:
                            f, t = u1, u2
                        else:
                            f, t = u2, u1
                        delta = remd + Dm[x, f] + Dm[t, y] - Dm[x, y]
                        if delta >= best:
                            continue
                        Sg, okS = cat(no(f, e, l, s), no(t, e, l, s), Dm)
                        if not okS:
                            continue
                        S1, ok1 = cat(pre(PD, PE, PL, R, b, j), Sg, Dm)
                        if not ok1:
                            continue
                        _, ok2 = cat(S1, suf(SD, SE, SL, R, Ln, b, j + 1), Dm)
                        if ok2:
                            best, ba, bi, bb, bj, bo = delta, a, i, b, j, o
    return best, ba, bi, bb, bj, bo, cnt


# --------------------------------------------------------------------------------------------- 3. Swap(1,1)
@njit(cache=True)
def k_swap11(R, Ln, PD, PE, PL, SD, SE, SL, CL, Dm, dem, e, l, s, Q):
    m = R.shape[0]
    best, ba, bi, bb, bj, cnt = 0, -1, -1, -1, -1, 0
    for a in range(m):
        La = Ln[a]
        loadA = CL[a, La + 1]
        for b in range(a + 1, m):
            Lb = Ln[b]
            loadB = CL[b, Lb + 1]
            for i in range(1, La + 1):
                u = R[a, i]
                pa = R[a, i - 1]
                sa = R[a, i + 1]
                for j in range(1, Lb + 1):
                    v = R[b, j]
                    if loadA - dem[u] + dem[v] > Q or loadB - dem[v] + dem[u] > Q:
                        continue
                    cnt += 4
                    pb = R[b, j - 1]
                    sb = R[b, j + 1]
                    delta = Dm[pa, v] + Dm[v, sa] - Dm[pa, u] - Dm[u, sa] + Dm[pb, u] + Dm[u, sb] - Dm[pb, v] - Dm[v, sb]
                    if delta >= best:
                        continue
                    S1, ok1 = cat(pre(PD, PE, PL, R, a, i - 1), no(v, e, l, s), Dm)
                    if not ok1:
                        continue
                    _, ok2 = cat(S1, suf(SD, SE, SL, R, Ln, a, i + 1), Dm)
                    if not ok2:
                        continue
                    S3, ok3 = cat(pre(PD, PE, PL, R, b, j - 1), no(u, e, l, s), Dm)
                    if not ok3:
                        continue
                    _, ok4 = cat(S3, suf(SD, SE, SL, R, Ln, b, j + 1), Dm)
                    if ok4:
                        best, ba, bi, bb, bj = delta, a, i, b, j
    return best, ba, bi, bb, bj, 0, cnt


# --------------------------------------------------------------------------------------------- 4. 2-opt* (troca de caudas)
@njit(cache=True)
def k_twoopt_star(R, Ln, PD, PE, PL, SD, SE, SL, CL, Dm, dem, e, l, s, Q):
    m = R.shape[0]
    best, ba, bi, bb, bj, cnt = 0, -1, -1, -1, -1, 0
    for a in range(m):
        La = Ln[a]
        loadA = CL[a, La + 1]
        for b in range(a + 1, m):
            Lb = Ln[b]
            loadB = CL[b, Lb + 1]
            for i in range(0, La + 1):
                for j in range(0, Lb + 1):
                    nA = CL[a, i] + loadB - CL[b, j]
                    nB = CL[b, j] + loadA - CL[a, i]
                    if nA > Q or nB > Q:
                        continue
                    cnt += 2
                    ai, ai1 = R[a, i], R[a, i + 1]
                    bj_, bj1 = R[b, j], R[b, j + 1]
                    delta = Dm[ai, bj1] + Dm[bj_, ai1] - Dm[ai, ai1] - Dm[bj_, bj1]
                    if delta >= best:
                        continue
                    _, ok1 = cat(pre(PD, PE, PL, R, a, i), suf(SD, SE, SL, R, Ln, b, j + 1), Dm)
                    if not ok1:
                        continue
                    _, ok2 = cat(pre(PD, PE, PL, R, b, j), suf(SD, SE, SL, R, Ln, a, i + 1), Dm)
                    if ok2:
                        best, ba, bi, bb, bj = delta, a, i, b, j
    return best, ba, bi, bb, bj, 0, cnt


# --------------------------------------------------------------------------------------------- 5. Move(1) intra-rota
@njit(cache=True)
def k_move1(R, Ln, PD, PE, PL, SD, SE, SL, CL, Dm, dem, e, l, s, Q):
    m = R.shape[0]
    best, ba, bi, bj, cnt = 0, -1, -1, -1, 0
    for a in range(m):
        La = Ln[a]
        midD = np.zeros(La + 2, np.int64)
        midE = np.zeros(La + 2, np.int64)
        midL = np.zeros(La + 2, np.int64)
        midF = np.zeros(La + 2, np.int64)
        midT = np.zeros(La + 2, np.int64)
        for i in range(1, La + 1):
            u = R[a, i]
            p = R[a, i - 1]
            sx = R[a, i + 1]
            rem = Dm[p, sx] - Dm[p, u] - Dm[u, sx]
            # MID(j+1..i-1) para j < i-1, guardado em j, calculado de tras para frente (trecho contiguo de rota viavel: sempre viavel)
            if i >= 2:
                cur = no(R[a, i - 1], e, l, s)
                for j in range(i - 2, -1, -1):
                    if j < i - 2:
                        cur, _ = cat(no(R[a, j + 1], e, l, s), cur, Dm)
                    midD[j], midE[j], midL[j], midF[j], midT[j] = cur[0], cur[1], cur[2], cur[3], cur[4]
            for j in range(0, La + 1):
                if j == i - 1 or j == i:
                    continue
                cnt += 3
                x = R[a, j]
                y = R[a, j + 1]
                delta = rem + Dm[x, u] + Dm[u, y] - Dm[x, y]
                if delta >= best:
                    continue
                if j < i - 1:
                    MID = (midD[j], midE[j], midL[j], midF[j], midT[j])
                    S1, ok1 = cat(pre(PD, PE, PL, R, a, j), no(u, e, l, s), Dm)
                    if not ok1:
                        continue
                    S2, ok2 = cat(S1, MID, Dm)
                    if not ok2:
                        continue
                    _, ok3 = cat(S2, suf(SD, SE, SL, R, Ln, a, i + 1), Dm)
                    if ok3:
                        best, ba, bi, bj = delta, a, i, j
                else:
                    # j > i: PRE(i-1) + MID(i+1..j) + u + SUF(j+1)
                    S1 = pre(PD, PE, PL, R, a, i - 1)
                    okc = True
                    for q in range(i + 1, j + 1):
                        S1, okq = cat(S1, no(R[a, q], e, l, s), Dm)
                        if not okq:
                            okc = False
                            break
                    if not okc:
                        continue
                    S2, ok2 = cat(S1, no(u, e, l, s), Dm)
                    if not ok2:
                        continue
                    _, ok3 = cat(S2, suf(SD, SE, SL, R, Ln, a, j + 1), Dm)
                    if ok3:
                        best, ba, bi, bj = delta, a, i, j
    return best, ba, bi, bj, 0, 0, cnt


# --------------------------------------------------------------------------------------------- 6. Or-opt(2-3) intra-rota
@njit(cache=True)
def k_oropt23(R, Ln, PD, PE, PL, SD, SE, SL, CL, Dm, dem, e, l, s, Q):
    m = R.shape[0]
    best, ba, bi, bk, bj, bo, cnt = 0, -1, -1, -1, -1, -1, 0
    for a in range(m):
        La = Ln[a]
        for k in range(2, 4):
            for i in range(1, La - k + 2):
                p = R[a, i - 1]
                sx = R[a, i + k]
                f0 = R[a, i]
                l0 = R[a, i + k - 1]
                remd = Dm[p, sx] - Dm[p, f0] - Dm[l0, sx]
                for j in range(0, La + 1):
                    if j >= i - 1 and j <= i + k - 1:
                        continue
                    cnt += 6
                    x = R[a, j]
                    y = R[a, j + 1]
                    for o in range(2):
                        if o == 0:
                            f, t = f0, l0
                        else:
                            f, t = l0, f0
                        delta = remd + Dm[x, f] + Dm[t, y] - Dm[x, y]
                        if delta >= best:
                            continue
                        Sg, okS = seg_nos(R, a, i, k, o == 1, Dm, e, l, s)
                        if not okS:
                            continue
                        if j < i - 1:
                            S1, ok1 = cat(pre(PD, PE, PL, R, a, j), Sg, Dm)
                            if not ok1:
                                continue
                            for q in range(j + 1, i):
                                S1, okq = cat(S1, no(R[a, q], e, l, s), Dm)
                                if not okq:
                                    ok1 = False
                                    break
                            if not ok1:
                                continue
                            _, ok3 = cat(S1, suf(SD, SE, SL, R, Ln, a, i + k), Dm)
                        else:
                            S1 = pre(PD, PE, PL, R, a, i - 1)
                            ok1 = True
                            for q in range(i + k, j + 1):
                                S1, okq = cat(S1, no(R[a, q], e, l, s), Dm)
                                if not okq:
                                    ok1 = False
                                    break
                            if not ok1:
                                continue
                            S2, ok2 = cat(S1, Sg, Dm)
                            if not ok2:
                                continue
                            _, ok3 = cat(S2, suf(SD, SE, SL, R, Ln, a, j + 1), Dm)
                        if ok3:
                            best, ba, bi, bk, bj, bo = delta, a, i, k, j, o
    return best, ba, bi, bk, bj, bo, cnt


# --------------------------------------------------------------------------------------------- 7. Swap intra-rota
@njit(cache=True)
def k_swapintra(R, Ln, PD, PE, PL, SD, SE, SL, CL, Dm, dem, e, l, s, Q):
    m = R.shape[0]
    best, ba, bi, bj, cnt = 0, -1, -1, -1, 0
    for a in range(m):
        La = Ln[a]
        for i in range(1, La):
            u = R[a, i]
            pa = R[a, i - 1]
            sa = R[a, i + 1]
            MID = no(u, e, l, s)                 # MID(i+1..j-1): so usado para j >= i+2
            for j in range(i + 1, La + 1):
                v = R[a, j]
                pb = R[a, j - 1]
                sb = R[a, j + 1]
                if j == i + 2:
                    MID = no(R[a, i + 1], e, l, s)
                elif j > i + 2:
                    MID, _ = cat(MID, no(R[a, j - 1], e, l, s), Dm)
                cnt += 4
                if j == i + 1:
                    delta = Dm[pa, v] + Dm[u, sb] - Dm[pa, u] - Dm[v, sb]
                else:
                    delta = Dm[pa, v] + Dm[v, sa] - Dm[pa, u] - Dm[u, sa] + Dm[pb, u] + Dm[u, sb] - Dm[pb, v] - Dm[v, sb]
                if delta >= best:
                    continue
                S1, ok1 = cat(pre(PD, PE, PL, R, a, i - 1), no(v, e, l, s), Dm)
                if not ok1:
                    continue
                if j > i + 1:
                    S1, ok1 = cat(S1, MID, Dm)
                    if not ok1:
                        continue
                S2, ok2 = cat(S1, no(u, e, l, s), Dm)
                if not ok2:
                    continue
                _, ok3 = cat(S2, suf(SD, SE, SL, R, Ln, a, j + 1), Dm)
                if ok3:
                    best, ba, bi, bj = delta, a, i, j
    return best, ba, bi, bj, 0, 0, cnt


# --------------------------------------------------------------------------------------------- 8/9. CROSS
@njit(cache=True)
def k_cross(R, Ln, PD, PE, PL, SD, SE, SL, CL, Dm, dem, e, l, s, Q, maxlen, minmax):
    m = R.shape[0]
    best, ba, bi, bla, bb, bj, blb, cnt = 0, -1, -1, -1, -1, -1, -1, 0
    for a in range(m):
        La = Ln[a]
        loadA = CL[a, La + 1]
        for b in range(a + 1, m):
            Lb = Ln[b]
            loadB = CL[b, Lb + 1]
            for i in range(1, La + 1):
                for la in range(1, maxlen + 1):
                    if i + la - 1 > La:
                        break
                    dA = CL[a, i + la - 1] - CL[a, i - 1]
                    pa = R[a, i - 1]
                    sa = R[a, i + la]
                    fA = R[a, i]
                    tA = R[a, i + la - 1]
                    SgA, okA = seg_nos(R, a, i, la, False, Dm, e, l, s)
                    for j in range(1, Lb + 1):
                        for lb in range(1, maxlen + 1):
                            if j + lb - 1 > Lb:
                                break
                            if max(la, lb) < minmax:
                                continue
                            dB = CL[b, j + lb - 1] - CL[b, j - 1]
                            if loadA - dA + dB > Q or loadB - dB + dA > Q:
                                continue
                            cnt += 4
                            pb = R[b, j - 1]
                            sb = R[b, j + lb]
                            fB = R[b, j]
                            tB = R[b, j + lb - 1]
                            delta = (Dm[pa, fB] + Dm[tB, sa] - Dm[pa, fA] - Dm[tA, sa]
                                     + Dm[pb, fA] + Dm[tA, sb] - Dm[pb, fB] - Dm[tB, sb])
                            if delta >= best:
                                continue
                            if not okA:
                                continue
                            SgB, okB = seg_nos(R, b, j, lb, False, Dm, e, l, s)
                            if not okB:
                                continue
                            S1, ok1 = cat(pre(PD, PE, PL, R, a, i - 1), SgB, Dm)
                            if not ok1:
                                continue
                            _, ok2 = cat(S1, suf(SD, SE, SL, R, Ln, a, i + la), Dm)
                            if not ok2:
                                continue
                            S3, ok3 = cat(pre(PD, PE, PL, R, b, j - 1), SgA, Dm)
                            if not ok3:
                                continue
                            _, ok4 = cat(S3, suf(SD, SE, SL, R, Ln, b, j + lb), Dm)
                            if ok4:
                                best, ba, bi, bla, bb, bj, blb = delta, a, i, la, b, j, lb
    return best, ba, bi, bla, bb, bj, blb, cnt


# --------------------------------------------------------------------------------------------- empacotamento e aplicacao (Python)
def _arrays(inst, routes):
    rs = [list(r) for r in routes if r]
    m = len(rs)
    W = max(len(r) for r in rs) + 2
    R = np.zeros((m, W), np.int64)
    Ln = np.zeros(m, np.int64)
    for k, r in enumerate(rs):
        R[k, 1:len(r) + 1] = r
        Ln[k] = len(r)
    return rs, R, Ln


def _contexto(inst, routes):
    rs, R, Ln = _arrays(inst, routes)
    pre_ = prepara(R, Ln, inst.D, inst.dem, inst.e, inst.l, inst.s)
    return rs, (R, Ln) + pre_ + (inst.D, inst.dem, inst.e, inst.l, inst.s, inst.Q)


def _limpa(rs):
    return [r for r in rs if r]


def _op(kernel, aplica):
    def best_fn(inst, routes):
        rs, args = _contexto(inst, routes)
        res = kernel(*args)
        inst.chamadas += int(res[-1])
        if res[0] < 0:
            return True, _limpa(aplica(rs, res))
        return False, routes
    return best_fn


def _ap_reloc1(rs, r):
    _, a, i, b, j, _, _ = r
    u = rs[a][i - 1]
    del rs[a][i - 1]
    rs[b].insert(j, u)
    return rs


def _ap_reloc2(rs, r):
    _, a, i, b, j, o, _ = r
    seg = rs[a][i - 1:i + 1]
    del rs[a][i - 1:i + 1]
    if o == 1:
        seg = seg[::-1]
    rs[b][j:j] = seg
    return rs


def _ap_swap11(rs, r):
    _, a, i, b, j, _, _ = r
    rs[a][i - 1], rs[b][j - 1] = rs[b][j - 1], rs[a][i - 1]
    return rs


def _ap_twoopt(rs, r):
    _, a, i, b, j, _, _ = r
    ta, tb = rs[a][i:], rs[b][j:]
    rs[a] = rs[a][:i] + tb
    rs[b] = rs[b][:j] + ta
    return rs


def _ap_move1(rs, r):
    _, a, i, j, _, _, _ = r
    u = rs[a][i - 1]
    rt = rs[a]
    if j < i - 1:
        del rt[i - 1]
        rt.insert(j, u)
    else:
        rt.insert(j, u)
        del rt[i - 1]
    return rs


def _ap_or23(rs, r):
    _, a, i, k, j, o, _ = r
    rt = rs[a]
    seg = rt[i - 1:i - 1 + k]
    if o == 1:
        seg = seg[::-1]
    if j < i - 1:
        del rt[i - 1:i - 1 + k]
        rt[j:j] = seg
    else:
        rt[j:j] = seg
        del rt[i - 1:i - 1 + k]
    return rs


def _ap_swapintra(rs, r):
    _, a, i, j, _, _, _ = r
    rs[a][i - 1], rs[a][j - 1] = rs[a][j - 1], rs[a][i - 1]
    return rs


def _cross_op(maxlen, minmax):
    def best_fn(inst, routes):
        rs, args = _contexto(inst, routes)
        res = k_cross(*args, maxlen, minmax)
        inst.chamadas += int(res[-1])
        if res[0] < 0:
            _, a, i, la, b, j, lb, _ = res
            sa, sb = rs[a][i - 1:i - 1 + la], rs[b][j - 1:j - 1 + lb]
            rs[a][i - 1:i - 1 + la] = sb
            rs[b][j - 1:j - 1 + lb] = sa
            return True, _limpa(rs)
        return False, routes
    return best_fn


# --------------------------------------------------------------------------------------------- movimentos aleatorios (shake)
def _viavel_rotas(inst, rotas):
    return all(inst.viavel_rota(r) for r in rotas)


def _rand_wrap(gerador, tentativas=60):
    def f(inst, routes, rng):
        rs = [list(r) for r in routes if r]
        for _ in range(tentativas):
            novo = gerador(inst, rs, rng)
            if novo is None:
                continue
            novo = [r for r in novo if r]
            if _viavel_rotas(inst, novo):
                return novo
        return None
    return f


def _g_move1(inst, rs, rng):
    a = rng.randrange(len(rs))
    if len(rs[a]) < 2:
        return None
    novo = [list(r) for r in rs]
    i = rng.randrange(len(novo[a]))
    u = novo[a].pop(i)
    j = rng.randrange(len(novo[a]) + 1)
    novo[a].insert(j, u)
    return novo


def _g_swap11(inst, rs, rng):
    if len(rs) < 2:
        return None
    a, b = rng.sample(range(len(rs)), 2)
    novo = [list(r) for r in rs]
    i, j = rng.randrange(len(novo[a])), rng.randrange(len(novo[b]))
    novo[a][i], novo[b][j] = novo[b][j], novo[a][i]
    if sum(inst.dem[c] for c in novo[a]) > inst.Q or sum(inst.dem[c] for c in novo[b]) > inst.Q:
        return None
    return novo


def _g_reloc1(inst, rs, rng):
    if len(rs) < 2:
        return None
    a, b = rng.sample(range(len(rs)), 2)
    novo = [list(r) for r in rs]
    u = novo[a].pop(rng.randrange(len(novo[a])))
    novo[b].insert(rng.randint(0, len(novo[b])), u)
    return novo if sum(inst.dem[c] for c in novo[b]) <= inst.Q else None


def _g_reloc2(inst, rs, rng):
    cand = [k for k, r in enumerate(rs) if len(r) >= 2]
    if not cand or len(rs) < 2:
        return None
    a = rng.choice(cand)
    b = rng.choice([k for k in range(len(rs)) if k != a])
    novo = [list(r) for r in rs]
    i = rng.randrange(len(novo[a]) - 1)
    seg = novo[a][i:i + 2]
    del novo[a][i:i + 2]
    if rng.random() < 0.5:
        seg = seg[::-1]
    j = rng.randint(0, len(novo[b]))
    novo[b][j:j] = seg
    return novo if sum(inst.dem[c] for c in novo[b]) <= inst.Q else None


def _g_twoopt(inst, rs, rng):
    if len(rs) < 2:
        return None
    a, b = rng.sample(range(len(rs)), 2)
    i, j = rng.randint(0, len(rs[a])), rng.randint(0, len(rs[b]))
    novo = [list(r) for r in rs]
    novo[a] = rs[a][:i] + rs[b][j:]
    novo[b] = rs[b][:j] + rs[a][i:]
    for r in (novo[a], novo[b]):
        if sum(inst.dem[c] for c in r) > inst.Q:
            return None
    return novo


def _nao_usado(inst, routes, rng):
    return None


# nome, melhor_fn, aleatoria_fn.  P5 = os cinco primeiros (analogo dos 5 operadores originais); a ordem da lista e a de custo nominal
# crescente dentro de cada grupo e e fixa ANTES dos pilotos (Entrada 116).
P9 = [("Move(1)", _op(k_move1, _ap_move1), _rand_wrap(_g_move1)),
      ("Swap(1,1)", _op(k_swap11, _ap_swap11), _rand_wrap(_g_swap11)),
      ("Relocate(1,0)", _op(k_reloc1, _ap_reloc1), _rand_wrap(_g_reloc1)),
      ("Relocate(2,0)", _op(k_reloc2, _ap_reloc2), _rand_wrap(_g_reloc2)),
      ("2-opt*", _op(k_twoopt_star, _ap_twoopt), _rand_wrap(_g_twoopt)),
      ("Or-opt(2-3)", _op(k_oropt23, _ap_or23), _nao_usado),
      ("Swap-intra", _op(k_swapintra, _ap_swapintra), _nao_usado),
      ("CROSS(<=3)", _cross_op(3, 2), _nao_usado),
      ("CROSS(4-5)", _cross_op(5, 4), _nao_usado)]
P5 = P9[:5]
