"""
tw_alns.py
==========
Operadores do ALNS para o CVRPTW (Estudo 16): a maquinaria de alns.py (Estado, destruicoes aleatoria/pior/rota, laco de
alns_loop.rodar_alns) com viabilidade de JANELAS DE TEMPO nas insercoes. Mesma estrutura do Estudo 9 e do Estudo 13:
4 destruicoes x 3 reparos, as duas escolhas por duas instancias independentes da mesma politica.

Destruir (indices): 0 aleatoria, 1 pior, 2 relacionada (Shaw com termo de tempo, como Ropke & Pisinger 2006 para o VRPTW:
                    phi = 9 distancia, chi = 3 demanda, psi = 2 inicio da janela), 3 rota.
Reparar:            0 guloso, 1 regret-2, 2 barato (ordem aleatoria).

Viabilidade de uma insercao em O(1) por posicao: cada rota guarda o inicio de servico `a[k]` de cada no e o INICIO DE SERVICO
MAIS TARDIO `L[k]` que ainda deixa o resto da rota viavel (L[k] = min(l[no], L[k+1] - s[no] - d(no, prox))). Inserir c entre u e
v (posicoes p-1 e p) e viavel sse
    inicio_c = max(a[p-1] + s[u] + d(u,c), e[c]) <= l[c]   e   inicio_c + s[c] + d(c,v) <= L[p].
Todas as grandezas sao inteiras (decimos), logo a conta e exata. A viabilidade da solucao completa e SEMPRE conferida de novo
por simulacao (TWInst.is_feasible) no laco do ALNS; como as distancias sao truncadas a uma casa, a desigualdade triangular pode
falhar por 1 decimo e a REMOCAO de um cliente pode tornar inviavel uma rota apertada: essas rotas (`ok[i] = False`) nao recebem
insercoes e a candidata e rejeitada pelo laco (contada em `inviaveis`).

Contabilidade: igual a de alns.py. Cada rota examinada para um cliente custa len(nos) - 1 avaliacoes de distancia (so se a
capacidade permite examina-la); abrir rota nova custa 1; as arestas existentes sao lidas da matriz sem custo.
"""
from heapq import nsmallest

from alns import (Estado, destruir_aleatoria, destruir_pior, destruir_rota, _remover_conjunto,
                  _escolha_aleatorizada)
from tw_inst import TWInst

INF = float("inf")


class TWAlnsInst(TWInst):
    """TWInst com a interface que alns.py espera (dist metrada, matriz em listas, demandas, capacidade, deposito)."""

    @classmethod
    def desde(cls, inst):
        o = cls.__new__(cls)
        o.__dict__.update(inst.__dict__)
        o._D = inst.D.tolist()
        o._e = inst.e.tolist()
        o._l = inst.l.tolist()
        o._s = inst.s.tolist()
        o.demands = inst.dem.tolist()
        o.capacity = int(inst.Q)
        o.depot_id = 0
        o.chamadas = 0
        return o

    def dist(self, a, b):
        self.chamadas += 1
        return self._D[a][b]


# ------------------------------------------------------------------------------------------ estado com tempos
class EstadoTW(Estado):
    """Estado do ALNS com os tempos de cada rota: `nos`, inicio de servico `a`, mais tardio `L` e viabilidade exata `ok`."""

    def __init__(self, inst, rotas, dmax=1.0):
        super().__init__(inst, rotas, dmax)
        self.D, self.e, self.l, self.s = inst._D, inst._e, inst._l, inst._s
        self.nos, self.a, self.L, self.ok = [], [], [], []
        for r in self.rotas:
            self._anexa(self._calc(r))

    def _calc(self, r):
        D, e, l, s = self.D, self.e, self.l, self.s
        nos = [0] + r + [0]
        n = len(nos)
        a = [0] * n
        ok = True
        t = 0
        for k in range(1, n):
            u, v = nos[k - 1], nos[k]
            t += s[u] + D[u][v]
            if t < e[v]:
                t = e[v]
            a[k] = t
            if t > l[v]:
                ok = False
        L = [0] * n
        L[n - 1] = l[0]
        for k in range(n - 2, -1, -1):
            v, w = nos[k], nos[k + 1]
            x = L[k + 1] - s[v] - D[v][w]
            L[k] = x if x < l[v] else l[v]
        return nos, a, L, ok

    def _anexa(self, t):
        self.nos.append(t[0])
        self.a.append(t[1])
        self.L.append(t[2])
        self.ok.append(t[3])

    def _refaz(self, i):
        super()._refaz(i)
        self.nos[i], self.a[i], self.L[i], self.ok[i] = self._calc(self.rotas[i])

    def abrir(self, c):
        i = super().abrir(c)
        self._anexa(self._calc(self.rotas[i]))
        return i

    def compactar(self):
        keep = [i for i, r in enumerate(self.rotas) if r]
        super().compactar()
        self.nos = [self.nos[i] for i in keep]
        self.a = [self.a[i] for i in keep]
        self.L = [self.L[i] for i in keep]
        self.ok = [self.ok[i] for i in keep]


def melhor_posicao_tw(est, i, c):
    """(delta de custo, posicao na lista da rota) da melhor insercao VIAVEL de c na rota i, ou None. Conta len(nos)-1 avaliacoes."""
    nos, a, L, arest = est.nos[i], est.a[i], est.L[i], est.arest[i]
    D, e, l, s = est.D, est.e, est.l, est.s
    est.inst.chamadas += len(nos) - 1
    Dc = D[c]
    ec, lc, sc = e[c], l[c], s[c]
    melhor, bp = INF, -1
    for p in range(1, len(nos)):
        u, v = nos[p - 1], nos[p]
        duc = D[u][c]
        t = a[p - 1] + s[u] + duc
        if t < ec:
            t = ec
        if t > lc or t + sc + Dc[v] > L[p]:
            continue
        delta = duc + Dc[v] - arest[p - 1]
        if delta < melhor:
            melhor, bp = delta, p - 1
    return (melhor, bp) if bp >= 0 else None


def _opcoes_tw(est, c):
    """{indice de rota: (custo, posicao)} de c nas rotas onde cabe por capacidade e por janelas."""
    out = {}
    for i in range(len(est.rotas)):
        if est.ok[i] and est.cargas[i] + est.dem[c] <= est.Q:
            r = melhor_posicao_tw(est, i, c)
            if r is not None:
                out[i] = r
    return out


# ------------------------------------------------------------------------------------------ destruicao
def destruir_relacionada_tw(est, q, rng, p=6, phi=9.0, chi=3.0, psi=2.0):
    inst = est.inst
    todos = [c for r in est.rotas for c in r]
    dmax = est.dmax
    qmax = max(est.dem[c] for c in todos) or 1
    H = est.l[0] or 1
    removidos = [rng.choice(todos)]
    resto = set(todos) - set(removidos)
    while len(removidos) < min(q, len(todos)):
        r = rng.choice(removidos)
        ordem = sorted(resto, key=lambda j: (phi * inst.dist(r, j) / dmax + chi * abs(est.dem[r] - est.dem[j]) / qmax
                                             + psi * abs(est.e[r] - est.e[j]) / H, j))
        c = ordem[_escolha_aleatorizada(rng, len(ordem), p)]
        removidos.append(c)
        resto.discard(c)
    return _remover_conjunto(est, set(removidos))


# ------------------------------------------------------------------------------------------ reparo
def reparar_arrependimento_tw(est, removidos, rng, k=2, max_rotas=None):
    """k=1 guloso, k=2 regret-2, com janelas de tempo. Mesmas decisoes de alns.reparar_arrependimento_v1 (desempate por (custo, rota));
    abrir rota nova (custo 2 d(0,c), sempre viavel sozinho) e (custo, -1) enquanto len(rotas) < max_rotas."""
    inst, dep = est.inst, est.dep
    pend = list(removidos)
    op = {c: _opcoes_tw(est, c) for c in pend}
    novo = {c: inst.dist(dep, c) * 2 for c in pend}

    def top2(c, pode_abrir):
        t = [(v[0], i) for i, v in op[c].items()]
        if pode_abrir:
            t.append((novo[c], -1))
        if len(t) <= 2:
            t.sort()
            return t
        return nsmallest(2, t)

    while pend:
        melhor_c, melhor_chave, escolha = None, None, None
        pode_abrir = max_rotas is None or len(est.rotas) < max_rotas
        for c in pend:
            custos = top2(c, pode_abrir)
            if not custos:
                return False
            if k == 1:
                chave = (custos[0][0], 0)
            elif len(custos) == 1:
                chave = (-INF, custos[0][0])
            else:
                chave = (-(custos[1][0] - custos[0][0]), custos[0][0])
            if melhor_chave is None or chave < melhor_chave:
                melhor_chave, melhor_c, escolha = chave, c, custos[0]
        c = melhor_c
        pend.remove(c)
        i = escolha[1]
        if i == -1:
            i = est.abrir(c)
        else:
            est.inserir(i, op[c][i][1], c)
        for cc in pend:
            op[cc].pop(i, None)
            if est.ok[i] and est.cargas[i] + est.dem[cc] <= est.Q:
                r = melhor_posicao_tw(est, i, cc)
                if r is not None:
                    op[cc][i] = r
    return True


def reparar_barato_tw(est, removidos, rng, max_rotas=None):
    inst, dep = est.inst, est.dep
    ordem = list(removidos)
    rng.shuffle(ordem)
    for c in ordem:
        op = _opcoes_tw(est, c)
        melhor_i = min(op, key=lambda i: (op[i][0], i)) if op else None
        custo_novo = inst.dist(dep, c) * 2
        pode_abrir = max_rotas is None or len(est.rotas) < max_rotas
        if melhor_i is None and not pode_abrir:
            return False
        if melhor_i is None or (pode_abrir and custo_novo < op[melhor_i][0]):
            est.abrir(c)
        else:
            est.inserir(melhor_i, op[melhor_i][1], c)
    return True


DESTRUIR_TW = [("aleatoria", destruir_aleatoria), ("pior", destruir_pior), ("relacionada", destruir_relacionada_tw),
               ("rota", destruir_rota)]
REPARAR_TW = [("guloso", lambda est, rem, rng, mr: reparar_arrependimento_tw(est, rem, rng, 1, mr)),
              ("regret2", lambda est, rem, rng, mr: reparar_arrependimento_tw(est, rem, rng, 2, mr)),
              ("barato", lambda est, rem, rng, mr: reparar_barato_tw(est, rem, rng, mr))]
