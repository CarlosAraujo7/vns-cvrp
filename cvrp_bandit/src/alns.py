"""
alns.py
=======
ALNS (adaptive large neighborhood search; Ropke & Pisinger, 2006) para o CVRP, no MESMO arcabouco dos Estudos 1-8
(contador de avaliacoes de distancia via InstanciaContadora, solucao inicial fixada em disco, politicas de selecao
intercambiaveis). Estudo 9 da Direcao A: AOS no cenario CANONICO de operadores ESTOCASTICOS.

Operadores de destruicao (cada um remove q clientes):
  0 aleatoria   q clientes sorteados
  1 pior        remove os clientes de maior "ganho de remocao" (com aleatorizacao y^p, p=3)
  2 relacionada Shaw: remove clientes parecidos (distancia e demanda) com um ja removido (aleatorizacao y^p, p=6)
  3 rota        remove todos os clientes de uma rota sorteada (so rotas com <= qmax clientes; senao, 'pior')
Operadores de reparo (reinsercao dos removidos; viabilidade de capacidade; abre rota nova se permitido):
  0 guloso      insere sempre o par (cliente, posicao) de menor custo
  1 regret2     insere o cliente de maior arrependimento (2a melhor rota - melhor rota)
  2 barato      clientes em ordem aleatoria, cada um na melhor posicao (custo O(q n); fraco e barato)

Contabilidade: toda avaliacao de custo de insercao e de relacao conta em `instance.chamadas` (distancia metrada
via `instance.dist`); o comprimento das arestas ja existentes e lido da matriz sem contar (pre-computado, como
nos operadores de vizinhanca). A funcao de custo da solucao completa (`solution_cost`) nao conta.
"""
import math

INF = float("inf")
EPS = 1e-9


# ------------------------------------------------------------------ utilidades --------------------------------
def _nos(dep, rota):
    return [dep] + rota + [dep]


def _arestas(Dm, nos):
    return [Dm[nos[i]][nos[i + 1]] for i in range(len(nos) - 1)]


def melhor_posicao(inst, nos, arestas, c):
    """Menor custo de inserir c entre dois nos consecutivos de `nos` (depositos nas pontas). len(nos)-1 avaliacoes."""
    d = inst.dist
    dcd = d(c, nos[0])
    ult = len(nos) - 1
    prev_d = dcd
    melhor, bp = INF, 1
    for p in range(1, ult + 1):
        dv = dcd if p == ult else d(c, nos[p])
        delta = prev_d + dv - arestas[p - 1]
        if delta < melhor:
            melhor, bp = delta, p
        prev_d = dv
    return melhor, bp - 1          # posicao na lista da rota (0 = antes do 1o cliente)


class Estado:
    """Solucao parcial: rotas (listas de clientes), cargas e arestas por rota."""

    def __init__(self, inst, rotas, dmax=1.0):
        self.inst = inst
        self.dmax = dmax
        self.dep = inst.depot_id
        self.Dm = inst._D
        self.dem = inst.demands
        self.Q = inst.capacity + 1e-6
        self.rotas = [list(r) for r in rotas if r]
        self.cargas = [sum(self.dem[c] for c in r) for r in self.rotas]
        self.arest = [_arestas(self.Dm, _nos(self.dep, r)) for r in self.rotas]

    def _refaz(self, i):
        self.arest[i] = _arestas(self.Dm, _nos(self.dep, self.rotas[i]))
        self.cargas[i] = sum(self.dem[c] for c in self.rotas[i])

    def remover(self, i, pos):
        c = self.rotas[i].pop(pos)
        self._refaz(i)
        return c

    def inserir(self, i, pos, c):
        self.rotas[i].insert(pos, c)
        self._refaz(i)

    def abrir(self, c):
        self.rotas.append([c])
        self.cargas.append(self.dem[c])
        self.arest.append(_arestas(self.Dm, _nos(self.dep, [c])))
        return len(self.rotas) - 1

    def compactar(self):
        keep = [i for i, r in enumerate(self.rotas) if r]
        self.rotas = [self.rotas[i] for i in keep]
        self.cargas = [self.cargas[i] for i in keep]
        self.arest = [self.arest[i] for i in keep]


def _escolha_aleatorizada(rng, n, p):
    return min(n - 1, int(n * (rng.random() ** p)))


# ------------------------------------------------------------------ destruicao ---------------------------------
def destruir_aleatoria(est, q, rng):
    cli = [(i, j) for i, r in enumerate(est.rotas) for j in range(len(r))]
    alvo = {est.rotas[i][j] for i, j in rng.sample(cli, min(q, len(cli)))}
    return _remover_conjunto(est, alvo)


def _remover_conjunto(est, alvo):
    removidos = []
    for i, r in enumerate(est.rotas):
        if any(c in alvo for c in r):
            removidos.extend(c for c in r if c in alvo)
            est.rotas[i] = [c for c in r if c not in alvo]
            est._refaz(i)
    est.compactar()
    return removidos


def destruir_pior(est, q, rng, p=3):
    inst, dep = est.inst, est.dep
    ganho = {}
    for i, r in enumerate(est.rotas):
        nos = _nos(dep, r)
        for j, c in enumerate(r):
            prev, nxt = nos[j], nos[j + 2]
            ganho[c] = est.arest[i][j] + est.arest[i][j + 1] - inst.dist(prev, nxt)
    # posicao de cada cliente (rota, indice) e remocao um a um com atualizacao dos vizinhos
    removidos = []
    cand = list(ganho)
    for _ in range(min(q, len(cand))):
        cand.sort(key=lambda c: -ganho[c])
        c = cand.pop(_escolha_aleatorizada(rng, len(cand), p))
        removidos.append(c)
        for i, r in enumerate(est.rotas):
            if c in r:
                j = r.index(c)
                est.remover(i, j)
                nos = _nos(dep, est.rotas[i])
                for jj in (j - 1, j):
                    if 0 <= jj < len(est.rotas[i]):
                        cc = est.rotas[i][jj]
                        ganho[cc] = est.arest[i][jj] + est.arest[i][jj + 1] - inst.dist(nos[jj], nos[jj + 2])
                break
    est.compactar()
    return removidos


def destruir_relacionada(est, q, rng, p=6, phi=9.0, chi=3.0):
    inst = est.inst
    todos = [c for r in est.rotas for c in r]
    dmax = est.dmax
    qmax = max(est.dem[c] for c in todos) or 1
    removidos = [rng.choice(todos)]
    resto = set(todos) - set(removidos)
    while len(removidos) < min(q, len(todos)):
        r = rng.choice(removidos)
        ordem = sorted(resto, key=lambda j: phi * inst.dist(r, j) / dmax + chi * abs(est.dem[r] - est.dem[j]) / qmax)
        c = ordem[_escolha_aleatorizada(rng, len(ordem), p)]
        removidos.append(c)
        resto.discard(c)
    return _remover_conjunto(est, set(removidos))


def destruir_rota(est, q, rng, qmax=None):
    qmax = q if qmax is None else qmax
    cand = [i for i, r in enumerate(est.rotas) if 0 < len(r) <= qmax]
    if not cand:
        return destruir_pior(est, q, rng)
    i = rng.choice(cand)
    removidos = list(est.rotas[i])
    est.rotas[i] = []
    est._refaz(i)
    est.compactar()
    return removidos


DESTRUIR = [("aleatoria", destruir_aleatoria), ("pior", destruir_pior), ("relacionada", destruir_relacionada),
            ("rota", destruir_rota)]


# ------------------------------------------------------------------ reparo -------------------------------------
def _opcoes(est, c, so_rota=None):
    """{indice de rota: (custo, posicao)} de c nas rotas viaveis (ou so em `so_rota`)."""
    inst, dep = est.inst, est.dep
    out = {}
    idx = range(len(est.rotas)) if so_rota is None else [so_rota]
    for i in idx:
        if est.cargas[i] + est.dem[c] <= est.Q:
            out[i] = melhor_posicao(inst, _nos(dep, est.rotas[i]), est.arest[i], c)
    return out


def reparar_arrependimento_v1(est, removidos, rng, k=2, max_rotas=None):
    # versao ORIGINAL (usada nas 325 primeiras execucoes do Estudo 9); mantida para o teste de regressao exata
    """k=1: insercao gulosa; k=2: regret-2. Mantem, para cada cliente, o melhor custo POR ROTA e so recalcula a
    rota modificada a cada insercao (tecnica padrao)."""
    inst, dep = est.inst, est.dep
    pend = list(removidos)
    op = {c: _opcoes(est, c) for c in pend}
    novo = {c: inst.dist(dep, c) * 2.0 for c in pend}     # custo de abrir uma rota so para c
    while pend:
        melhor_c, melhor_chave, escolha = None, None, None
        pode_abrir = max_rotas is None or len(est.rotas) < max_rotas
        for c in pend:
            custos = sorted([(v[0], i) for i, v in op[c].items()])
            if pode_abrir:
                custos.append((novo[c], -1))
                custos.sort()
            if not custos:
                return False
            if k == 1:
                chave = (custos[0][0], 0.0)
            elif len(custos) == 1:
                chave = (-INF, custos[0][0])          # so uma opcao: arrependimento infinito, prioridade maxima
            else:
                chave = (-(custos[1][0] - custos[0][0]), custos[0][0])
            if melhor_chave is None or chave < melhor_chave:
                melhor_chave, melhor_c, escolha = chave, c, custos[0]
        c = melhor_c
        pend.remove(c)
        i = escolha[1]
        if i == -1:
            i = est.abrir(c)
            novo_idx = i
        else:
            est.inserir(i, op[c][i][1], c)
            novo_idx = None
        for cc in pend:
            op[cc].pop(i, None)
            if est.cargas[i] + est.dem[cc] <= est.Q:
                op[cc][i] = melhor_posicao(inst, _nos(dep, est.rotas[i]), est.arest[i], cc)
    return True


def reparar_arrependimento(est, removidos, rng, k=2, max_rotas=None):
    """Mesmas decisoes que `reparar_arrependimento_v1` (regressao exata: tests/test_alns.py), com custo de CPU muito menor:
    em vez de ordenar as opcoes de TODAS as rotas de cada cliente pendente a cada insercao (O(q^2 R log R) por iteracao,
    inviavel com 100 rotas), mantem as duas menores opcoes (custo, rota) por cliente e so as recalcula quando a rota
    modificada estava entre elas. Desempates por (custo, indice da rota), como na versao original; a opcao 'abrir rota
    nova' e (custo, -1)."""
    from heapq import nsmallest
    inst, dep = est.inst, est.dep
    pend = list(removidos)
    op = {c: _opcoes(est, c) for c in pend}
    novo = {c: inst.dist(dep, c) * 2.0 for c in pend}     # custo de abrir uma rota so para c
    incremental = max_rotas is None                       # sem limite de rotas, abrir rota nova e sempre permitido

    def top2(c, pode_abrir):
        t = [(v[0], i) for i, v in op[c].items()]
        if pode_abrir:
            t.append((novo[c], -1))
        if len(t) <= 2:
            t.sort()
            return t
        return nsmallest(2, t)

    t2 = {c: top2(c, True) for c in pend} if incremental else None
    while pend:
        melhor_c, melhor_chave, escolha = None, None, None
        pode_abrir = max_rotas is None or len(est.rotas) < max_rotas
        for c in pend:
            custos = t2[c] if incremental else top2(c, pode_abrir)
            if not custos:
                return False
            if k == 1:
                chave = (custos[0][0], 0.0)
            elif len(custos) == 1:
                chave = (-INF, custos[0][0])          # so uma opcao: arrependimento infinito, prioridade maxima
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
            entrada = None
            if est.cargas[i] + est.dem[cc] <= est.Q:
                op[cc][i] = melhor_posicao(inst, _nos(dep, est.rotas[i]), est.arest[i], cc)
                entrada = (op[cc][i][0], i)
            if incremental:
                lst = t2[cc]
                if any(tt[1] == i for tt in lst):
                    t2[cc] = top2(cc, True)              # a entrada da rota modificada estava entre as duas menores
                elif entrada is not None and (len(lst) < 2 or entrada < lst[-1]):
                    lst.append(entrada)
                    lst.sort()
                    del lst[2:]
    return True


def reparar_barato(est, removidos, rng, max_rotas=None):
    inst, dep = est.inst, est.dep
    ordem = list(removidos)
    rng.shuffle(ordem)
    for c in ordem:
        op = _opcoes(est, c)
        melhor_i = min(op, key=lambda i: op[i][0]) if op else None
        custo_novo = inst.dist(dep, c) * 2.0
        pode_abrir = max_rotas is None or len(est.rotas) < max_rotas
        if melhor_i is None and not pode_abrir:
            return False
        if melhor_i is None or (pode_abrir and custo_novo < op[melhor_i][0]):
            est.abrir(c)
        else:
            est.inserir(melhor_i, op[melhor_i][1], c)
    return True


REPARAR = [("guloso", lambda est, rem, rng, mr: reparar_arrependimento(est, rem, rng, 1, mr)),
           ("regret2", lambda est, rem, rng, mr: reparar_arrependimento(est, rem, rng, 2, mr)),
           ("barato", lambda est, rem, rng, mr: reparar_barato(est, rem, rng, mr))]
