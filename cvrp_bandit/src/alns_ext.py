"""
alns_ext.py
===========
Operadores ADICIONAIS do ALNS do CVRP para o Estudo 21 (diario, Entrada 136 -- pre-registro): varredura do TAMANHO DO CONJUNTO DE OPERADORES.

Tres conjuntos ANINHADOS (destruicao x reparo; a selecao continua sendo feita por duas politicas independentes, como em alns_loop.py):
  P7   os do Estudo 9 (alns.DESTRUIR x alns.REPARAR): 4 x 3 = 7 logits da mistura estatica
  P12  P7 + destruir {radial, sequencia, demanda} + reparar {regret3, demanda_dec}: 7 x 5 = 12 logits
  P21  P12 + destruir {aleatoria_leve, aleatoria_forte, pior_leve, relacionada_forte, rota_pequena}
           + reparar {regret4, longe_primeiro, barato_ruidoso, blink}: 12 x 9 = 21 logits
Os quatro primeiros operadores de destruicao e os tres primeiros de reparo sao OS MESMOS OBJETOS de alns.py (a regressao exata dos Estudos 9, 12 e 20
depende disso); o operador de rota continua no indice 3 (alns_loop.rodar_alns passa `qmax` so a ele).

Definicoes (todas da literatura de ALNS/LNS para o VRP; fixadas ANTES de qualquer dado de avaliacao):
  radial             remove o cliente-semente sorteado e os q-1 mais proximos (Pisinger e Ropke 2007, remocao por proximidade); custo: n avaliacoes
  sequencia          remocao de cadeias (SISR, Christiaens e Vanden Berghe 2020, simplificada): cadeias contiguas de comprimento sorteado em [1, 10] nas
                     rotas visitadas por ordem de proximidade ao cliente-semente, ate somar q clientes; custo: n avaliacoes
  demanda            remove os q clientes de demanda mais parecida com a do cliente-semente (desempate aleatorio); nenhuma avaliacao de distancia
  aleatoria_leve     aleatoria com 0,4 q clientes (minimo 2)          aleatoria_forte   aleatoria com 2 q clientes
  pior_leve          pior (p=3) com 0,4 q clientes (minimo 2)         relacionada_forte relacionada (Shaw) com 1,7 q clientes
  rota_pequena       remove a rota com MENOS clientes (desempate aleatorio); se so ha uma rota, cai em 'pior'
  regret3, regret4   insercao por arrependimento-k: chave = soma_{j=2..k}(c_j - c_1); infinita se ha menos de k opcoes (Ropke e Pisinger 2006)
  demanda_dec        clientes em ordem DECRESCENTE de demanda, cada um na melhor posicao
  longe_primeiro     clientes em ordem decrescente de distancia ao deposito, cada um na melhor posicao
  barato_ruidoso     ordem aleatoria, melhor posicao com RUIDO uniforme em [-eta, eta] somado ao custo de cada posicao (eta = 0,025 x maior distancia;
                     Ropke e Pisinger 2006)
  blink              ordem sorteada (aleatoria 4/11, demanda decrescente 4/11, longe 2/11, perto 1/11), melhor posicao ignorando cada posicao candidata
                     com probabilidade 0,01 (SISR)
Contabilidade: igual a de alns.py (toda avaliacao de distancia passa por `inst.dist`).
"""
from heapq import nsmallest

from alns import (INF, DESTRUIR, REPARAR, _nos, _opcoes, _remover_conjunto, destruir_aleatoria, destruir_pior, destruir_relacionada,
                  melhor_posicao)

ETA_RUIDO = 0.025
PROB_BLINK = 0.01
LMAX_CADEIA = 10


# ------------------------------------------------------------------------------------------------ destruicao
def destruir_radial(est, q, rng):
    inst = est.inst
    todos = [c for r in est.rotas for c in r]
    q = min(q, len(todos))
    semente = rng.choice(todos)
    ordem = sorted(todos, key=lambda j: inst.dist(semente, j))          # estavel: desempate pela ordem das rotas
    return _remover_conjunto(est, set(ordem[:q]))


def destruir_sequencia(est, q, rng, lmax=LMAX_CADEIA):
    inst = est.inst
    todos = [c for r in est.rotas for c in r]
    q = min(q, len(todos))
    semente = rng.choice(todos)
    rota_de = {c: i for i, r in enumerate(est.rotas) for c in r}
    ordem = sorted(todos, key=lambda j: inst.dist(semente, j))
    vistas, removidos = set(), []
    for c in ordem:
        i = rota_de[c]
        if i in vistas:
            continue
        vistas.add(i)
        r = est.rotas[i]
        faltam = q - len(removidos)
        L = min(len(r), rng.randint(1, lmax), faltam)
        j = r.index(c)
        ini = max(0, min(j - rng.randint(0, L - 1), len(r) - L))        # a cadeia contem c
        removidos.extend(r[ini:ini + L])
        if len(removidos) >= q:
            break
    return _remover_conjunto(est, set(removidos))


def destruir_demanda(est, q, rng):
    todos = [c for r in est.rotas for c in r]
    q = min(q, len(todos))
    semente = rng.choice(todos)
    ordem = sorted(todos, key=lambda j: (abs(est.dem[j] - est.dem[semente]), rng.random()))
    return _remover_conjunto(est, set(ordem[:q]))


def _escala(q, f, minimo=2):
    return max(minimo, int(round(f * q)))


def destruir_aleatoria_leve(est, q, rng):
    return destruir_aleatoria(est, _escala(q, 0.4), rng)


def destruir_aleatoria_forte(est, q, rng):
    return destruir_aleatoria(est, _escala(q, 2.0), rng)


def destruir_pior_leve(est, q, rng):
    return destruir_pior(est, _escala(q, 0.4), rng)


def destruir_relacionada_forte(est, q, rng):
    return destruir_relacionada(est, _escala(q, 1.7), rng)


def destruir_rota_pequena(est, q, rng):
    if len(est.rotas) <= 1:
        return destruir_pior(est, q, rng)
    menor = min(len(r) for r in est.rotas)
    i = rng.choice([i for i, r in enumerate(est.rotas) if len(r) == menor])
    removidos = list(est.rotas[i])
    est.rotas[i] = []
    est._refaz(i)
    est.compactar()
    return removidos


# ------------------------------------------------------------------------------------------------ reparo
def reparar_arrependimento_k(est, removidos, rng, k=3, max_rotas=None):
    """Arrependimento-k (k >= 3) com a mesma estrutura incremental do regret-2 de alns.py (as k menores opcoes por cliente; recalcula so quando a
    rota modificada estava entre elas). Conferido contra `reparar_arrependimento_k_ingenuo` em tests/test_alns_ext.py."""
    inst, dep = est.inst, est.dep
    pend = list(removidos)
    op = {c: _opcoes(est, c) for c in pend}
    novo = {c: inst.dist(dep, c) * 2.0 for c in pend}
    incremental = max_rotas is None

    def topk(c, pode_abrir):
        t = [(v[0], i) for i, v in op[c].items()]
        if pode_abrir:
            t.append((novo[c], -1))
        if len(t) <= k:
            t.sort()
            return t
        return nsmallest(k, t)

    tk = {c: topk(c, True) for c in pend} if incremental else None
    while pend:
        melhor_c, melhor_chave, escolha = None, None, None
        pode_abrir = max_rotas is None or len(est.rotas) < max_rotas
        for c in pend:
            custos = tk[c] if incremental else topk(c, pode_abrir)
            if not custos:
                return False
            if len(custos) < k:
                chave = (-INF, custos[0][0])
            else:
                chave = (-sum(custos[j][0] - custos[0][0] for j in range(1, k)), custos[0][0])
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
                lst = tk[cc]
                if any(tt[1] == i for tt in lst):
                    tk[cc] = topk(cc, True)
                elif entrada is not None and (len(lst) < k or entrada < lst[-1]):
                    lst.append(entrada)
                    lst.sort()
                    del lst[k:]
    return True


def reparar_arrependimento_k_ingenuo(est, removidos, rng, k=3, max_rotas=None):
    """Referencia lenta (todas as opcoes ordenadas a cada passo, como `reparar_arrependimento_v1`): so para os testes."""
    inst, dep = est.inst, est.dep
    pend = list(removidos)
    op = {c: _opcoes(est, c) for c in pend}
    novo = {c: inst.dist(dep, c) * 2.0 for c in pend}
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
            if len(custos) < k:
                chave = (-INF, custos[0][0])
            else:
                chave = (-sum(custos[j][0] - custos[0][0] for j in range(1, k)), custos[0][0])
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
            if est.cargas[i] + est.dem[cc] <= est.Q:
                op[cc][i] = melhor_posicao(inst, _nos(dep, est.rotas[i]), est.arest[i], cc)
    return True


def _melhor_posicao_pert(inst, nos, arestas, c, rng, eta, blink):
    """Como `melhor_posicao` (len(nos)-1 avaliacoes), com ruido aditivo uniforme em [-eta, eta] e/ou ignorando cada posicao com prob. `blink`.
    Se todas as posicoes foram ignoradas, usa a melhor sem perturbacao."""
    d = inst.dist
    dcd = d(c, nos[0])
    ult = len(nos) - 1
    prev_d = dcd
    melhor, bp = INF, None
    sem, bp_sem = INF, 1
    for p in range(1, ult + 1):
        dv = dcd if p == ult else d(c, nos[p])
        delta = prev_d + dv - arestas[p - 1]
        prev_d = dv
        if delta < sem:
            sem, bp_sem = delta, p
        if blink and rng.random() < blink:
            continue
        if eta:
            delta = max(0.0, delta + rng.uniform(-eta, eta))
        if delta < melhor:
            melhor, bp = delta, p
    if bp is None:
        melhor, bp = sem, bp_sem
    return melhor, bp - 1


def _opcoes_pert(est, c, rng, eta, blink):
    inst, dep = est.inst, est.dep
    out = {}
    for i in range(len(est.rotas)):
        if est.cargas[i] + est.dem[c] <= est.Q:
            out[i] = _melhor_posicao_pert(inst, _nos(dep, est.rotas[i]), est.arest[i], c, rng, eta, blink)
    return out


def inserir_sequencial(est, ordem, rng, max_rotas=None, eta=0.0, blink=0.0):
    """Cada cliente de `ordem`, na sua vez, na melhor posicao viavel (ou numa rota nova se for mais barato e permitido); com `eta` e `blink` = 0
    e o mesmo laco de `alns.reparar_barato` depois do embaralhamento."""
    inst, dep = est.inst, est.dep
    for c in ordem:
        op = _opcoes(est, c) if not (eta or blink) else _opcoes_pert(est, c, rng, eta, blink)
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


def reparar_demanda_dec(est, removidos, rng, max_rotas=None):
    ordem = sorted(removidos, key=lambda c: (-est.dem[c], rng.random()))
    return inserir_sequencial(est, ordem, rng, max_rotas)


def reparar_longe_primeiro(est, removidos, rng, max_rotas=None):
    inst, dep = est.inst, est.dep
    ordem = sorted(removidos, key=lambda c: -inst.dist(dep, c))
    return inserir_sequencial(est, ordem, rng, max_rotas)


def reparar_barato_ruidoso(est, removidos, rng, max_rotas=None):
    ordem = list(removidos)
    rng.shuffle(ordem)
    return inserir_sequencial(est, ordem, rng, max_rotas, eta=ETA_RUIDO * est.dmax)


def reparar_blink(est, removidos, rng, max_rotas=None):
    inst, dep = est.inst, est.dep
    u = rng.random() * 11.0
    if u < 4.0:
        ordem = list(removidos)
        rng.shuffle(ordem)
    elif u < 8.0:
        ordem = sorted(removidos, key=lambda c: (-est.dem[c], rng.random()))
    elif u < 10.0:
        ordem = sorted(removidos, key=lambda c: -inst.dist(dep, c))
    else:
        ordem = sorted(removidos, key=lambda c: inst.dist(dep, c))
    return inserir_sequencial(est, ordem, rng, max_rotas, blink=PROB_BLINK)


# ------------------------------------------------------------------------------------------------ tabelas
DESTRUIR_P7 = list(DESTRUIR)
REPARAR_P7 = list(REPARAR)
DESTRUIR_P12 = DESTRUIR_P7 + [("radial", destruir_radial), ("sequencia", destruir_sequencia), ("demanda", destruir_demanda)]
REPARAR_P12 = REPARAR_P7 + [("regret3", lambda est, rem, rng, mr: reparar_arrependimento_k(est, rem, rng, 3, mr)),
                            ("demanda_dec", reparar_demanda_dec)]
DESTRUIR_P21 = DESTRUIR_P12 + [("aleatoria_leve", destruir_aleatoria_leve), ("aleatoria_forte", destruir_aleatoria_forte),
                               ("pior_leve", destruir_pior_leve), ("relacionada_forte", destruir_relacionada_forte),
                               ("rota_pequena", destruir_rota_pequena)]
REPARAR_P21 = REPARAR_P12 + [("regret4", lambda est, rem, rng, mr: reparar_arrependimento_k(est, rem, rng, 4, mr)),
                             ("longe_primeiro", reparar_longe_primeiro), ("barato_ruidoso", reparar_barato_ruidoso),
                             ("blink", reparar_blink)]
POOLS = {"P7": (DESTRUIR_P7, REPARAR_P7), "P12": (DESTRUIR_P12, REPARAR_P12), "P21": (DESTRUIR_P21, REPARAR_P21)}
assert [len(d) + len(r) for d, r in POOLS.values()] == [7, 12, 21]
assert DESTRUIR_P21[:4] == DESTRUIR and REPARAR_P21[:3] == REPARAR and DESTRUIR_P21[3][0] == "rota"
