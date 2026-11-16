"""
tw_alns_ext.py
==============
Operadores ADICIONAIS do ALNS do CVRPTW para o Estudo 28 (diario, Entrada 148 -- pre-registro): o conjunto de 21 operadores (12 destruicoes x 9 reparos) do ALNS do CVRP (alns_ext.py, Estudo 21) com
viabilidade de JANELAS DE TEMPO nas insercoes (tw_alns.py). Os conjuntos continuam ANINHADOS: P7 = o do Estudo 16 (tw_alns.DESTRUIR_TW x REPARAR_TW, 4 x 3), P12 (+ radial, sequencia, demanda; regret3, demanda_dec) e P21
(+ aleatoria_leve, aleatoria_forte, pior_leve, relacionada_forte, rota_pequena; regret4, longe_primeiro, barato_ruidoso, blink). Os quatro primeiros operadores de destruicao e os tres primeiros de reparo sao OS MESMOS
OBJETOS de tw_alns.py (a regressao exata do Estudo 16 depende disso); o operador de rota continua no indice 3.

As destruicoes de alns_ext.py nao dependem de janelas (so removem clientes; a remocao pode, no maximo, tornar inviavel uma rota apertada por arredondamento, caso que o laco ja trata) e sao reusadas; a
`relacionada_forte` usa a relacionada com termo de tempo de tw_alns. Os reparos sao os de alns_ext.py com a insercao viavel de tw_alns (`melhor_posicao_tw`, `_opcoes_tw`): a regra do arrependimento-k, a
insercao sequencial em ordem dada (demanda decrescente, mais longe primeiro) e as versoes perturbadas (ruido uniforme de ate eta no custo de cada posicao viavel; cada posicao viavel ignorada com
probabilidade `blink`; se todas as posicoes viaveis foram ignoradas usa-se a melhor sem perturbacao). Contabilidade: igual a de tw_alns.py (len(nos) - 1 avaliacoes por rota examinada).
"""
from heapq import nsmallest

from alns import INF
from alns_ext import (ETA_RUIDO, PROB_BLINK, _escala, destruir_aleatoria_forte, destruir_aleatoria_leve, destruir_demanda, destruir_pior_leve, destruir_radial, destruir_rota_pequena,
                      destruir_sequencia)
from tw_alns import DESTRUIR_TW, REPARAR_TW, _opcoes_tw, destruir_relacionada_tw, melhor_posicao_tw


# ------------------------------------------------------------------------------------------------ destruicao
def destruir_relacionada_forte_tw(est, q, rng):
    return destruir_relacionada_tw(est, _escala(q, 1.7), rng)


# ------------------------------------------------------------------------------------------------ reparo
def reparar_arrependimento_tw_k(est, removidos, rng, k=3, max_rotas=None):
    """Arrependimento-k com janelas de tempo (k >= 1): a mesma estrutura de `tw_alns.reparar_arrependimento_tw` (as opcoes por cliente, atualizadas so na rota modificada), com as k menores opcoes por cliente;
    chave = soma_{j=2..k}(c_j - c_1), infinita se ha menos de k opcoes; k = 1 e o guloso e k = 2 reproduz `tw_alns.reparar_arrependimento_tw` exatamente (tests/test_tw_alns_ext.py)."""
    inst, dep = est.inst, est.dep
    pend = list(removidos)
    op = {c: _opcoes_tw(est, c) for c in pend}
    novo = {c: inst.dist(dep, c) * 2 for c in pend}

    def topk(c, pode_abrir):
        t = [(v[0], i) for i, v in op[c].items()]
        if pode_abrir:
            t.append((novo[c], -1))
        if len(t) <= k:
            t.sort()
            return t
        return nsmallest(k, t)

    while pend:
        melhor_c, melhor_chave, escolha = None, None, None
        pode_abrir = max_rotas is None or len(est.rotas) < max_rotas
        for c in pend:
            custos = topk(c, pode_abrir)
            if not custos:
                return False
            if k == 1:
                chave = (custos[0][0], 0)
            elif len(custos) < k:
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
            if est.ok[i] and est.cargas[i] + est.dem[cc] <= est.Q:
                r = melhor_posicao_tw(est, i, cc)
                if r is not None:
                    op[cc][i] = r
    return True


def _melhor_posicao_tw_pert(est, i, c, rng, eta, blink):
    """Como `tw_alns.melhor_posicao_tw` (len(nos)-1 avaliacoes) com ruido aditivo uniforme em [-eta, eta] e/ou ignorando cada posicao VIAVEL com probabilidade `blink`; sem perturbacao e identica a ela."""
    nos, a, L, arest = est.nos[i], est.a[i], est.L[i], est.arest[i]
    D, e, l, s = est.D, est.e, est.l, est.s
    est.inst.chamadas += len(nos) - 1
    Dc = D[c]
    ec, lc, sc = e[c], l[c], s[c]
    melhor, bp = INF, -1
    sem, bp_sem = INF, -1
    for p in range(1, len(nos)):
        u, v = nos[p - 1], nos[p]
        duc = D[u][c]
        t = a[p - 1] + s[u] + duc
        if t < ec:
            t = ec
        if t > lc or t + sc + Dc[v] > L[p]:
            continue
        delta = duc + Dc[v] - arest[p - 1]
        if delta < sem:
            sem, bp_sem = delta, p - 1
        if blink and rng.random() < blink:
            continue
        if eta:
            delta = max(0.0, delta + rng.uniform(-eta, eta))
        if delta < melhor:
            melhor, bp = delta, p - 1
    if bp < 0:                                        # nenhuma posicao viavel, ou todas ignoradas: a melhor sem perturbacao (se houver)
        return (sem, bp_sem) if bp_sem >= 0 else None
    return melhor, bp


def _opcoes_tw_pert(est, c, rng, eta, blink):
    out = {}
    for i in range(len(est.rotas)):
        if est.ok[i] and est.cargas[i] + est.dem[c] <= est.Q:
            r = _melhor_posicao_tw_pert(est, i, c, rng, eta, blink)
            if r is not None:
                out[i] = r
    return out


def inserir_sequencial_tw(est, ordem, rng, max_rotas=None, eta=0.0, blink=0.0):
    """Cada cliente de `ordem`, na sua vez, na melhor posicao viavel (ou numa rota nova se for mais barato e permitido); com `eta` e `blink` = 0 e o mesmo laco de `tw_alns.reparar_barato_tw` depois do embaralhamento."""
    inst, dep = est.inst, est.dep
    for c in ordem:
        op = _opcoes_tw(est, c) if not (eta or blink) else _opcoes_tw_pert(est, c, rng, eta, blink)
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


def reparar_demanda_dec_tw(est, removidos, rng, max_rotas=None):
    ordem = sorted(removidos, key=lambda c: (-est.dem[c], rng.random()))
    return inserir_sequencial_tw(est, ordem, rng, max_rotas)


def reparar_longe_primeiro_tw(est, removidos, rng, max_rotas=None):
    inst, dep = est.inst, est.dep
    ordem = sorted(removidos, key=lambda c: -inst.dist(dep, c))
    return inserir_sequencial_tw(est, ordem, rng, max_rotas)


def reparar_barato_ruidoso_tw(est, removidos, rng, max_rotas=None):
    ordem = list(removidos)
    rng.shuffle(ordem)
    return inserir_sequencial_tw(est, ordem, rng, max_rotas, eta=ETA_RUIDO * est.dmax)


def reparar_blink_tw(est, removidos, rng, max_rotas=None):
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
    return inserir_sequencial_tw(est, ordem, rng, max_rotas, blink=PROB_BLINK)


# ------------------------------------------------------------------------------------------------ tabelas
DESTRUIR_P7 = list(DESTRUIR_TW)
REPARAR_P7 = list(REPARAR_TW)
DESTRUIR_P12 = DESTRUIR_P7 + [("radial", destruir_radial), ("sequencia", destruir_sequencia), ("demanda", destruir_demanda)]
REPARAR_P12 = REPARAR_P7 + [("regret3", lambda est, rem, rng, mr: reparar_arrependimento_tw_k(est, rem, rng, 3, mr)), ("demanda_dec", reparar_demanda_dec_tw)]
DESTRUIR_P21 = DESTRUIR_P12 + [("aleatoria_leve", destruir_aleatoria_leve), ("aleatoria_forte", destruir_aleatoria_forte), ("pior_leve", destruir_pior_leve),
                               ("relacionada_forte", destruir_relacionada_forte_tw), ("rota_pequena", destruir_rota_pequena)]
REPARAR_P21 = REPARAR_P12 + [("regret4", lambda est, rem, rng, mr: reparar_arrependimento_tw_k(est, rem, rng, 4, mr)), ("longe_primeiro", reparar_longe_primeiro_tw),
                             ("barato_ruidoso", reparar_barato_ruidoso_tw), ("blink", reparar_blink_tw)]
POOLS_TW = {"P7": (DESTRUIR_P7, REPARAR_P7), "P12": (DESTRUIR_P12, REPARAR_P12), "P21": (DESTRUIR_P21, REPARAR_P21)}
assert [len(d) + len(r) for d, r in POOLS_TW.values()] == [7, 12, 21]
assert DESTRUIR_P21[:4] == DESTRUIR_TW and REPARAR_P21[:3] == REPARAR_TW and DESTRUIR_P21[3][0] == "rota"
