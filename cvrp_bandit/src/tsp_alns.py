"""
tsp_alns.py
===========
Tabelas de operadores do ALNS para o TSP (Estudo 13). O TSP e tratado como o caso de UM veiculo sem restricao de
capacidade da maquinaria de alns.py (Estado, destruicao aleatoria/pior/relacionada, reparos guloso/regret-2/barato, com
`max_rotas=1`); a unica diferenca de operadores em relacao ao Estudo 9 e que a destruicao "rota" (que no TSP seria
remover o passeio inteiro) e substituida pela destruicao por SEGMENTO: remove q clientes consecutivos do passeio, a
partir de uma posicao sorteada (nenhuma avaliacao de distancia). Como a relacionada usa demandas iguais, a relatedness
no TSP e so a distancia.

Destruir (indices): 0 aleatoria, 1 pior, 2 relacionada, 3 segmento.   Reparar: 0 guloso, 1 regret2, 2 barato.
"""
from alns import destruir_aleatoria, destruir_pior, destruir_relacionada, REPARAR


def destruir_segmento(est, q, rng, qmax=None):
    r = est.rotas[0]
    q = min(q, len(r))
    s = rng.randint(0, len(r) - q)
    removidos = r[s:s + q]
    est.rotas[0] = r[:s] + r[s + q:]
    est._refaz(0)
    est.compactar()
    return list(removidos)


DESTRUIR_TSP = [("aleatoria", destruir_aleatoria), ("pior", destruir_pior), ("relacionada", destruir_relacionada),
                ("segmento", destruir_segmento)]
REPARAR_TSP = REPARAR
MAX_ROTAS_TSP = 1
