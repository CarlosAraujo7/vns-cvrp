#!/usr/bin/env python3
"""
uchoa_gerador_core.py
======================
Refatoração do gerador OFICIAL de instâncias do conjunto X da CVRPLIB
(achado real #28, diario_pesquisa_novo.md Entrada 37 -- fonte:
https://galgos.inf.puc-rio.br/cvrplib/uploads/files/xml100/generator.py,
mesmo gerador usado para criar as instâncias X reais deste projeto) como
FUNÇÃO reutilizável, em vez de script standalone via linha de comando --
necessário para chamar o gerador em processo, dentro do loop de treino do
Kool et al., sem overhead de subprocess a cada amostra.

IMPORTANTE sobre fidelidade: esta função replica não só as FÓRMULAS do
gerador oficial, mas também a ORDEM EXATA de todas as chamadas à fonte de
aleatoriedade (mesma sequência de randint/uniform/shuffle, incluindo a
peculiaridade de o script oficial sortear `nSeeds=randint(2,6)` e
`route_size=uniform(...)` SEMPRE, incondicionalmente, antes mesmo de saber
`cust_pos` -- o valor de nSeeds é descartado depois se cust_pos==1, mas o
sorteio já aconteceu e consumiu um número da sequência) e usa um `set()`
real do Python para acumular os pontos, replicando inclusive a ORDEM DE
ITERAÇÃO do set original (que depende de hash, não da ordem de inserção --
mas para tuplas de inteiros o hash do CPython não é salgado por
PYTHONHASHSEED, então a ordem de iteração É determinística e reprodutível
entre processos, dado o mesmo conteúdo e histórico de inserção).

Consequência prática: para a MESMA seed, esta função produz depósito,
clientes (na mesma ordem), demandas (após shuffle) e capacidade IDÊNTICOS
byte-a-byte ao gerar a mesma instância via `generator.py` original --
verificado em `validar_uchoa_gerador.py` (achado real #31, ver diário)
rodando o script oficial como subprocesso e comparando diretamente. Isso é
mais forte do que "estatisticamente equivalente": qualquer erro de fórmula
teria feito o teste falhar imediatamente.

A lógica abaixo é uma tradução linha a linha do gerador oficial (nenhuma
constante foi "corrigida" para bater com a prosa do paper -- ver a
discrepância documentada na Entrada 37: o CÓDIGO oficial usa nSeeds~U{2,6}
e peso 2^(-d/40), diferente do que o texto do paper diz -- usamos o código
porque é o que de fato gerou as instâncias reais que este projeto usa).

Uso:
    from uchoa_gerador_core import gerar_instancia_uchoa
    depot, clientes, demandas, capacidade = gerar_instancia_uchoa(
        n=50, depot_pos=2, cust_pos=2, demand_type=7, avg_route_size=3,
        rng=random.Random(42),
    )
    # depot: (x, y) em [0, 1000]. clientes: lista de (x, y), len==n.
    # demandas: lista de int, mesmo tamanho/ordem de clientes.
    # capacidade: int.
"""
import math
import random

MAX_COORD = 1000
DECAY = 40

# (demandMin, demandMax) por demand_type (1-indexado, ver help do gerador oficial).
# Arrays têm 9 entradas no original (só as 7 primeiras são usadas -- deixamos
# como está, fiel ao original, embora as 2 últimas nunca sejam acessadas).
_DEMAND_MIN = [1, 1, 5, 1, 50, 1, 51, 50, 1]
_DEMAND_MAX = [1, 10, 10, 100, 100, 50, 100, 100, 10]
_ROUTE_SIZE_RANGE = {1: (3, 5), 2: (5, 8), 3: (8, 12), 4: (12, 16), 5: (16, 25), 6: (25, 50)}

_DEMAND_MIN_QUAD_PAR = 51
_DEMAND_MAX_QUAD_PAR = 100
_DEMAND_MIN_GRANDE = 50
_DEMAND_MAX_GRANDE = 100
_FRACAO_GRANDES_POR_ROTA = 1.5
_DEMAND_MIN_PEQUENA = 1
_DEMAND_MAX_PEQUENA = 10


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def gerar_instancia_uchoa(n, depot_pos, cust_pos, demand_type, avg_route_size, rng=None):
    """Gera UMA instância sintética seguindo fielmente o gerador oficial
    do conjunto X (ver docstring do módulo, inclusive quanto à ordem exata
    de consumo de aleatoriedade). Parâmetros (mesma convenção do gerador
    oficial via CLI):

    depot_pos: 1=aleatório, 2=centralizado, 3=no canto.
    cust_pos: 1=aleatório, 2=clusterizado, 3=metade aleatório/metade clusterizado.
    demand_type: 1-7 (ver _DEMAND_MIN/_DEMAND_MAX; 7 = "poucos grandes, muitos pequenos").
    avg_route_size: 1-6 (muito curta a ultra longa -- ver _ROUTE_SIZE_RANGE).
    rng: instância de random.Random (recomendado passar para reprodutibilidade
         e para não interferir/depender do estado global do módulo `random`
         -- se None, usa o módulo random global, replicando o comportamento
         do script original via `random.seed(...)`).

    Retorna (depot_xy, clientes_xy, demandas, capacidade) -- coordenadas em
    [0, MAX_COORD], demandas inteiras (mesma ordem/tamanho de clientes_xy),
    capacidade inteira. NENHUM arquivo é escrito (função pura, para uso em
    memória no treino).
    """
    r = rng if rng is not None else random

    if not (1 <= demand_type <= 7):
        raise ValueError(f"demand_type fora de faixa: {demand_type}")
    if avg_route_size not in _ROUTE_SIZE_RANGE:
        raise ValueError(f"avg_route_size fora de faixa: {avg_route_size}")
    if depot_pos not in (1, 2, 3):
        raise ValueError(f"depot_pos fora de faixa: {depot_pos}")
    if cust_pos not in (1, 2, 3):
        raise ValueError(f"cust_pos fora de faixa: {cust_pos}")

    # ---- ORDEM EXATA do script oficial a partir daqui (comentários citam
    # a linha correspondente em external/uchoa_generator/generator.py) ----

    # linha 73: sorteado INCONDICIONALMENTE, antes de saber cust_pos.
    n_seeds_sorteado = r.randint(2, 6)

    # linha 80: tamanho médio de rota, sorteado logo em seguida.
    lo, hi = _ROUTE_SIZE_RANGE[avg_route_size]
    route_size = r.uniform(lo, hi)

    # linhas 92-102: posicionamento do depósito.
    if depot_pos == 1:
        depot = (r.randint(0, MAX_COORD), r.randint(0, MAX_COORD))
    elif depot_pos == 2:
        c = int(MAX_COORD / 2.0)
        depot = (c, c)
    else:  # depot_pos == 3
        depot = (0, 0)

    # linhas 105-117: quantos clientes são aleatórios vs. clusterizados.
    if cust_pos == 3:
        n_rand_cust = int(n / 2.0)
        n_seeds = n_seeds_sorteado
    elif cust_pos == 2:
        n_rand_cust = 0
        n_seeds = n_seeds_sorteado
    else:  # cust_pos == 1
        n_rand_cust = n
        n_seeds = 0  # sobrescreve o sorteio da linha 73 (que já ocorreu)
    n_clust_cust = n - n_rand_cust

    S = set()  # mesmo tipo (set) do original -- iteração determinística

    def _sorteia_ponto_livre():
        x_ = r.randint(0, MAX_COORD)
        y_ = r.randint(0, MAX_COORD)
        while (x_, y_) in S or (x_, y_) == depot:
            x_ = r.randint(0, MAX_COORD)
            y_ = r.randint(0, MAX_COORD)
        return (x_, y_)

    # linhas 120-126: clientes aleatórios.
    for _ in range(1, n_rand_cust + 1):
        S.add(_sorteia_ponto_livre())

    seeds = []
    # linhas 132-175: clientes clusterizados (accept-reject).
    if n_clust_cust > 0:
        if n_clust_cust < n_seeds:
            raise ValueError("numero de clusters maior que clientes a clusterizar")

        for _ in range(n_seeds):
            p = _sorteia_ponto_livre()
            S.add(p)
            seeds.append(p)

        max_weight = 0.0
        for si in seeds:
            w_ij = 0.0
            for sj in seeds:
                w_ij += 2 ** (-_dist(si, sj) / DECAY)
            if w_ij > max_weight:
                max_weight = w_ij
        norm_factor = 1.0 / max_weight

        n_s = n_rand_cust + n_seeds
        while n_s < n:
            cand = _sorteia_ponto_livre()
            weight = 0.0
            for sj in seeds:
                weight += 2 ** (-_dist(cand, sj) / DECAY)
            weight *= norm_factor
            aceita = r.uniform(0, 1)
            if aceita <= weight:
                S.add(cand)
                n_s += 1
            # se rejeitado: NÃO adiciona a S (igual ao original) -- o mesmo
            # ponto poderia, em princípio, ser sorteado de novo depois.

    V = [depot] + list(S)  # ordem de list(S) é a mesma ordem usada no original

    # linhas 180-217: demandas, na ordem de V (V[0]=depot é pulado).
    d_min, d_max = _DEMAND_MIN[demand_type - 1], _DEMAND_MAX[demand_type - 1]
    D = []
    soma_demandas = 0
    max_demanda = 0
    for i in range(2, n + 2):
        pv = V[i - 1]
        j = int((d_max - d_min + 1) * r.uniform(0, 1) + d_min)
        if demand_type == 6:
            par_quad = (pv[0] < MAX_COORD / 2.0 and pv[1] < MAX_COORD / 2.0) or \
                       (pv[0] >= MAX_COORD / 2.0 and pv[1] >= MAX_COORD / 2.0)
            if par_quad:
                j = int((_DEMAND_MAX_QUAD_PAR - _DEMAND_MIN_QUAD_PAR + 1) * r.uniform(0, 1) + _DEMAND_MIN_QUAD_PAR)
        if demand_type == 7:
            if i < (n / route_size) * _FRACAO_GRANDES_POR_ROTA:
                j = int((_DEMAND_MAX_GRANDE - _DEMAND_MIN_GRANDE + 1) * r.uniform(0, 1) + _DEMAND_MIN_GRANDE)
            else:
                j = int((_DEMAND_MAX_PEQUENA - _DEMAND_MIN_PEQUENA + 1) * r.uniform(0, 1) + _DEMAND_MIN_PEQUENA)
        D.append(j)
        if j > max_demanda:
            max_demanda = j
        soma_demandas += j

    # linhas 212-217: capacidade (calculada ANTES do shuffle, igual ao original).
    if soma_demandas == n:
        capacidade = math.floor(route_size)
    else:
        capacidade = max(max_demanda, math.ceil(route_size * soma_demandas / n))
    capacidade = int(capacidade)

    # linhas 233-234: shuffle das demandas (não dos pontos) -- exceto tipo 6,
    # cuja demanda é amarrada ao quadrante do próprio ponto.
    if demand_type != 6:
        r.shuffle(D)

    clientes_xy = V[1:]
    return depot, clientes_xy, D, capacidade


# 3 posicoes de deposito x 3 de cliente x 7 de demanda x 6 de tamanho de rota
COMBINACOES_VALIDAS = [
    (dp, cp, dt, rs)
    for dp in (1, 2, 3) for cp in (1, 2, 3) for dt in range(1, 8) for rs in range(1, 7)
]


def amostrar_combinacao(rng):
    """Sorteia uniformemente uma das 378 combinações (depot_pos, cust_pos,
    demand_type, avg_route_size) -- mesmo espaço de atributos usado pelo
    dataset XML100 (Queiroga et al. 2022), citado na Entrada 37."""
    return rng.choice(COMBINACOES_VALIDAS)
