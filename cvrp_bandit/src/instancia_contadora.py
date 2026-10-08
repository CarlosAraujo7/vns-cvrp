"""
instancia_contadora.py
======================
Proxy de CVRPInstance que (a) usa uma MATRIZ de distancias pre-calculada
(valores identicos aos de `CVRPInstance.dist`, calculados por ele mesmo) e
(b) CONTA as chamadas a `dist`. Serve de medida DETERMINISTICA de custo
computacional dos operadores de vizinhanca na Etapa 3 da Direcao A
(diario, Entradas 75+): como os operadores tem custos que diferem em ~800x
(0,1 ms a 83 ms por aplicacao), um orcamento em "numero de aplicacoes"
trataria o SWAP* como gratuito; tempo de CPU seria ruidoso neste
notebook (nucleos P e E de velocidades diferentes). O numero de avaliacoes
de distancia e independente de hardware e reproduzivel dado o seed.

Os operadores de neighborhoods*.py so usam: dist, demands, capacity,
depot_id, n_vehicles_ref, vehicles_exact. Tudo mais e delegado a instancia
original (`__getattr__`). A funcao de custo (`solution_cost`) NAO passa
por este proxy -- fica na instancia original -- e portanto nao conta.
"""


class InstanciaContadora:
    def __init__(self, inst):
        self._inst = inst
        self.demands = inst.demands
        self.capacity = inst.capacity
        self.depot_id = inst.depot_id
        self.n_vehicles_ref = inst.n_vehicles_ref
        self.vehicles_exact = inst.vehicles_exact
        ids = list(inst.coords)
        tam = max(ids) + 1
        D = [[0.0] * tam for _ in range(tam)]
        for i in ids:
            for j in ids:
                D[i][j] = inst.dist(i, j)
        self._D = D
        self.chamadas = 0

    def dist(self, i, j):
        self.chamadas += 1
        return self._D[i][j]

    def __getattr__(self, nome):
        # so e chamado para atributos nao definidos acima
        return getattr(self._inst, nome)
