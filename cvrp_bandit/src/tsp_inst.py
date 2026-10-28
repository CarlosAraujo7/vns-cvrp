"""
tsp_inst.py
===========
Instancias do TSP simetrico euclidiano para o Estudo 13 (generalidade: o aparato dos Estudos 1-12 num segundo problema).

Convencoes (iguais as do projeto para o CVRP, para que o laco generico `aos_vnd.vnd_generico` e as politicas rodem sem
alteracao):
  * No 0 = ponto fixo de partida/chegada (o "deposito"); os demais N-1 nos sao os "clientes" (`n_customers`).
  * Uma solucao e `routes = [tour]`, com `tour` = lista dos clientes na ordem de visita (o no 0 fica implicito nas pontas).
  * Distancias EUC_2D da TSPLIB: nint(sqrt(dx^2 + dy^2)) = floor(x + 0.5). Para as sinteticas, coordenadas inteiras e a
    mesma regra de arredondamento.
  * `dist(i, j)` CONTA as chamadas em `chamadas` (medida de custo, avaliacoes de distancia); `solution_cost` e
    `is_feasible` NAO contam. Os operadores vetorizados (tsp_neighborhoods.py) somam a `chamadas` o numero de consultas
    d(.,.) que uma implementacao direta em lacos faria, com as arestas ja existentes lidas sem contar (como no CVRP).
"""
import gzip
import math
from pathlib import Path

import numpy as np


class TSPInst:
    def __init__(self, nome, coords, ref=None):
        self.nome = nome
        self.coords = np.asarray(coords, dtype=float)
        self.N = len(self.coords)
        self.n_customers = self.N - 1
        self.depot_id = 0
        dx = self.coords[:, 0][:, None] - self.coords[:, 0][None, :]
        dy = self.coords[:, 1][:, None] - self.coords[:, 1][None, :]
        self.Dn = np.floor(np.sqrt(dx * dx + dy * dy) + 0.5)      # nint, float64 (valores inteiros)
        self._D = self.Dn.tolist()
        self.chamadas = 0
        self.ref = ref
        # atributos do CVRP com que a maquinaria do ALNS (alns.py) trabalha: o TSP e o caso de UM veiculo sem restricao de
        # capacidade (demanda 1 por cliente e capacidade N); com max_rotas=1 nunca se abre outra rota.
        self.demands = [0] + [1] * (self.N - 1)
        self.capacity = float(self.N)
        self.n_vehicles_ref = 1
        self.vehicles_exact = False

    def dist(self, i, j):
        self.chamadas += 1
        return self._D[i][j]

    def solution_cost(self, routes):
        t = [0] + list(routes[0]) + [0]
        return float(sum(self._D[t[i]][t[i + 1]] for i in range(len(t) - 1)))

    def is_feasible(self, routes):
        r = routes[0] if routes else []
        if sorted(r) != list(range(1, self.N)):
            return False, "visita incompleta ou repetida"
        return True, ""


def ler_tsplib(caminho, ref=None):
    """Le um .tsp da TSPLIB com EDGE_WEIGHT_TYPE EUC_2D (aceita .gz). O no 1 da TSPLIB vira o no 0."""
    p = Path(caminho)
    txt = (gzip.open(p, "rt") if p.suffix == ".gz" else open(p, "r")).read().splitlines()
    tipo, coords, lendo = None, [], False
    for ln in txt:
        s = ln.strip()
        if s.startswith("EDGE_WEIGHT_TYPE"):
            tipo = s.split(":")[1].strip()
        elif s.startswith("NODE_COORD_SECTION"):
            lendo = True
        elif s.startswith("EOF"):
            break
        elif lendo and s:
            v = s.split()
            coords.append((int(v[0]), float(v[1]), float(v[2])))
    assert tipo == "EUC_2D", f"{p.name}: EDGE_WEIGHT_TYPE {tipo} nao suportado"
    coords.sort()
    assert [c[0] for c in coords] == list(range(1, len(coords) + 1))
    return TSPInst(p.stem.replace(".tsp", ""), [(x, y) for _, x, y in coords], ref=ref)


def escrever_tsplib(inst, caminho):
    """Escreve a instancia (coordenadas inteiras ou nao) em formato TSPLIB EUC_2D (para o LKH-3)."""
    with open(caminho, "w") as f:
        f.write(f"NAME : {inst.nome}\nTYPE : TSP\nDIMENSION : {inst.N}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(inst.coords, 1):
            f.write(f"{i} {x:.6f} {y:.6f}\n")
        f.write("EOF\n")


def custo_tour(inst, tour):
    """Custo do tour (lista de clientes) -- atalho."""
    return inst.solution_cost([tour])


def vizinho_mais_proximo(inst):
    """Solucao inicial determinista: vizinho mais proximo a partir do no 0 (desempate pelo menor indice)."""
    N = inst.N
    visto = np.zeros(N, dtype=bool)
    visto[0] = True
    atual, tour = 0, []
    for _ in range(N - 1):
        d = inst.Dn[atual].copy()
        d[visto] = np.inf
        prox = int(np.argmin(d))                       # argmin devolve o primeiro minimo
        tour.append(prox)
        visto[prox] = True
        atual = prox
    return tour
