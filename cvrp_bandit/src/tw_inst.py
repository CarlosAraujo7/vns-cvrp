"""
tw_inst.py
==========
Instancia do CVRPTW (Estudo 14) no formato de Solomon/CVRPLIB. Convencao de custo da CVRPLIB/DIMACS: a distancia (e o tempo de
viagem) entre dois nos e o comprimento euclidiano TRUNCADO a uma casa decimal; todas as grandezas sao guardadas em DECIMOS como
inteiros (distancia, janelas, servico), o que torna custo e viabilidade exatos. O objetivo e a distancia total (frota livre, ate
`nveic` veiculos); a rota viavel respeita capacidade e janelas de tempo (espera permitida; chegada antes de `l_i`; volta ao
deposito antes do horizonte). Rotas = listas de clientes (1..n); o deposito e o no 0.

O custo e a viabilidade de uma solucao sao SEMPRE recomputados do zero por simulacao (`custo_rota`, `viavel_rota`), nunca lidos de
um operador.
"""
import math
from pathlib import Path

import numpy as np


class TWInst:
    def __init__(self, nome, coords, dem, e, l, s, Q, nveic, ref=None):
        self.nome = nome
        self.coords = np.asarray(coords, dtype=float)
        self.N = len(self.coords)                       # nos, com o deposito
        self.n_customers = self.N - 1
        self.dem = np.asarray(dem, dtype=np.int64)
        self.e = np.asarray(e, dtype=np.int64)          # decimos
        self.l = np.asarray(l, dtype=np.int64)
        self.s = np.asarray(s, dtype=np.int64)
        self.Q = int(Q)
        self.nveic = int(nveic)
        self.ref = ref                                  # decimos (ou None)
        dx = self.coords[:, 0][:, None] - self.coords[:, 0][None, :]
        dy = self.coords[:, 1][:, None] - self.coords[:, 1][None, :]
        self.D = np.floor(10.0 * np.sqrt(dx * dx + dy * dy) + 1e-9).astype(np.int64)   # decimos, truncado
        self.chamadas = 0

    # ------------------------------------------------------------------ simulacao exata (referencia)
    def custo_rota(self, r):
        if not r:
            return 0
        t = [0] + list(r) + [0]
        return int(sum(self.D[t[i], t[i + 1]] for i in range(len(t) - 1)))

    def viavel_rota(self, r):
        if not r:
            return True
        if int(self.dem[list(r)].sum()) > self.Q:
            return False
        t, prev = 0, 0
        for c in r:
            t = max(t + self.D[prev, c], self.e[c])
            if t > self.l[c]:
                return False
            t += self.s[c]
            prev = c
        return t + self.D[prev, 0] <= self.l[0]

    def solution_cost(self, routes):
        return int(sum(self.custo_rota(r) for r in routes))

    def is_feasible(self, routes):
        vistos = sorted(c for r in routes for c in r)
        if vistos != list(range(1, self.N)):
            return False, "clientes nao cobertos exatamente uma vez"
        if len([r for r in routes if r]) > self.nveic:
            return False, "frota"
        for r in routes:
            if not self.viavel_rota(r):
                return False, "rota inviavel"
        return True, ""


def ler_solomon(caminho, ref=None):
    """Le um arquivo de Solomon (CVRPLIB: C101.txt). Retorna TWInst; os tempos vao a decimos de unidade."""
    p = Path(caminho)
    lin = p.read_text().splitlines()
    nome = lin[0].strip()
    i = next(k for k, x in enumerate(lin) if x.strip().startswith("NUMBER"))
    nveic, Q = (int(float(v)) for v in lin[i + 1].split()[:2])
    j = next(k for k, x in enumerate(lin) if x.strip().startswith("CUST NO"))
    rows = [x.split() for x in lin[j + 1:] if x.strip()]
    a = np.array([[float(v) for v in r[:7]] for r in rows])
    return TWInst(nome, a[:, 1:3], a[:, 3], np.rint(10 * a[:, 4]).astype(np.int64), np.rint(10 * a[:, 5]).astype(np.int64),
                  np.rint(10 * a[:, 6]).astype(np.int64), Q, nveic, ref)


def ler_bks(caminho):
    """Rotas de um .sol da CVRPLIB ('Route #k: a b c'); clientes numerados como no arquivo da instancia."""
    rotas = []
    for x in Path(caminho).read_text().splitlines():
        if x.startswith("Route"):
            rotas.append([int(v) for v in x.split(":")[1].split()])
    return rotas


def construcao_inicial(inst):
    """Insercao sequencial determinista: clientes por janela final crescente (desempate pelo indice); cada um vai para a posicao viavel de
    menor custo adicional entre as rotas existentes, ou abre rota. Sempre viavel se cada cliente e viavel sozinho."""
    ordem = sorted(range(1, inst.N), key=lambda c: (inst.l[c], c))
    rotas = []
    for c in ordem:
        melhor, mp = None, None
        for k, r in enumerate(rotas):
            t = [0] + r + [0]
            for q in range(len(r) + 1):
                nova = r[:q] + [c] + r[q:]
                if not inst.viavel_rota(nova):
                    continue
                add = inst.D[t[q], c] + inst.D[c, t[q + 1]] - inst.D[t[q], t[q + 1]]
                if melhor is None or add < melhor:
                    melhor, mp = add, (k, nova)
        if mp is None:
            rotas.append([c])
        else:
            rotas[mp[0]] = mp[1]
    return rotas
