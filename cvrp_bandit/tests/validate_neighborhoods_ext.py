#!/usr/bin/env python3
"""
validate_neighborhoods_ext.py
=============================
Valida os operadores novos de src/neighborhoods_ext.py contra uma
ENUMERACAO INDEPENDENTE POR FORCA BRUTA: para cada operador, constroi
explicitamente TODOS os vizinhos (fatiando listas), calcula o custo
completo de cada um via instance.solution_cost e confere viabilidade via
instance.is_feasible. Entao exige:
  - se existe vizinho viavel estritamente melhor: o operador devolve
    melhorou=True, a solucao e viavel e o custo e IGUAL ao melhor da
    enumeracao (best-improvement);
  - se nao existe: o operador devolve melhorou=False.
Cobre: solucoes aleatorias ruins (muitas melhoras), solucoes ja em otimo
local dos 5 operadores originais (ramo "sem melhora"), distancias inteiras
e de ponto flutuante, e instancias com K exato (restricao de veiculos).

Uso: python validate_neighborhoods_ext.py   (de dentro de tests/)
"""
import itertools
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from instance import CVRPInstance  # noqa: E402
from neighborhoods import NEIGHBORHOODS  # noqa: E402
from neighborhoods_ext import NEIGHBORHOODS_EXT_NOVOS  # noqa: E402
from vns import vnd  # noqa: E402

TOL = 1e-6


def instancia_aleatoria(rng, n, arredonda, k_exato):
    coords = {1: (rng.uniform(0, 100), rng.uniform(0, 100))}
    demands = {1: 0}
    for i in range(2, n + 2):
        coords[i] = (rng.uniform(0, 100), rng.uniform(0, 100))
        demands[i] = rng.randint(1, 9)
    total = sum(demands.values())
    cap = max(max(demands.values()), int(total / rng.randint(3, 5)) + 3)
    return CVRPInstance(name="rnd", dimension=n + 1, capacity=cap, coords=coords, demands=demands,
                        depot_id=1, round_distances=arredonda, optimum=0.0, reference_routes=[],
                        n_vehicles_ref=0, vehicles_exact=k_exato)


def solucao_aleatoria(inst, rng):
    clientes = inst.customers[:]
    rng.shuffle(clientes)
    rotas, atual, carga = [], [], 0
    for c in clientes:
        if carga + inst.demands[c] > inst.capacity:
            rotas.append(atual)
            atual, carga = [], 0
        atual.append(c)
        carga += inst.demands[c]
    if atual:
        rotas.append(atual)
    return rotas


# ------------------------- enumeradores independentes (forca bruta) --------
def viz_2opt_star(routes):
    n = len(routes)
    for ra in range(n):
        for rb in range(ra + 1, n):
            a, b = routes[ra], routes[rb]
            for i in range(len(a) + 1):
                for j in range(len(b) + 1):
                    nv = [list(r) for r in routes]
                    nv[ra] = a[:i] + b[j:]
                    nv[rb] = b[:j] + a[i:]
                    yield nv


def viz_or_opt(routes):
    for ridx, r in enumerate(routes):
        for L in (2, 3):
            for p in range(len(r) - L + 1):
                seg = r[p:p + L]
                rest = r[:p] + r[p + L:]
                for q in range(len(rest) + 1):
                    for s in (seg, seg[::-1]):
                        nv = [list(x) for x in routes]
                        nv[ridx] = rest[:q] + list(s) + rest[q:]
                        yield nv


def viz_swap_intra(routes):
    for ridx, r in enumerate(routes):
        for i, j in itertools.combinations(range(len(r)), 2):
            nv = [list(x) for x in routes]
            nv[ridx][i], nv[ridx][j] = nv[ridx][j], nv[ridx][i]
            yield nv


def viz_cross(routes):
    n = len(routes)
    for ra in range(n):
        for rb in range(ra + 1, n):
            a, b = routes[ra], routes[rb]
            for la in (1, 2, 3):
                for lb in (1, 2, 3):
                    if max(la, lb) < 2:
                        continue
                    for pa in range(len(a) - la + 1):
                        for pb in range(len(b) - lb + 1):
                            nv = [list(x) for x in routes]
                            nv[ra] = a[:pa] + b[pb:pb + lb] + a[pa + la:]
                            nv[rb] = b[:pb] + a[pa:pa + la] + b[pb + lb:]
                            yield nv


def viz_swap_star(routes):
    n = len(routes)
    for ra in range(n):
        for rb in range(ra + 1, n):
            a, b = routes[ra], routes[rb]
            for iu, u in enumerate(a):
                for iv, v in enumerate(b):
                    a_sem = a[:iu] + a[iu + 1:]
                    b_sem = b[:iv] + b[iv + 1:]
                    for pv in range(len(a_sem) + 1):
                        for pu in range(len(b_sem) + 1):
                            nv = [list(x) for x in routes]
                            nv[ra] = a_sem[:pv] + [v] + a_sem[pv:]
                            nv[rb] = b_sem[:pu] + [u] + b_sem[pu:]
                            yield nv


ENUMERADORES = {
    "2-opt*": viz_2opt_star, "Or-opt(2-3)": viz_or_opt, "Swap-intra": viz_swap_intra,
    "CROSS(<=3)": viz_cross, "SWAP*": viz_swap_star,
}


def conferir(inst, nome, best_fn, routes, rotulo):
    custo0 = inst.solution_cost(routes)
    melhor_bf = math.inf
    for nv in ENUMERADORES[nome](routes):
        nv = [r for r in nv if r]
        ok, _ = inst.is_feasible(nv)
        if ok:
            melhor_bf = min(melhor_bf, inst.solution_cost(nv))
    deve_melhorar = melhor_bf < custo0 - TOL
    melhorou, novo = best_fn(inst, routes)
    if deve_melhorar != melhorou:
        return f"{rotulo} {nome}: operador melhorou={melhorou}, forca bruta deve_melhorar={deve_melhorar} " \
               f"(custo0={custo0:.4f}, melhor_bf={melhor_bf:.4f})"
    if melhorou:
        ok, motivo = inst.is_feasible(novo)
        if not ok:
            return f"{rotulo} {nome}: solucao devolvida inviavel: {motivo}"
        c = inst.solution_cost(novo)
        if abs(c - melhor_bf) > TOL:
            return f"{rotulo} {nome}: custo devolvido {c:.4f} != melhor da forca bruta {melhor_bf:.4f}"
    return None


def main():
    rng = random.Random(20261005)
    falhas, testes = [], 0
    contagem = {n: [0, 0] for n, _, _ in NEIGHBORHOODS_EXT_NOVOS}  # [com melhora, sem melhora]
    for config in range(160):
        n = rng.randint(8, 18)
        arred = config % 2 == 0
        k_exato = config % 4 in (0, 1)
        inst = instancia_aleatoria(rng, n, arred, k_exato)
        base = solucao_aleatoria(inst, rng)
        if k_exato:
            inst.n_vehicles_ref = len(base)
        for etapa, routes in (("aleatoria", base),
                              ("otimo_local_5op", vnd(inst, base, NEIGHBORHOODS))):
            if not inst.is_feasible(routes)[0]:
                continue
            for nome, best_fn, _ in NEIGHBORHOODS_EXT_NOVOS:
                msg = conferir(inst, nome, best_fn, routes, f"[cfg{config} {etapa} n={n} K_exato={k_exato}]")
                testes += 1
                if msg:
                    falhas.append(msg)
                else:
                    melhorou, _ = best_fn(inst, routes)
                    contagem[nome][0 if melhorou else 1] += 1
    print(f"Testes executados: {testes}")
    print("Cobertura por operador [com melhora, sem melhora]:")
    for nome, (a, b) in contagem.items():
        print(f"  {nome:12s} {a:4d} / {b:4d}")
    if falhas:
        print(f"\nFALHAS: {len(falhas)}")
        for f in falhas[:15]:
            print(" ", f)
        sys.exit(1)
    print("\nTODOS OS TESTES PASSARAM (operadores novos == forca bruta independente).")


if __name__ == "__main__":
    main()
