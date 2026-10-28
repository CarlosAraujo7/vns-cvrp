#!/usr/bin/env python3
"""
validate_tsp_neighborhoods.py
=============================
Valida os operadores vetorizados de src/tsp_neighborhoods.py contra uma ENUMERACAO INDEPENDENTE por forca bruta:
para cada movimento da vizinhanca (na mesma ordem de varredura), o novo tour e CONSTRUIDO explicitamente e seu custo
recalculado do zero (delta = custo(novo) - custo(antigo)); a contagem de avaliacoes de distancia e a de uma implementacao
em lacos que consulta d(.,.) uma vez por aresta nova (as existentes sao lidas sem contar). Confere, para cada operador,
instancia e tour: (a) o melhor delta, (b) o tour resultante (inclui o desempate: primeiro na ordem de varredura),
(c) o numero de chamadas somado a `inst.chamadas`, (d) que o resultado e uma permutacao valida e que o custo do tour
novo e o antigo mais o delta. Tambem confere os movimentos aleatorios (permutacao valida).

Uso: python tests/validate_tsp_neighborhoods.py
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tsp_inst import TSPInst  # noqa: E402
import tsp_neighborhoods as nb  # noqa: E402


def custo(D, t):
    return float(sum(D[t[i]][t[i + 1]] for i in range(len(t) - 1)))


def ref_best(D, t, movimentos):
    """movimentos: gerador de (novo_t, n_avaliacoes) na ordem de varredura. Devolve (melhor_delta, melhor_t, total_calls)."""
    c0 = custo(D, t)
    melhor, bt, calls = float("inf"), None, 0
    for novo, nc in movimentos:
        calls += nc
        d = custo(D, novo) - c0
        if d < melhor:
            melhor, bt = d, novo
    return melhor, bt, calls


def mov_2opt(t, N):
    for i in range(0, N):
        for j in range(i + 2, N):
            yield t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:], 2


def mov_or(t, N, Ls):
    for L in Ls:
        for s in range(1, N - L + 1):
            seg0 = t[s:s + L]
            rest = t[:s] + t[s + L:]
            primeiro = True
            for k in range(0, N):
                if s - 1 <= k <= s + L - 1:
                    continue
                pos = k if k < s else k - L
                for o in ((0,) if L == 1 else (0, 1)):
                    seg = seg0 if o == 0 else seg0[::-1]
                    yield rest[:pos + 1] + seg + rest[pos + 1:], 2 + (1 if (primeiro and o == 0) else 0) * 0
                primeiro = False
        # a contagem de fechamento (1 por segmento) entra a parte, em contar_or


def contar_or(N, Ls):
    tot = 0
    for L in Ls:
        ns = N - L
        if ns < 1:
            continue
        tot += ns + ns * (N - L - 1) * (2 if L == 1 else 4)
    return tot


def mov_swap(t, N):
    for p in range(1, N):
        for q in range(p + 1, N):
            n = list(t)
            n[p], n[q] = n[q], n[p]
            yield n, (2 if q == p + 1 else 4)


def mov_cross(t, N):
    for la in (1, 2, 3):
        for lb in (1, 2, 3):
            if la == 1 and lb == 1:
                continue
            for p in range(1, N - la - lb + 1):
                for q in range(p + la, N - lb + 1):
                    S1, S2 = t[p:p + la], t[q:q + lb]
                    yield t[:p] + S2 + t[p + la:q] + S1 + t[q + lb:], (3 if q == p + la else 4)


def mov_3a(t, N):
    for i in range(0, N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                yield t[:i + 1] + t[j + 1:k + 1] + t[i + 1:j + 1] + t[k + 1:], 3


def mov_3b(t, N):
    for i in range(0, N):
        for j in range(i + 2, N):
            for k in range(j + 2, N):
                yield t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:k + 1][::-1] + t[k + 1:], 3


def testar(N, semente, ncasos=8):
    rng = random.Random(semente)
    falhas, verificados = 0, 0
    coords = [(rng.randint(0, 40), rng.randint(0, 40)) for _ in range(N)]   # coordenadas pequenas: muitos empates
    inst = TSPInst(f"teste{N}_{semente}", coords)
    D = inst._D
    operadores = [
        ("2-opt", nb.P9[0][1], lambda t: ref_best(D, t, mov_2opt(t, N)), None),
        ("Or-opt(1)", nb.P9[1][1], lambda t: ref_best(D, t, mov_or(t, N, (1,))), (1,)),
        ("Swap(1,1)", nb.P9[2][1], lambda t: ref_best(D, t, mov_swap(t, N)), None),
        ("Or-opt(2)", nb.P9[3][1], lambda t: ref_best(D, t, mov_or(t, N, (2,))), (2,)),
        ("Or-opt(3)", nb.P9[4][1], lambda t: ref_best(D, t, mov_or(t, N, (3,))), (3,)),
        ("Cross(<=3)", nb.P9[5][1], lambda t: ref_best(D, t, mov_cross(t, N)), None),
        ("Or-opt(4-10)", nb.P9[6][1], lambda t: ref_best(D, t, mov_or(t, N, tuple(range(4, 11)))), tuple(range(4, 11))),
        ("3-opt A", nb.P9[7][1], lambda t: ref_best(D, t, mov_3a(t, N)), None),
        ("3-opt B", nb.P9[8][1], lambda t: ref_best(D, t, mov_3b(t, N)), None),
    ]
    for caso in range(ncasos):
        tour = list(range(1, N))
        rng.shuffle(tour)
        t = [0] + tour + [0]
        c_ant = custo(D, t)
        for nome, fn, ref, Ls in operadores:
            d_ref, t_ref, calls_ref = ref(t)
            if Ls is not None:
                calls_ref += contar_or(N, Ls) - 0      # ref_best contou 2/4 por (k,o); aqui so soma o fechamento
                # correcao: mov_or ja contou 2 por (s,k,o); a contagem correta e contar_or(N, Ls) por inteiro
                calls_ref = contar_or(N, Ls)
            inst.chamadas = 0
            melhorou, novo = fn(inst, [tour])
            chamadas = inst.chamadas
            verificados += 1
            if t_ref is None:                                          # nenhum movimento existe
                ok = (not melhorou) and chamadas == calls_ref
            elif d_ref < -1e-9:
                ok = (melhorou and [0] + novo[0] + [0] == t_ref and chamadas == calls_ref
                      and abs(custo(D, [0] + novo[0] + [0]) - (c_ant + d_ref)) < 1e-9
                      and sorted(novo[0]) == list(range(1, N)))
            else:
                ok = (not melhorou) and chamadas == calls_ref
            if not ok:
                falhas += 1
                print(f"FALHA N={N} semente={semente} caso={caso} {nome}: d_ref={d_ref} melhorou={melhorou} "
                      f"chamadas={chamadas} ref={calls_ref}")
    # movimentos aleatorios
    for _ in range(200):
        tour = list(range(1, N))
        rng.shuffle(tour)
        for nome, _fb, rf in nb.P5:
            novo = rf(inst, [tour], rng)
            if novo is not None:
                verificados += 1
                if sorted(novo[0]) != list(range(1, N)):
                    falhas += 1
                    print(f"FALHA movimento aleatorio {nome}: nao e permutacao")
    return falhas, verificados


def main():
    tot_f, tot_v = 0, 0
    for N, semente in [(6, 1), (7, 2), (8, 3), (9, 4), (10, 5), (12, 6), (14, 7), (16, 8), (20, 9), (26, 10), (31, 11)]:
        f, v = testar(N, semente)
        tot_f += f
        tot_v += v
        print(f"N={N:3d}: {v} verificacoes, {f} falhas")
    print(f"TOTAL: {tot_v} verificacoes, {tot_f} falhas")
    sys.exit(1 if tot_f else 0)


if __name__ == "__main__":
    main()
