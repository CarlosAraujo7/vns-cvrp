#!/usr/bin/env python3
"""
bruteforce_crosscheck.py
=========================
Segunda camada de validação dos 5 operadores, independente das contas à
mão de validate_neighborhoods.py: para cada operador, gera TODOS os
vizinhos válidos por força bruta (sem nenhum atalho de delta incremental,
recalculando o custo total do zero via instance.solution_cost a cada
candidato) e confere que o melhor vizinho encontrado por força bruta bate
EXATAMENTE com o que best_move() (implementação otimizada por delta)
retorna. Isso pega qualquer bug de delta que uma conta à mão pontual possa
não cobrir (foi assim que o caso ambíguo do Relocate(2,0) em
validate_neighborhoods.py foi esclarecido: a força bruta confirmou que o
resultado do best_move() -- não a minha expectativa manual -- estava
certo).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import CVRPInstance  # noqa: E402
import neighborhoods as nb  # noqa: E402


def make_instance(coords, demands, capacity):
    return CVRPInstance(
        name="synthetic", dimension=len(coords), capacity=capacity,
        coords=coords, demands=demands, depot_id=1,
        round_distances=False, optimum=0.0, reference_routes=[],
    )


def _feasible_or_none(instance, routes):
    routes = [r for r in routes if len(r) > 0]
    ok, _ = instance.is_feasible(routes)
    return routes if ok else None


def bruteforce_two_opt(instance, routes):
    best = instance.solution_cost(routes)
    best_routes = routes
    for ridx, route in enumerate(routes):
        n = len(route)
        for i in range(n):
            for j in range(i + 1, n):
                cand = [list(r) for r in routes]
                cand[ridx] = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                c = instance.solution_cost(cand)
                if c < best - 1e-9:
                    best, best_routes = c, cand
    return best, best_routes


def bruteforce_move1(instance, routes):
    best = instance.solution_cost(routes)
    best_routes = routes
    for ridx, route in enumerate(routes):
        n = len(route)
        for p in range(n):
            rest = route[:p] + route[p + 1:]
            c_val = route[p]
            for q in range(len(rest) + 1):
                cand_route = rest[:q] + [c_val] + rest[q:]
                cand = [list(r) for r in routes]
                cand[ridx] = cand_route
                c = instance.solution_cost(cand)
                if c < best - 1e-9:
                    best, best_routes = c, cand
    return best, best_routes


def bruteforce_swap11(instance, routes):
    best = instance.solution_cost(routes)
    best_routes = routes
    n_routes = len(routes)
    for ra in range(n_routes):
        for pa in range(len(routes[ra])):
            for rb in range(ra + 1, n_routes):
                for pb in range(len(routes[rb])):
                    cand = [list(r) for r in routes]
                    cand[ra][pa], cand[rb][pb] = cand[rb][pb], cand[ra][pa]
                    feas = _feasible_or_none(instance, cand)
                    if feas is None:
                        continue
                    c = instance.solution_cost(cand)
                    if c < best - 1e-9:
                        best, best_routes = c, cand
    return best, best_routes


def bruteforce_relocate10(instance, routes):
    best = instance.solution_cost(routes)
    best_routes = routes
    n_routes = len(routes)
    for ra in range(n_routes):
        for pa in range(len(routes[ra])):
            c_val = routes[ra][pa]
            for rb in range(n_routes):
                if rb == ra:
                    continue
                for q in range(len(routes[rb]) + 1):
                    cand = [list(r) for r in routes]
                    cand[ra].pop(pa)
                    cand[rb].insert(q, c_val)
                    feas = _feasible_or_none(instance, cand)
                    if feas is None:
                        continue
                    c = instance.solution_cost(feas)
                    if c < best - 1e-9:
                        best, best_routes = c, feas
    return best, best_routes


def bruteforce_relocate20(instance, routes):
    best = instance.solution_cost(routes)
    best_routes = routes
    n_routes = len(routes)
    for ra in range(n_routes):
        if len(routes[ra]) < 2:
            continue
        for pa in range(len(routes[ra]) - 1):
            bloco_fwd = routes[ra][pa:pa + 2]
            bloco_bwd = list(reversed(bloco_fwd))
            for rb in range(n_routes):
                if rb == ra:
                    continue
                for q in range(len(routes[rb]) + 1):
                    for bloco in (bloco_fwd, bloco_bwd):
                        cand = [list(r) for r in routes]
                        del cand[ra][pa:pa + 2]
                        cand[rb][q:q] = bloco
                        feas = _feasible_or_none(instance, cand)
                        if feas is None:
                            continue
                        c = instance.solution_cost(feas)
                        if c < best - 1e-9:
                            best, best_routes = c, feas
    return best, best_routes


CASES = [
    ("2-Opt", nb.two_opt_best, bruteforce_two_opt,
     {1: (5, 5), 2: (0, 0), 3: (10, 10), 4: (10, 0), 5: (0, 10)},
     {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}, 1000, [[2, 3, 4, 5]]),
    ("Move(1)", nb.move1_best, bruteforce_move1,
     {1: (0, 0), 2: (1, 0), 3: (5, 0), 4: (2, 0)},
     {1: 0, 2: 1, 3: 1, 4: 1}, 1000, [[3, 2, 4]]),
    ("Swap(1,1)", nb.swap11_best, bruteforce_swap11,
     {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (0, 1), 5: (0, 2)},
     {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}, 1000, [[4, 3], [2, 5]]),
    ("Relocate(1,0)", nb.relocate10_best, bruteforce_relocate10,
     {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (0, 4), 5: (0, 6), 6: (0, 5)},
     {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}, 1000, [[2, 3, 6], [4, 5]]),
    ("Relocate(2,0)", nb.relocate20_best, bruteforce_relocate20,
     {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (0, 4), 5: (0, 7), 6: (0, 5), 7: (0, 6)},
     {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}, 1000, [[2, 6, 7, 3], [4, 5]]),
]


def main():
    import random
    todos_ok = True

    print("=== Cenários curados (os mesmos de validate_neighborhoods.py) ===")
    for nome, best_fn, bf_fn, coords, demands, cap, routes in CASES:
        inst = make_instance(coords, demands, cap)
        custo_bf, rotas_bf = bf_fn(inst, routes)
        _, rotas_impl = best_fn(inst, routes)
        custo_impl = inst.solution_cost(rotas_impl)
        ok = abs(custo_bf - custo_impl) < 1e-6
        status = "OK" if ok else "FALHA"
        print(f"[{status}] {nome}: forca_bruta={custo_bf:.4f} implementacao={custo_impl:.4f}")
        if not ok:
            print(f"       rotas forca_bruta={rotas_bf}  rotas implementacao={rotas_impl}")
        todos_ok = todos_ok and ok

    print("\n=== Instâncias aleatórias pequenas (n<=9 clientes, 2-3 rotas) ===")
    rng = random.Random(7)
    for trial in range(15):
        n_cust = rng.randint(5, 9)
        coords = {1: (0.0, 0.0)}
        demands = {1: 0}
        for k in range(2, n_cust + 2):
            coords[k] = (rng.uniform(-10, 10), rng.uniform(-10, 10))
            demands[k] = rng.randint(1, 3)
        cap = rng.randint(4, 7)
        inst = make_instance(coords, demands, cap)
        from construction import clarke_wright_savings
        routes = clarke_wright_savings(inst)
        if len(routes) < 2:
            continue  # precisa de >=2 rotas para os operadores inter-rota
        for nome, best_fn, bf_fn, *_ in CASES:
            custo_bf, _ = bf_fn(inst, routes)
            _, rotas_impl = best_fn(inst, routes)
            custo_impl = inst.solution_cost(rotas_impl)
            ok = abs(custo_bf - custo_impl) < 1e-6
            if not ok:
                todos_ok = False
                print(f"[FALHA] trial={trial} {nome}: forca_bruta={custo_bf:.4f} "
                      f"implementacao={custo_impl:.4f} routes_iniciais={routes}")
    if todos_ok:
        print("[OK] Todas as instâncias aleatórias: implementação bate com força bruta.")

    print()
    if todos_ok:
        print("RESULTADO: OK -- best_move() de todos os operadores bate com força bruta "
              "em todos os cenários (curados + aleatórios).")
    else:
        print("RESULTADO: FALHOU -- ver divergências acima.")
        sys.exit(1)


if __name__ == "__main__":
    main()
