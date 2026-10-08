#!/usr/bin/env python3
"""
validate_construction.py
=========================
Valida clarke_wright_savings() com uma instância sintética pequena, cujo
resultado é conferível a mão (não depende de nenhum artigo externo --
apenas geometria simples).

Instância: depósito em (0,0); clientes 2=(10,0), 3=(20,0) alinhados no
eixo x, e 4=(0,10), 5=(0,20) alinhados no eixo y. Demanda 1 cada,
capacidade 2 (força rotas de no máximo 2 clientes).

Economias (à mão, distância euclidiana exata):
  S_23 = 10+20-10 = 20
  S_45 = 10+20-10 = 20
  S_35 = 20+20-28.284... = 11.716
  S_25 = S_34 = 10+20-22.36... = 7.639 (approx)
  S_24 = 10+10-14.142... = 5.858

Com capacidade 2, só (2,3) e (4,5) cabem (qualquer outro merge exigiria
uma rota de 3+ clientes = demanda 3 > capacidade 2). Resultado esperado:
rotas {2,3} e {4,5}, custo total = (10+10+20) + (10+10+20) = 80.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import CVRPInstance  # noqa: E402
from construction import clarke_wright_savings  # noqa: E402


def make_synthetic_instance():
    coords = {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (20.0, 0.0), 4: (0.0, 10.0), 5: (0.0, 20.0)}
    demands = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}
    return CVRPInstance(
        name="synthetic-XY-cross",
        dimension=5,
        capacity=2,
        coords=coords,
        demands=demands,
        depot_id=1,
        round_distances=False,
        optimum=80.0,
        reference_routes=[],
    )


def main():
    inst = make_synthetic_instance()
    routes = clarke_wright_savings(inst)

    print("Rotas geradas:", routes)
    route_sets = [frozenset(r) for r in routes]
    esperado = [frozenset({2, 3}), frozenset({4, 5})]

    erros = []
    if len(routes) != 2:
        erros.append(f"esperado 2 rotas, obtido {len(routes)}")
    if set(route_sets) != set(esperado):
        erros.append(f"conjuntos de clientes por rota inesperados: {route_sets} != {esperado}")

    viavel, motivo = inst.is_feasible(routes)
    if not viavel:
        erros.append(f"solução inviável: {motivo}")

    custo = inst.solution_cost(routes)
    print(f"Custo total: {custo} (esperado 80.0)")
    if abs(custo - 80.0) > 1e-6:
        erros.append(f"custo total {custo} != 80.0 esperado")

    if erros:
        print("\nRESULTADO: FALHOU")
        for e in erros:
            print(" -", e)
        sys.exit(1)
    else:
        print("\nRESULTADO: OK -- Clarke-Wright Savings bate com a solução conferida à mão.")
        sys.exit(0)


if __name__ == "__main__":
    main()
