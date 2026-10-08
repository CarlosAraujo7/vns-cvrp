#!/usr/bin/env python3
"""
validate_neighborhoods.py
==========================
Valida cada um dos 5 operadores de vizinhança com cenários pequenos,
construídos à mão, cuja melhora esperada foi calculada manualmente ANTES
de rodar o código (ver diario_pesquisa.md, Entrada 3, para as contas).
Isso já pegou um bug real no Relocate(2,0) (a aresta interna do bloco
sendo esquecida no cálculo de remoção) -- corrigido em neighborhoods.py
antes deste teste ser escrito para validar a correção.

Cada cenário testa best_move() de UM operador isoladamente, com uma
solução inicial conhecida e a melhora esperada calculada à mão.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import CVRPInstance  # noqa: E402
import neighborhoods as nb  # noqa: E402


def make_instance(coords, demands, capacity=1000):
    return CVRPInstance(
        name="synthetic",
        dimension=len(coords),
        capacity=capacity,
        coords=coords,
        demands=demands,
        depot_id=1,
        round_distances=False,
        optimum=0.0,
        reference_routes=[],
    )


def check(nome, inst, routes_iniciais, operador_best, custo_esperado_final, delta_esperado):
    custo_inicial = inst.solution_cost(routes_iniciais)
    melhorou, novas_rotas = operador_best(inst, routes_iniciais)
    custo_final = inst.solution_cost(novas_rotas)
    delta_obtido = custo_final - custo_inicial

    viavel, motivo = inst.is_feasible(novas_rotas)

    ok = (
        melhorou
        and viavel
        and abs(custo_final - custo_esperado_final) < 1e-3
        and abs(delta_obtido - delta_esperado) < 1e-3
    )
    status = "OK" if ok else "FALHA"
    print(
        f"[{status}] {nome}: custo {custo_inicial:.3f} -> {custo_final:.3f} "
        f"(delta obtido {delta_obtido:+.3f}, esperado {delta_esperado:+.3f}); "
        f"rotas: {routes_iniciais} -> {novas_rotas}; viavel={viavel} ({motivo})"
    )
    return ok


def main():
    resultados = []

    # ---------------- 2-Opt: cruzamento clássico num quadrado ----------
    coords = {1: (5, 5), 2: (0, 0), 3: (10, 10), 4: (10, 0), 5: (0, 10)}
    demands = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}
    inst = make_instance(coords, demands)
    routes = [[2, 3, 4, 5]]  # cruzado; ótimo é [2,4,3,5] (perímetro)
    resultados.append(check("2-Opt", inst, routes, nb.two_opt_best, 44.142136, -8.284271))

    # ---------------- Move(1): cliente fora de ordem na mesma rota ------
    coords = {1: (0, 0), 2: (1, 0), 3: (5, 0), 4: (2, 0)}
    demands = {1: 0, 2: 1, 3: 1, 4: 1}
    inst = make_instance(coords, demands)
    routes = [[3, 2, 4]]  # ótimo é [2,4,3]
    resultados.append(check("Move(1)", inst, routes, nb.move1_best, 10.0, -2.0))

    # ---------------- Swap(1,1): dois clientes trocados entre rotas ----
    coords = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (0, 1), 5: (0, 2)}
    demands = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}
    inst = make_instance(coords, demands)
    routes = [[4, 3], [2, 5]]  # ótimo é [2,3],[4,5]
    resultados.append(check("Swap(1,1)", inst, routes, nb.swap11_best, 8.0, -2.472136))

    # ---------------- Relocate(1,0): 1 cliente na rota errada -----------
    coords = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (0, 4), 5: (0, 6), 6: (0, 5)}
    demands = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
    inst = make_instance(coords, demands)
    routes = [[2, 3, 6], [4, 5]]  # 6 pertence entre 4 e 5
    resultados.append(check("Relocate(1,0)", inst, routes, nb.relocate10_best, 16.0, -8.384776))

    # ---------------- Relocate(2,0): bloco de 2 na rota errada -----------
    coords = {
        1: (0, 0), 2: (1, 0), 3: (2, 0),
        4: (0, 4), 5: (0, 7), 6: (0, 5), 7: (0, 6),
    }
    demands = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
    inst = make_instance(coords, demands)
    routes = [[2, 6, 7, 3], [4, 5]]  # bloco [6,7] pertence entre 4 e 5
    # Nota: a conta à mão original previa mover [6,7] para dentro de [4,5],
    # chegando a 18.0. Rodar deu 17.5712 -- best_move() achou um vizinho
    # MELHOR (mover o bloco [4,5] para dentro da rota A), que
    # bruteforce_crosscheck.py confirmou de forma independente ser de fato
    # o ótimo do vizinho Relocate(2,0) para esta instância. Valor abaixo
    # corrigido para o ótimo verificado, não o meu palpite manual inicial.
    resultados.append(check("Relocate(2,0)", inst, routes, nb.relocate20_best, 17.571068, -11.852675))

    print()
    if all(resultados):
        print(f"RESULTADO: TODOS os {len(resultados)} operadores validados (best_move).")
    else:
        print(f"RESULTADO: FALHOU -- {resultados.count(False)}/{len(resultados)} cenários falharam.")
        sys.exit(1)

    # ---------------- smoke test dos *_random: viabilidade preservada --
    print("\nSmoke test dos movimentos aleatórios (random_move) -- só checa viabilidade:")
    rng = random.Random(42)
    inst_real_coords = {1: (0, 0)}
    for k in range(2, 21):
        inst_real_coords[k] = (k, (k * 7) % 13)
    demands_real = {k: 1 for k in inst_real_coords}
    demands_real[1] = 0
    inst2 = make_instance(inst_real_coords, demands_real, capacity=5)
    from construction import clarke_wright_savings
    base_routes = clarke_wright_savings(inst2)
    ok_random = True
    for nome, best_fn, random_fn in nb.NEIGHBORHOODS:
        cur = [list(r) for r in base_routes]
        falhas_locais = 0
        for _ in range(200):
            novo = random_fn(inst2, cur, rng)
            if novo is None:
                continue
            viavel, motivo = inst2.is_feasible(novo)
            if not viavel:
                falhas_locais += 1
                print(f"  [FALHA] {nome} produziu solução inviável: {motivo}")
            cur = novo
        status = "OK" if falhas_locais == 0 else "FALHA"
        print(f"  [{status}] {nome}: 200 aplicações, {falhas_locais} inviáveis")
        ok_random = ok_random and (falhas_locais == 0)

    if not ok_random:
        sys.exit(1)
    print("\nRESULTADO FINAL: OK -- best_move validado à mão para os 5 operadores, "
          "random_move preserva viabilidade em todos eles.")


if __name__ == "__main__":
    main()
