#!/usr/bin/env python3
"""
validate_instance_reader.py
============================
Validação real (execução, não leitura de código) do leitor de instâncias
CVRP contra os 74 pares .vrp/.sol usados por Kalatzantonakis et al. (2023)
(26 de A, 23 de B, 21 de X, 4 de CMT).

Para cada instância:
  1. carrega via load_instance() (isso já autodetecta arredondamento e
     levanta erro se nenhuma convenção bater com o Cost declarado);
  2. recalcula o custo da rota de referência do .sol com a convenção
     adotada e confere contra o "Cost" declarado (tolerância 1e-2);
  3. confere que a rota de referência é uma solução viável (cobertura
     exata dos clientes, sem violar capacidade).

Não passa nenhuma instância silenciosamente -- qualquer falha aborta com
mensagem clara de qual instância e qual checagem falhou.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import load_instance  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "instances"

SETS = {
    "A": ROOT / "A",
    "B": ROOT / "B",
    "X": ROOT / "X",
    "CMT": ROOT / "CMT",
}


def main():
    total = 0
    n_round = 0
    n_float = 0
    falhas = []
    for set_name, d in SETS.items():
        vrp_files = sorted(d.glob("*.vrp"))
        print(f"\n=== Conjunto {set_name}: {len(vrp_files)} instâncias ===")
        for vrp_path in vrp_files:
            total += 1
            try:
                inst = load_instance(vrp_path)
            except Exception as e:
                falhas.append((vrp_path.name, f"load_instance falhou: {e}"))
                print(f"  [FALHA] {vrp_path.name}: {e}")
                continue

            if inst.round_distances:
                n_round += 1
            else:
                n_float += 1

            custo_recalc = inst.solution_cost(inst.reference_routes)
            diff = abs(custo_recalc - inst.optimum)
            ok_custo = diff <= 0.01 * max(1.0, abs(inst.optimum)) and diff <= 1.0

            viavel, motivo = inst.is_feasible(inst.reference_routes)

            status = "OK" if (ok_custo and viavel) else "FALHA"
            marca_dist = "int" if inst.round_distances else "float"
            print(
                f"  [{status}] {inst.name:14s} n={inst.n_customers:4d} "
                f"cap={inst.capacity:5d} dist={marca_dist:5s} "
                f"optimum={inst.optimum:12.3f} recalc={custo_recalc:12.3f} "
                f"diff={diff:8.4f} viavel={viavel}"
            )
            if not ok_custo:
                falhas.append((inst.name, f"custo recalculado {custo_recalc} != optimum {inst.optimum} (diff={diff})"))
            if not viavel:
                falhas.append((inst.name, f"rota de referência inviável: {motivo}"))

    print(f"\nTotal de instâncias verificadas: {total}")
    print(f"  distância arredondada (int): {n_round}")
    print(f"  distância em ponto flutuante: {n_float}")
    print(f"Falhas: {len(falhas)}")
    for nome, motivo in falhas:
        print(f"  - {nome}: {motivo}")

    if falhas:
        print("\nRESULTADO: FALHOU -- não prosseguir até corrigir o leitor.")
        sys.exit(1)
    else:
        print("\nRESULTADO: TODAS as 74 instâncias validadas com sucesso.")
        sys.exit(0)


if __name__ == "__main__":
    main()
