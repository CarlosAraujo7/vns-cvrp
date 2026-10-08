#!/usr/bin/env python3
"""
validate_repair.py
====================
Valida repair_route_count(): depois de CWS + reparo, o número de rotas
não-vazias deve bater EXATAMENTE com instance.n_vehicles_ref para todas as
26+23+4=53 instâncias A/B/CMT (vehicles_exact=True), e ser >= para as 21
de X (vehicles_exact=False). Roda nas 74 instâncias reais.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "instances"
SETS = ["A", "B", "CMT", "X"]


def main():
    falhas = []
    total = 0
    for set_name in SETS:
        d = ROOT / set_name
        for vrp_path in sorted(d.glob("*.vrp")):
            total += 1
            # vehicles_exact=None -> autodetectado por load_instance() a
            # partir do NOME da instancia (ver instance.py,
            # _vehicles_exact_default -- corrigido em 29/08/2026, diario
            # Entrada 3: CMT tambem e' limite inferior, nao exato).
            inst = load_instance(vrp_path)
            vehicles_exact = inst.vehicles_exact
            cws = clarke_wright_savings(inst)
            n_cws = len(cws)
            reparado = repair_route_count(inst, cws)
            n_final = len([r for r in reparado if len(r) > 0])

            viavel, motivo = inst.is_feasible(reparado)
            if vehicles_exact:
                ok_contagem = (n_final == inst.n_vehicles_ref)
            else:
                ok_contagem = (n_final >= inst.n_vehicles_ref)

            status = "OK" if (viavel and ok_contagem) else "FALHA"
            marca = "" if n_cws == n_final else f" (CWS deu {n_cws}, reparado p/ {n_final})"
            print(f"[{status}] {inst.name:14s} alvo={inst.n_vehicles_ref:3d} "
                  f"({'exato' if vehicles_exact else '>='}) final={n_final:3d}{marca} "
                  f"viavel={viavel}")
            if not (viavel and ok_contagem):
                falhas.append((inst.name, motivo, n_final, inst.n_vehicles_ref))

    print(f"\nTotal: {total} instâncias. Falhas: {len(falhas)}")
    for f in falhas:
        print("  -", f)
    if falhas:
        sys.exit(1)
    print("RESULTADO: OK -- repair_route_count() acerta o numero de veiculos em todas as 74 instâncias.")


if __name__ == "__main__":
    main()
