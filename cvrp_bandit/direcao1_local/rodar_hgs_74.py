#!/usr/bin/env python3
"""
rodar_hgs_74.py
================
Roda o HGS-CVRP (`src/solver_hgs.py`) nas 74 instâncias, mesmo esquema de
CSV de `rodar_lkh3_74.py` / `rodar_ortools_74.py`, 5 seeds (1000..1004).

Dois regimes de orçamento (diário, Entrada 61):
  matched  -- tempo médio do HG-UCB por instância (0,2-10 s): reproduz o
              protocolo da Direção 1 e permite comparar HGS contra HG-UCB,
              LKH3 e OR-Tools sob o MESMO (minúsculo) orçamento.
  n01      -- 0,1 s por cliente (A/B: 3-8 s; X: 10-19 s): orçamento
              realista para calibrar o teto de qualidade. O padrão da
              literatura para o HGS é 2,4 s/cliente; 0,1 é ~4% disso,
              escolhido só para caber em ~1 h no total.

Roda dentro do WSL (binário Linux). Uso:
    python rodar_hgs_74.py --regime matched
    python rodar_hgs_74.py --regime n01
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from instance import load_all  # noqa: E402
from solver_hgs import solve_hgs, HGS_BIN  # noqa: E402

N_SEEDS = 5
SEED_BASE = 1000
CONJUNTOS = ("A", "B", "X", "CMT")
TEMPOS_CSV = ROOT / "results" / "tempos_hgucb_74.csv"
CAMPOS = ["conjunto", "instancia", "n_customers", "seed", "time_limit_s", "tempo_gasto_s",
          "n_rotas", "n_vehicles_ref", "custo", "otimo", "gap_pct", "viavel", "motivo"]


def orcamentos_matched():
    soma, conta = {}, {}
    with open(TEMPOS_CSV, encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            n = linha["instancia"]
            soma[n] = soma.get(n, 0.0) + float(linha["tempo_s"])
            conta[n] = conta.get(n, 0) + 1
    return {n: soma[n] / conta[n] for n in soma}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["matched", "n01"], required=True)
    args = ap.parse_args()
    if not HGS_BIN.exists():
        sys.exit(f"ERRO: binário do HGS não encontrado em {HGS_BIN}")
    matched = orcamentos_matched() if args.regime == "matched" else None
    out_csv = ROOT / "results" / f"hgs_74_{args.regime}.csv"
    ja_existe = out_csv.exists()
    f_out = open(out_csv, "a" if ja_existe else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=CAMPOS)
    if not ja_existe:
        writer.writeheader()
    feitos = set()
    if ja_existe:  # retomada: pula (instancia, seed) já gravados
        with open(out_csv, encoding="utf-8") as f:
            feitos = {(r["instancia"], int(r["seed"])) for r in csv.DictReader(f)}

    t_ini = time.perf_counter()
    for set_name in CONJUNTOS:
        for inst in load_all(ROOT / "instances" / set_name):
            t_lim = matched[inst.name] if matched else 0.1 * inst.n_customers
            gaps = []
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                if (inst.name, seed) in feitos:
                    continue
                try:
                    routes, wall = solve_hgs(inst, t_lim, seed=seed)
                except RuntimeError as e:
                    print(f"    AVISO: {inst.name} seed={seed}: {e}")
                    writer.writerow({"conjunto": set_name, "instancia": inst.name,
                                     "n_customers": inst.n_customers, "seed": seed,
                                     "time_limit_s": t_lim, "tempo_gasto_s": "", "n_rotas": 0,
                                     "n_vehicles_ref": inst.n_vehicles_ref, "custo": "",
                                     "otimo": inst.optimum, "gap_pct": "", "viavel": False,
                                     "motivo": "HGS sem solucao"})
                    continue
                custo = inst.solution_cost(routes)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                viavel, motivo = inst.is_feasible(routes)
                if viavel:
                    gaps.append(gap)
                writer.writerow({"conjunto": set_name, "instancia": inst.name,
                                 "n_customers": inst.n_customers, "seed": seed,
                                 "time_limit_s": t_lim, "tempo_gasto_s": wall,
                                 "n_rotas": len([r for r in routes if r]),
                                 "n_vehicles_ref": inst.n_vehicles_ref, "custo": custo,
                                 "otimo": inst.optimum, "gap_pct": gap,
                                 "viavel": viavel, "motivo": motivo})
            f_out.flush()
            m = sum(gaps) / len(gaps) if gaps else float("nan")
            print(f"  {inst.name:16s} n={inst.n_customers:4d} t={t_lim:7.2f}s "
                  f"gap_medio={m:6.2f}% ({len(gaps)}/{N_SEEDS} viaveis)", flush=True)
    f_out.close()
    print(f"\nTotal: {(time.perf_counter()-t_ini)/60:.1f} min -> {out_csv}")


if __name__ == "__main__":
    main()
