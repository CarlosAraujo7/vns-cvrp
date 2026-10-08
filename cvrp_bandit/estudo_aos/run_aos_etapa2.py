#!/usr/bin/env python3
"""
run_aos_etapa2.py
=================
Runner da Etapa 2 da Direcao A (2A decomposicao do loop + 2B AOS dentro do
VND). PROTOCOLO PRE-REGISTRADO em diario_pesquisa.md, Entrada 71 -- leia la
antes de alterar qualquer coisa aqui.

Modos:
  --pilot   controles de regressao EXATA contra a Etapa 1 + contagem de
            operacoes + viabilidade (NAO entra em inferencia):
              ctrl_gvns     = FO/best/LO   deve reproduzir `gvns_classico`
              ctrl_uniforme = U/shaken/F40 deve reproduzir `uniforme`
  --full    8 bracos x 74 instancias x 10 seeds x B in {800, 4000}.

Resumivel por chave (braco, B, instancia, seed). Roda no WSL (venv de
direcao1_local, que tem `river`):
    python3 -u run_aos_etapa2.py --pilot
    python3 -u run_aos_etapa2.py --full --workers 11
"""
import argparse
import csv
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_aos_study import _instancia, _contar, todas_instancias  # noqa: E402
from neighborhoods import NEIGHBORHOODS  # noqa: E402
from aos_vnd import vnd_generico, construir_politica  # noqa: E402

BUDGETS = (800, 4000)
SEED_BASE = 1000
# nome do braco -> (politica, aceitacao, termino)
ARMS = {
    "s2a_FO_shaken_LO": ("FO", "shaken", "LO"),
    "s2a_U_best_LO": ("U", "best", "LO"),
    "s2a_U_best_F40": ("U", "best", "F40"),
    "s2a_U_shaken_LO": ("U", "shaken", "LO"),
    "s2b_ucb1": ("ucb1", "best", "LO"),
    "s2b_eps_greedy": ("eps", "best", "LO"),
    "s2b_prob_matching": ("pm", "best", "LO"),
    "s2b_hgucb_norm": ("hgucb", "best", "LO"),
}
CONTROLES = {
    "ctrl_gvns": ("FO", "best", "LO"),
    "ctrl_uniforme": ("U", "shaken", "F40"),
}
TODOS = {**ARMS, **CONTROLES}

CAMPOS = ["braco", "B", "conjunto", "instancia", "n_customers", "seed", "custo", "otimo",
          "gap_pct", "viavel", "n_rotas", "n_ops_contadas", "n_ops_loop", "shakes",
          "uso_operadores", "tempo_s"]


def executar(braco, B, set_name, nome_inst, seed):
    inst, cws = _instancia(set_name, nome_inst)
    rng = random.Random(seed)
    contador, uso = [0], [0] * len(NEIGHBORHOODS)
    nbhs = _contar(contador, uso)
    pol_nome, aceit, term = TODOS[braco]
    politica, modo = construir_politica(pol_nome, len(nbhs), rng)
    t0 = time.perf_counter()
    rotas, info = vnd_generico(inst, [list(r) for r in cws], rng, B, nbhs, politica,
                               aceit, term, modo_recompensa=modo)
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(rotas)
    viavel, _ = inst.is_feasible(rotas)
    return {
        "braco": braco, "B": B, "conjunto": set_name, "instancia": inst.name,
        "n_customers": inst.n_customers, "seed": seed, "custo": custo, "otimo": inst.optimum,
        "gap_pct": 100 * (custo - inst.optimum) / inst.optimum, "viavel": viavel,
        "n_rotas": len([r for r in rotas if r]), "n_ops_contadas": contador[0],
        "n_ops_loop": info["usado"], "shakes": info["shakes"],
        "uso_operadores": "|".join(map(str, uso)), "tempo_s": dt,
    }


def _worker(args):
    return executar(*args)


def rodar(tarefas, out_csv, workers):
    feitos = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            feitos = {(r["braco"], int(r["B"]), r["instancia"], int(r["seed"]))
                      for r in csv.DictReader(f)}
    pend = [t for t in tarefas if (t[0], t[1], t[3], t[4]) not in feitos]
    print(f"{len(tarefas)} tarefas, {len(feitos)} ja feitas, {len(pend)} pendentes, "
          f"{workers} workers", flush=True)
    novo = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f_out, fieldnames=CAMPOS)
    if novo:
        w.writeheader()
    t0 = time.perf_counter()
    feitas = 0
    with mp.Pool(workers) as pool:
        for linha in pool.imap_unordered(_worker, pend, chunksize=1):
            w.writerow(linha)
            feitas += 1
            if feitas % 50 == 0:
                f_out.flush()
                print(f"  {feitas}/{len(pend)}  {(time.perf_counter()-t0)/60:.1f} min "
                      f"(ETA enganoso: tarefas de custo desigual)", flush=True)
    f_out.close()
    print(f"Concluido em {(time.perf_counter()-t0)/60:.1f} min -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    args = ap.parse_args()
    insts = todas_instancias()
    if args.pilot:
        escolhidas = {"A-n32-k5", "B-n41-k6", "X-n101-k25", "CMT1"}
        tarefas = [(a, 800, s, n, SEED_BASE + sd) for (s, n, _) in insts if n in escolhidas
                   for a in TODOS for sd in range(2)]
        rodar(tarefas, ROOT / "results" / "aos_study" / "etapa2_pilot.csv", args.workers)
    else:
        tarefas = [(a, B, s, n, SEED_BASE + sd)
                   for B in sorted(BUDGETS)
                   for (s, n, nc) in sorted(insts, key=lambda t: -t[2])
                   for a in ARMS for sd in range(10)]
        rodar(tarefas, ROOT / "results" / "aos_study" / "etapa2_full.csv", args.workers)


if __name__ == "__main__":
    main()
