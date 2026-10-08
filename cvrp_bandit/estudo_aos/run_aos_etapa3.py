#!/usr/bin/env python3
"""
run_aos_etapa3.py
=================
Runner da Etapa 3 da Direcao A (AOS com conjunto estendido e heterogeneo de
operadores, fatorial termino LO x R5). PROTOCOLO PRE-REGISTRADO em
diario_pesquisa.md, Entrada 77 -- leia la antes de alterar qualquer coisa.

Orcamento em AVALIACOES DE DISTANCIA (InstanciaContadora): E1 = 600*n^2,
E2 = 5*E1. O shake usa so os 5 operadores originais.

Modos:
  --pilot   controles de regressao/contabilidade (NAO entra em inferencia):
              ctrl_apps = P5/U/LO com orcamento em APLICACOES (B=800) deve
                          reproduzir EXATAMENTE `s2a_U_best_LO` da Etapa 2
              estouro do orcamento < custo da maior aplicacao
              viabilidade; regra R5; ordem FOC
  --full    13 bracos x 74 instancias x 10 seeds x {E1, E2}.

Resumivel por chave (braco, nivel, instancia, seed). Roda no WSL:
    python3 -u run_aos_etapa3.py --pilot
    python3 -u run_aos_etapa3.py --full --workers 11
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
from run_aos_study import _instancia, todas_instancias  # noqa: E402
from neighborhoods import NEIGHBORHOODS  # noqa: E402
from neighborhoods_ext import NEIGHBORHOODS_EXT_NOVOS  # noqa: E402
from instancia_contadora import InstanciaContadora  # noqa: E402
from aos_vnd import vnd_generico, construir_politica  # noqa: E402

KAPPA = 600
NIVEIS = (1, 5)                      # E1 = KAPPA*n^2, E2 = 5*E1
SEED_BASE = 1000
POOL_P10 = NEIGHBORHOODS + NEIGHBORHOODS_EXT_NOVOS
POOL_P5 = NEIGHBORHOODS
FOC_ORDEM = [0, 7, 1, 5, 6, 4, 3, 8, 2, 9]   # fixada no pre-registro (Entrada 77)
PACIENCIA = 5

# nome -> (pool, politica, custo_sensivel, termino)
ARMS = {
    "s3_LO_FOC": ("P10", "FOC", False, "LO"),
    "s3_LO_U": ("P10", "U", False, "LO"),
    "s3_LO_ucb1c": ("P10", "ucb1", True, "LO"),
    "s3_LO_pmc": ("P10", "pm", True, "LO"),
    "s3_LO_hgucbc": ("P10", "hgucb", True, "LO"),
    "s3_R5_FOC": ("P10", "FOC", False, "R"),
    "s3_R5_U": ("P10", "U", False, "R"),
    "s3_R5_ucb1c": ("P10", "ucb1", True, "R"),
    "s3_R5_pmc": ("P10", "pm", True, "R"),
    "s3_R5_hgucbc": ("P10", "hgucb", True, "R"),
    "s3_P5_LO_U": ("P5", "U", False, "LO"),
    "s3_P5_LO_FO": ("P5", "FO", False, "LO"),
    "s3_R5_hgucb_cego": ("P10", "hgucb", False, "R"),   # exploratorio
}
CONTROLES = {"ctrl_apps": ("P5", "U", False, "LO")}   # orcamento em aplicacoes (B=800)

CAMPOS = ["braco", "nivel", "conjunto", "instancia", "n_customers", "seed", "orcamento",
          "custo", "otimo", "gap_pct", "viavel", "n_rotas", "calls_usadas", "max_calls_app",
          "apps", "shakes", "paciencia_fim", "uso_operadores", "calls_por_operador", "tempo_s"]

_PROXY = {}


def _proxy(set_name, nome):
    chave = (set_name, nome)
    if chave not in _PROXY:
        inst, _ = _instancia(set_name, nome)
        _PROXY[chave] = InstanciaContadora(inst)
    return _PROXY[chave]


def executar(braco, nivel, set_name, nome_inst, seed):
    inst, cws = _instancia(set_name, nome_inst)
    proxy = _proxy(set_name, nome_inst)
    proxy.chamadas = 0
    pool_nome, pol_nome, custo_sens, term = {**ARMS, **CONTROLES}[braco]
    pool = POOL_P10 if pool_nome == "P10" else POOL_P5
    k = len(pool)
    uso, calls_op, maximo = [0] * k, [0] * k, [0]

    def envolver(idx, bf):
        def f(_inst, routes):
            c0 = proxy.chamadas
            uso[idx] += 1
            res = bf(proxy, routes)
            d = proxy.chamadas - c0
            calls_op[idx] += d
            if d > maximo[0]:
                maximo[0] = d
            return res
        return f

    nbhs = [(nm, envolver(i, bf), rf) for i, (nm, bf, rf) in enumerate(pool)]
    rng = random.Random(seed)
    politica, modo = construir_politica(pol_nome, k, rng, custo=custo_sens,
                                        ordem=FOC_ORDEM if pol_nome == "FOC" else None)
    if braco == "ctrl_apps":
        orcamento, medida = 800, None
    else:
        orcamento = nivel * KAPPA * inst.n_customers ** 2
        medida = lambda: proxy.chamadas  # noqa: E731
    t0 = time.perf_counter()
    rotas, info = vnd_generico(inst, [list(r) for r in cws], rng, orcamento, nbhs, politica,
                               "best", term, modo_recompensa=modo,
                               neighborhoods_shake=NEIGHBORHOODS, medida=medida,
                               paciencia=PACIENCIA)
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(rotas)
    viavel, _ = inst.is_feasible(rotas)
    return {
        "braco": braco, "nivel": nivel, "conjunto": set_name, "instancia": inst.name,
        "n_customers": inst.n_customers, "seed": seed, "orcamento": orcamento, "custo": custo,
        "otimo": inst.optimum, "gap_pct": 100 * (custo - inst.optimum) / inst.optimum,
        "viavel": viavel, "n_rotas": len([r for r in rotas if r]), "calls_usadas": proxy.chamadas,
        "max_calls_app": maximo[0], "apps": info["usado"], "shakes": info["shakes"],
        "paciencia_fim": info["paciencia_fim"], "uso_operadores": "|".join(map(str, uso)),
        "calls_por_operador": "|".join(map(str, calls_op)), "tempo_s": dt,
    }


def _worker(args):
    return executar(*args)


def rodar(tarefas, out_csv, workers):
    feitos = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            feitos = {(r["braco"], int(r["nivel"]), r["instancia"], int(r["seed"]))
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
        tarefas = [(a, 1, s, n, SEED_BASE + sd) for (s, n, _) in insts if n in escolhidas
                   for a in list(ARMS) + ["ctrl_apps"] for sd in range(2)]
        rodar(tarefas, ROOT / "results" / "aos_study" / "etapa3_pilot.csv", args.workers)
    else:
        tarefas = [(a, nv, s, n, SEED_BASE + sd)
                   for nv in NIVEIS
                   for (s, n, nc) in sorted(insts, key=lambda t: -t[2])
                   for a in ARMS for sd in range(10)]
        rodar(tarefas, ROOT / "results" / "aos_study" / "etapa3_full.csv", args.workers)


if __name__ == "__main__":
    main()
