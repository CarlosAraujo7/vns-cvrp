#!/usr/bin/env python3
"""
run_aos_etapa15.py
==================
Estudo 15: teste PROSPECTIVO da regularidade "a selecao adaptativa vence a uniforme exatamente quando o pool contem operadores que nao valem
o seu custo". Pools ANINHADOS P5 < P6 < P7 < P8 (prefixos de P9) no TSP (Estudo 13) e no CVRPTW (Estudo 14); o "peso morto" D_k = gap(U, P_k) -
gap(U, P5) e medido SO nas instancias de ajuste (disjuntas da avaliacao) e fixa a previsao ANTES da avaliacao.

Modos: --tuning-pools (mede D_k), --predizer (grava a previsao), --eval-pools (avaliacao). Retomavel. Uso (WSL): python3 -u run_aos_etapa15.py --eval-pools
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import run_aos_etapa13 as r13  # noqa: E402
import run_aos_etapa14 as r14  # noqa: E402

RES = r13.RES
POOLS_K = (5, 6, 7, 8)
LIMIAR_D = 0.30            # pp: previsao "ha ganho" se D_k >= 0,30 pp
SEEDS_TUNE = [3000, 3001, 3002, 3003, 3004]
SEMENTES_AVAL = 1000
for k in (5, 6, 7, 8, 9):
    r13.POOLS13[f"P{k}"] = (r13.P9[:k], [])
    r14.POOLS[f"P{k}"] = (r14.P9[:k], [])


def _ordem(prob, k):
    if prob == "t":
        return list(range(k))
    custo = r14._ordens()[0]
    return [i for i in custo if i < k]


def _arms(prob, pool, k, pref):
    sv = (r13 if prob == "t" else r14).spec_v
    n = lambda a: f"{pref}_{pool}_{a}"  # noqa: E731
    d = {"U": sv(n("U"), "U", pool=pool),
         "ucb1c": sv(n("ucb1c"), "ucb1", custo=True, params={"C": 2.0}, pool=pool),
         "pmc": sv(n("pmc"), "pm", custo=True, params={"gamma": 0.1, "p_min": 0.05}, pool=pool),
         "hgucbc": sv(n("hgucbc"), "hgucb", custo=True, params={"alpha": 0.5, "C": 2.0}, pool=pool),
         "ts": sv(n("ts"), "ts", pool=pool),
         "exp3": sv(n("exp3"), "exp3", params={"gamma": 0.1}, pool=pool),
         "eff": sv(n("eff"), "eff", params={"C": 1.0}, pool=pool),
         "FOC": dict(sv(n("FOC"), "FOC", pool=pool), ordem=_ordem(prob, k))}
    for sp in d.values():
        sp["config_id"] = "e15"
    return d


def modo_tuning_pools(workers):
    for prob, mod, conj in (("t", r13, "SYN_TUNE_T"), ("w", r14, "SYN_TUNE_W")):
        tar = []
        for k in POOLS_K + (9,):
            sp = dict(mod.spec_v(f"tune15{prob}_P{k}_U", "U", pool=f"P{k}"), config_id="e15t")
            tar += [(sp, 1, conj, n, s) for n in sorted(mod.nomes(conj), reverse=True) for s in SEEDS_TUNE]
        print(f"tuning de pools ({prob}): {len(tar)} execucoes", flush=True)
        with mp.Pool(workers) as pool:
            mod.rodar(tar, RES / f"etapa15_tuning_pools_{prob}.csv", pool, mod.CAMPOS_V, cada=100)


def modo_predizer():
    import pandas as pd
    out = {}
    for prob in ("t", "w"):
        d = pd.read_csv(RES / f"etapa15_tuning_pools_{prob}.csv")
        d = d[d.viavel]
        g = d.groupby(["braco", "instancia"]).gap_pct.mean().groupby("braco").mean()
        base = g[f"tune15{prob}_P5_U"]
        out[prob] = {}
        for k in POOLS_K + (9,):
            D = float(g[f"tune15{prob}_P{k}_U"] - base)
            out[prob][f"P{k}"] = {"D": round(D, 4), "previsao": "ganho" if D >= LIMIAR_D else "nenhum"}
    (RES / "etapa15_predicoes.json").write_text(json.dumps({"limiar_D_pp": LIMIAR_D, "pools": out}, indent=1))
    print(json.dumps(out, indent=1))


def modo_eval_pools(workers):
    for prob, mod, conjs, pref in (("t", r13, (("TSPLIB", 10), ("SYN_VAL_T", 5)), "s15t"), ("w", r14, (("SOLOMON", 10),), "s15w")):
        tar = []
        for k in POOLS_K:
            arms = _arms(prob, f"P{k}", k, pref)
            for conj, ns in conjs:
                for n in sorted(mod.nomes(conj), reverse=True):
                    tar += [(sp, 1, conj, n, SEMENTES_AVAL + sd) for sp in arms.values() for sd in range(ns)]
        print(f"eval pools ({prob}): {len(tar)} execucoes", flush=True)
        with mp.Pool(workers) as pool:
            mod.rodar(tar, RES / f"etapa15_eval_{prob}.csv", pool, mod.CAMPOS_V, cada=200)
        print(f"eval pools ({prob}) concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("tuning-pools", "predizer", "eval-pools"):
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.tuning_pools:
        modo_tuning_pools(a.workers)
    elif a.predizer:
        modo_predizer()
    else:
        modo_eval_pools(a.workers)


if __name__ == "__main__":
    main()
