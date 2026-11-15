#!/usr/bin/env python3
"""
piloto_etapa21.py -- PILOTO do Estudo 21 (diario, Entrada 136), SO nas instancias de ajuste SYN_TUNE (disjuntas das de avaliacao).
(a) cada operador NOVO sozinho: destruicao j com o reparo regret-2 fixo; reparo j com a destruicao `aleatoria` fixa (a de referencia e a do Estudo 9
    correspondente); 24 instancias x 1 seed (3100); orcamento kappa = 1.500 como nos Estudos 9 e 20;
(b) a selecao uniforme (U) em P7, P12 e P21 (24 x 1 seed): quanto o conjunto maior dilui a escolha uniforme.
Nao decide nada do desenho confirmatorio alem de conferir que os operadores rodam, nao sao patologicos e gastam o orcamento; o resumo e registrado
no diario.   Uso (WSL): python3 -u piloto_etapa21.py --workers 11
"""
import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa4 as r4  # noqa: E402
import run_aos_etapa9 as r9  # noqa: E402
import alns_ext as ax  # noqa: E402
from alns_loop import rodar_alns  # noqa: E402
from aos_vnd import EstaticaPolitica  # noqa: E402

SEED = 3100


def _um(args):
    rotulo, pool, pd_, pr_, inst_nome = args
    D, R = ax.POOLS[pool]
    inst, cws = r9._instancia("SYN_TUNE", inst_nome)
    px = r9._proxy("SYN_TUNE", inst_nome)
    px.chamadas = 0
    rng = random.Random(SEED)
    pol_d = EstaticaPolitica(pd_, rng=random.Random(SEED * 31 + 1))
    pol_r = EstaticaPolitica(pr_, rng=random.Random(SEED * 31 + 2))
    orc = int(r9.KAPPA9 * inst.n_customers ** 2)
    max_rotas = inst.n_vehicles_ref if inst.vehicles_exact else None
    t0 = time.perf_counter()
    best, info = rodar_alns(px, [list(r) for r in cws], rng, orc, pol_d, "score", pol_r, "score", lambda: px.chamadas, w0=r9.W09, tf_fator=r9.TF9,
                            max_rotas=max_rotas, destruir=D, reparar=R)
    gap = 100 * (inst.solution_cost(best) - inst.optimum) / inst.optimum
    viavel, _ = inst.is_feasible(best)
    return {"rotulo": rotulo, "inst": inst_nome, "gap": gap, "viavel": bool(viavel), "iter": info["iteracoes"], "inviaveis": info["inviaveis"],
            "t": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    nomes = sorted(r4.nomes("SYN_TUNE"), reverse=True)
    D21, R21 = ax.POOLS["P21"]
    kd, kr = len(D21), len(R21)
    tarefas = []

    def so(kd_, j, k):
        w = [0.0] * k
        w[j] = 1.0
        return w

    # (a) cada operador sozinho (P21): destruicao j + regret2 ; aleatoria + reparo j
    for j, (nd, _f) in enumerate(D21):
        for n in nomes:
            tarefas.append((f"D:{nd}", "P21", so(kd, j, kd), so(kr, 1, kr), n))
    for j, (nr, _f) in enumerate(R21):
        for n in nomes:
            tarefas.append((f"R:{nr}", "P21", so(kd, 0, kd), so(kr, j, kr), n))
    # (b) uniforme em P7, P12, P21
    for p in ("P7", "P12", "P21"):
        D, R = ax.POOLS[p]
        for n in nomes:
            tarefas.append((f"U:{p}", p, [1.0] * len(D), [1.0] * len(R), n))
    print(f"piloto: {len(tarefas)} execucoes", flush=True)
    t0 = time.perf_counter()
    with mp.Pool(a.workers) as pool:
        res = pool.map(_um, tarefas, chunksize=1)
    import pandas as pd
    d = pd.DataFrame(res)
    out = d.groupby("rotulo").agg(gap=("gap", "mean"), inv=("inviaveis", "sum"), it=("iter", "mean"), t=("t", "mean"), n=("gap", "size"),
                                  viaveis=("viavel", "mean")).round(3)
    print(out.to_string(), flush=True)
    (ROOT / "results" / "aos_study").mkdir(parents=True, exist_ok=True)
    d.to_csv(ROOT / "results" / "aos_study" / "etapa21_piloto.csv", index=False)
    print(f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
