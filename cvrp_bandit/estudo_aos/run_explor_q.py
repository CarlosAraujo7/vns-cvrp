#!/usr/bin/env python3
"""
run_explor_q.py -- PILOTO EXPLORATORIO (SYN_TUNE; nao e confirmatorio) do controle adaptativo do TAMANHO da destruicao q no ALNS.
Pergunta: o desempenho depende de q a ponto de uma politica sobre niveis de q ter folga maior que a da selecao de operadores?
Operadores de destruicao/reparo: uniformes. Niveis de q (fracao de n; teto 80): L1 [0,03; 0,08], L2 [0,08; 0,15], L3 [0,15; 0,30].
Bracos: q_mixed (a faixa do Estudo 9: U[max(5,0,1n), min(0,3n,60)]), q_fix1, q_fix2, q_fix3 (um nivel fixo), q_ucb1, q_rp, q_ts
(politica sobre os 3 niveis). ALNS e orcamento da calibracao do Estudo 9 (1500 n^2, w0=0,02, Tf/T0=1e-3).
Uso (WSL): python3 -u run_explor_q.py --workers 1
"""
import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_aos_study import _instancia  # noqa: E402
from run_aos_etapa3 import _proxy  # noqa: E402
import run_aos_etapa4 as r4  # noqa: E402
from alns import DESTRUIR, REPARAR  # noqa: E402
from alns_loop import construir_politica_alns  # noqa: E402
import alns_loop_q  # noqa: E402

RES = ROOT / "results" / "aos_study"
LEVELS = [(0.03, 0.08), (0.08, 0.15), (0.15, 0.30)]
ARMS = {"q_mixed": (None, None), "q_fix1": ([LEVELS[0]], None), "q_fix2": ([LEVELS[1]], None),
        "q_fix3": ([LEVELS[2]], None), "q_ucb1": (LEVELS, "ucb1"), "q_rp": (LEVELS, "RP"), "q_ts": (LEVELS, "ts")}
CAMPOS = ["braco", "instancia", "n_customers", "seed", "custo", "ref", "gap_pct", "viavel", "iteracoes", "uso_q", "tempo_s"]


def executar(arm, nome_inst, seed):
    inst, cws = _instancia("SYN_TUNE", nome_inst)
    px = _proxy("SYN_TUNE", nome_inst)
    px.chamadas = 0
    rng = random.Random(seed)
    pol_d, md = construir_politica_alns("U", len(DESTRUIR), random.Random(seed * 31 + 1), {})
    pol_r, mr = construir_politica_alns("U", len(REPARAR), random.Random(seed * 31 + 2), {})
    levels, qpol = ARMS[arm]
    pol_q, mq = (None, None)
    if qpol:
        pol_q, mq = construir_politica_alns(qpol, len(levels), random.Random(seed * 31 + 3), {})
    orc = int(1500 * inst.n_customers ** 2)
    t0 = time.perf_counter()
    best, info = alns_loop_q.rodar_alns(px, [list(r) for r in cws], rng, orc, pol_d, md, pol_r, mr, lambda: px.chamadas,
                                        w0=0.02, tf_fator=1e-3, qlevels=levels, pol_q=pol_q, modo_q=mq)
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(best)
    viavel, _ = inst.is_feasible(best)
    return {"braco": arm, "instancia": inst.name, "n_customers": inst.n_customers, "seed": seed, "custo": custo,
            "ref": inst.optimum, "gap_pct": 100 * (custo - inst.optimum) / inst.optimum, "viavel": viavel,
            "iteracoes": info["iteracoes"], "uso_q": "|".join(map(str, info["uso_q"])), "tempo_s": dt}


def _w(a):
    return executar(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    out = RES / "etapa12_pilot_q.csv"
    feitos = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            feitos = {(r["braco"], r["instancia"], int(r["seed"])) for r in csv.DictReader(f)}
    nm = sorted(r4.nomes("SYN_TUNE"), reverse=True)
    tarefas = [(arm, n, s) for n in nm for arm in ARMS for s in r4.SEEDS_TUNE[:3] if (arm, n, s) not in feitos]
    print(f"{len(tarefas)} tarefas pendentes", flush=True)
    novo = not out.exists()
    with mp.Pool(a.workers) as pool, open(out, "a", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=CAMPOS)
        if novo:
            w.writeheader()
        for k, linha in enumerate(pool.imap_unordered(_w, tarefas, chunksize=1), 1):
            w.writerow(linha)
            f_out.flush()
            if k % 20 == 0:
                print(f"    {k}/{len(tarefas)}", flush=True)
    print("concluido", flush=True)


if __name__ == "__main__":
    main()
