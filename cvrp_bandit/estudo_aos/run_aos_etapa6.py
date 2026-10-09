#!/usr/bin/env python3
"""
run_aos_etapa6.py
=================
Runner da Etapa 6 (familias adicionais de AOS e politica de eficiencia por
avaliacao de distancia). PROTOCOLO PRE-REGISTRADO em diario_pesquisa.md,
Entrada 91 -- leia la antes de mexer.

Estrutura fixa (a mesma do Estudo 4): P10, termino R5, aceitacao "best", shake
dos 5 operadores originais, orcamento E1 = 600 n^2 e E2 = 5 E1 em avaliacoes de
distancia, 74 instancias, seeds 1000-1009, solucao inicial fixada em disco.
Braco de referencia (reaproveitado): s3_R5_U.

Uso (WSL, venv de direcao1_local):
    python3 -u run_aos_etapa6.py --pilot
    python3 -u run_aos_etapa6.py --full --workers 11
    python3 -u run_aos_etapa6.py --val --workers 11      (30 instancias sinteticas, E1, 5 seeds; exploratorio)
Resumivel por chave (braco, config, nivel, instancia, seed).
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(AQUI))
import run_aos_etapa4 as r4  # noqa: E402
from run_aos_study import todas_instancias  # noqa: E402

RES = ROOT / "results" / "aos_study"
# braco -> (politica, parametros)
ARMS = {"s6_R5_exp3": ("exp3", {"gamma": 0.1}),
        "s6_R5_ts": ("ts", {}),
        "s6_R5_swucb": ("swucb", {"window": 200, "xi": 0.6}),
        "s6_R5_ap": ("ap", {"alpha": 0.1, "beta": 0.1, "p_min": 0.05}),
        "s6_R5_eff": ("eff", {"C": 1.0}),
        "s6_R5_eff2": ("eff", {"C": 2.0}),
        # extensao do pre-registro (Entrada 93, antes de qualquer dado): politicas contextuais
        "s6_R5_ctxts": ("ctx_ts", {"k0": 5.0}),
        "s6_R5_ctxeff": ("ctx_eff", {"k0": 5.0})}


def specs():
    return {n: r4.spec(n, pol, termino="R", custo=False, params=par, config_id="e6", pool="P10")
            for n, (pol, par) in ARMS.items()}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--val", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    sp = specs()
    insts = todas_instancias()
    if a.pilot:
        esc = {"A-n32-k5", "B-n41-k6", "CMT1", "X-n101-k25"}
        ctrl = r4.spec("ctrl_U", "U", termino="R", custo=False, params=None, config_id="e6", pool="P10")
        tarefas = [(s, 1, c, n, 1000 + sd) for (c, n, _) in insts if n in esc
                   for s in list(sp.values()) + [ctrl] for sd in range(2)]
        out = RES / "etapa6_pilot.csv"
    elif a.full:
        tarefas = [(s, nv, c, n, 1000 + sd) for nv in (1, 5)
                   for (c, n, nc) in sorted(insts, key=lambda t: -t[2]) for s in sp.values() for sd in range(10)]
        out = RES / "etapa6_full.csv"
    else:
        tarefas = [(s, 1, "SYN_VAL", n, sd) for n in r4.nomes("SYN_VAL") for s in sp.values() for sd in r4.SEEDS_VAL]
        out = RES / "etapa6_val.csv"
    print(f"{len(tarefas)} tarefas", flush=True)
    with mp.Pool(a.workers) as pool:
        r4.rodar(tarefas, out, pool)
    print("concluido", flush=True)


if __name__ == "__main__":
    main()
