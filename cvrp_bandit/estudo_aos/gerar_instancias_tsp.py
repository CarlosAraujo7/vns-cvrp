#!/usr/bin/env python3
"""
gerar_instancias_tsp.py
=======================
Conjuntos sinteticos do Estudo 13 (TSP): instancias euclidianas com coordenadas inteiras em [0, 1000]^2, metade uniformes e
metade agrupadas (3 a 8 centros uniformes, pontos gaussianos com desvio 30-80, truncados na caixa), distancia EUC_2D (nint).
  SYN_TUNE_T : 24 instancias de AJUSTE  (8 com 50, 8 com 100, 8 com 150 nos)         semente 20261028
  SYN_VAL_T  : 30 instancias de AVALIACAO (10 com 100, 10 com 150, 10 com 200 nos)   semente 20261029
Disjuntas entre si e da TSPLIB. Referencia = melhor de DUAS chamadas independentes do LKH-3 (RUNS=2, MAX_TRIALS=N, tempo
0,2 s por no cada, sementes 11 e 12); nao e otimo provado -- registramos se as duas chamadas concordam.
Saidas: instances/<conjunto>/<nome>.tsp e instances/<conjunto>/ref.csv. Pula conjuntos que ja existem.
Uso (WSL, venv): python gerar_instancias_tsp.py
"""
import csv
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tsp_inst import TSPInst, escrever_tsplib  # noqa: E402
from solver_lkh_tsp import resolver_tsp_lkh  # noqa: E402

CONJUNTOS = {"SYN_TUNE_T": (20261028, [50] * 8 + [100] * 8 + [150] * 8, "synt"),
             "SYN_VAL_T": (20261029, [100] * 10 + [150] * 10 + [200] * 10, "synv")}


def gerar(rng, n, tipo):
    if tipo == "uniforme":
        pts = [(rng.randint(0, 1000), rng.randint(0, 1000)) for _ in range(n)]
    else:
        k = rng.randint(3, 8)
        centros = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(k)]
        sd = rng.uniform(30, 80)
        pts = []
        for _ in range(n):
            cx, cy = rng.choice(centros)
            pts.append((min(1000, max(0, round(rng.gauss(cx, sd)))), min(1000, max(0, round(rng.gauss(cy, sd))))))
    return pts


def main():
    for conj, (semente, tamanhos, tag) in CONJUNTOS.items():
        pasta = ROOT / "instances" / conj
        if (pasta / "ref.csv").exists():
            print(f"{conj} ja existe; pulando")
            continue
        pasta.mkdir(parents=True, exist_ok=True)
        rng = random.Random(semente)
        linhas = []
        for idx, n in enumerate(tamanhos):
            tipo = "uniforme" if idx % 2 == 0 else "agrupada"
            nome = f"T-{tag}{idx:02d}-n{n}-{tipo[:3]}"
            inst = TSPInst(nome, gerar(rng, n, tipo))
            escrever_tsplib(inst, pasta / f"{nome}.tsp")
            res = [resolver_tsp_lkh(inst, runs=2, tempo_limite=0.2 * n, seed=s)[1] for s in (11, 12)]
            linhas.append(dict(instancia=nome, n=n, tipo=tipo, ref=int(min(res)), lkh1=int(res[0]), lkh2=int(res[1]),
                               concordam=res[0] == res[1]))
            print(linhas[-1], flush=True)
        with open(pasta / "ref.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0]))
            w.writeheader()
            w.writerows(linhas)
        print(f"{conj}: {len(linhas)} instancias; as duas chamadas do LKH concordam em {sum(l['concordam'] for l in linhas)}")


if __name__ == "__main__":
    main()
