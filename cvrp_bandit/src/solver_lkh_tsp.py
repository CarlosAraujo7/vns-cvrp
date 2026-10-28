"""
solver_lkh_tsp.py
=================
Referencia do Estudo 13: LKH-3 (binario em external/LKH-3.0.7/LKH, Linux) aplicado a uma TSPInst. Para o TSP simetrico o
LKH-3 e o LKH classico de Helsgaun: acha o otimo conhecido de instancias de ate algumas centenas de nos em segundos
(verificado nas 32 instancias da TSPLIB por estudo_aos/verifica_tsplib_lkh.py). O custo devolvido e SEMPRE recalculado por
`inst.solution_cost` (nunca o impresso pelo LKH).

    tour, custo, tempo = resolver_tsp_lkh(inst, runs=2, tempo_limite=40.0, seed=1)
"""
import subprocess
import tempfile
import time
from pathlib import Path

from tsp_inst import escrever_tsplib

LKH_BIN = Path(__file__).resolve().parent.parent / "external" / "LKH-3.0.7" / "LKH"


def resolver_tsp_lkh(inst, runs=2, tempo_limite=None, seed=1, max_trials=None):
    tempo_limite = tempo_limite if tempo_limite is not None else 0.2 * inst.N
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        escrever_tsplib(inst, d / "i.tsp")
        par = (f"PROBLEM_FILE = {d / 'i.tsp'}\nOUTPUT_TOUR_FILE = {d / 'i.tour'}\nRUNS = {runs}\n"
               f"MAX_TRIALS = {max_trials if max_trials is not None else inst.N}\nSEED = {seed}\n"
               f"TIME_LIMIT = {tempo_limite}\nTRACE_LEVEL = 0\n")
        (d / "i.par").write_text(par)
        t0 = time.time()
        subprocess.run([str(LKH_BIN), str(d / "i.par")], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=tempo_limite * 3 + 120)
        tempo = time.time() - t0
        ids, lendo = [], False
        for ln in (d / "i.tour").read_text().splitlines():
            s = ln.strip()
            if s.startswith("TOUR_SECTION"):
                lendo = True
            elif lendo:
                v = int(s.split()[0])
                if v == -1:
                    break
                ids.append(v - 1)
    k = ids.index(0)
    ids = ids[k:] + ids[:k]                                  # rotaciona para comecar no no 0
    tour = ids[1:]
    assert sorted(tour) == list(range(1, inst.N))
    if inst.solution_cost([tour[::-1]]) < inst.solution_cost([tour]):
        tour = tour[::-1]                                    # sentido irrelevante (simetrico); so por reprodutibilidade
    return tour, inst.solution_cost([tour]), tempo
