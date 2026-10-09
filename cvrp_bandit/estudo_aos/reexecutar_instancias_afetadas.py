#!/usr/bin/env python3
"""
reexecutar_instancias_afetadas.py
=================================
Correcao do achado F1 (diario, Entrada 88): reexecuta, com a solucao inicial
FIXADA em disco (src/solucao_inicial.py), TODAS as execucoes dos Estudos 1-4
nas instancias cuja solucao inicial era nao deterministica (as que usaram o
fallback CP-SAT; lista em instances/_init/_relatorio.csv, coluna
usou_fallback_cpsat). Passos, para cada CSV de resultados:
  1. arquiva o CSV original em results/aos_study/superseded_init_nao_det/
     (uma unica vez; nunca sobrescreve um arquivo ja arquivado);
  2. remove do CSV as linhas das instancias afetadas;
  3. reexecuta as tarefas dessas instancias com o runner do estudo (resumivel).
Roda no WSL:  python3 -u reexecutar_instancias_afetadas.py [--workers 11]
"""
import argparse
import csv
import json
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(AQUI))
RES = ROOT / "results" / "aos_study"
ARQ = RES / "superseded_init_nao_det"


def afetadas():
    rel = ROOT / "instances" / "_init" / "_relatorio.csv"
    return {r["instancia"]: r["conjunto"] for r in csv.DictReader(open(rel, encoding="utf-8"))
            if r["usou_fallback_cpsat"] == "True"}


def arquivar_e_filtrar(nome_csv, coluna_inst, nomes):
    f = RES / nome_csv
    if not f.exists():
        return 0
    ARQ.mkdir(exist_ok=True)
    dest = ARQ / nome_csv
    if not dest.exists():
        shutil.copy2(f, dest)
    with open(f, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        campos = rd.fieldnames
        linhas = list(rd)
    mantidas = [r for r in linhas if r[coluna_inst] not in nomes]
    with open(f, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(mantidas)
    return len(linhas) - len(mantidas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    af = afetadas()
    print("instancias afetadas:", sorted(af), flush=True)
    nomes = set(af)
    # ---- Estudo 1 (+ subconjunto exploratorio B=20000) ----
    import run_aos_study as r1
    rem = arquivar_e_filtrar("full.csv", "instancia", nomes)
    print(f"Estudo 1: {rem} linhas removidas (arquivadas em {ARQ.name}/)", flush=True)
    t1 = [(arm, B, af[n], n, r1.SEED_BASE + sd) for n in sorted(nomes) for B in (800, 4000)
          for arm in r1.ARMS for sd in range(10)]
    r1.rodar(t1, RES / "full.csv", a.workers)
    rem = arquivar_e_filtrar("exploratorio_b20000.csv", "instancia", nomes)
    t1b = [(arm, 20000, af[n], n, r1.SEED_BASE + sd) for n in sorted(nomes) for arm in
           ("gvns_classico", "uniforme", "hgucb_norm") for sd in range(5)]
    sub = {r["instancia"] for r in csv.DictReader(open(ARQ / "exploratorio_b20000.csv", encoding="utf-8"))}
    t1b = [t for t in t1b if t[3] in sub and rem > 0]
    if t1b:
        r1.rodar(t1b, RES / "exploratorio_b20000.csv", a.workers)
    # ---- Estudo 2 ----
    import run_aos_etapa2 as r2
    rem = arquivar_e_filtrar("etapa2_full.csv", "instancia", nomes)
    print(f"Estudo 2: {rem} linhas removidas", flush=True)
    t2 = [(arm, B, af[n], n, r2.SEED_BASE + sd) for n in sorted(nomes) for B in r2.BUDGETS
          for arm in r2.ARMS for sd in range(10)]
    r2.rodar(t2, RES / "etapa2_full.csv", a.workers)
    # ---- Estudo 3 ----
    import run_aos_etapa3 as r3
    rem = arquivar_e_filtrar("etapa3_full.csv", "instancia", nomes)
    print(f"Estudo 3: {rem} linhas removidas", flush=True)
    t3 = [(arm, nv, af[n], n, r3.SEED_BASE + sd) for n in sorted(nomes) for nv in r3.NIVEIS
          for arm in r3.ARMS for sd in range(10)]
    r3.rodar(t3, RES / "etapa3_full.csv", a.workers)
    # ---- Estudo 4 (avaliacao nas 74) ----
    import run_aos_etapa4 as r4
    rem = arquivar_e_filtrar("etapa4_eval74.csv", "instancia", nomes)
    print(f"Estudo 4 (eval74): {rem} linhas removidas", flush=True)
    sp = r4.specs_tuned()
    instr = r4.spec("s4_R5_U_instr", "U", custo=False, config_id="instr")
    t4 = [(s, nv, af[n], n, sd) for n in sorted(nomes) for nv in (1, 5) for s in sp.values() for sd in r4.SEEDS_TEST]
    t4 += [(instr, 1, af[n], n, sd) for n in sorted(nomes) for sd in r4.SEEDS_TEST]
    with mp.Pool(a.workers) as pool:
        r4.rodar(t4, RES / "etapa4_eval74.csv", pool)
    print("reexecucao concluida", flush=True)


if __name__ == "__main__":
    main()
