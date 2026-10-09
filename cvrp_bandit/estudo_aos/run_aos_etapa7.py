#!/usr/bin/env python3
"""
run_aos_etapa7.py
=================
Runner da Etapa 7: selecao adaptativa do TIPO DE MOVIMENTO dentro da busca local do HGS-CVRP (estado da arte). PROTOCOLO
PRE-REGISTRADO em diario_pesquisa.md, Entrada 97 -- leia la antes de mexer.

Binario: external/HGS-CVRP-aos/hgs_aos (direcao1_local/build_hgs_aos.sh; fonte modificada em src/hgs_aos). Braco
`fixed` = ordem original do HGS (controle de regressao: identico ao binario original); `unif` = ordem aleatoria dos
movimentos aplicaveis de cada par (controle nao adaptativo); `ucb1`, `ts`, `pm`, `hg` = politicas adaptativas
(recompensa binaria: o movimento achou melhora). Orcamento = numero TOTAL de geracoes (`-maxit`), igual para todos os
bracos; uma execucao ate NMAX geracoes fornece, por prefixo, a melhor solucao viavel em cada ponto de controle
(`.PG.csv` com a geracao de cada melhora), pois o fluxo aleatorio do HGS e a politica sao deterministicos dado o seed.

Modos:
  --calib   calibracao dos orcamentos (instancias de calibracao e seeds 9001-9002, NAO entram em inferencia)
  --pilot   controle de regressao e viabilidade (nao entra em inferencia)
  --full    experimento pre-registrado

Uso (WSL): python3 -u run_aos_etapa7.py --full --workers 11     (resumivel por chave)
"""
import argparse
import csv
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from instance import load_all  # noqa: E402
from solver_hgs import _ler_solucao  # noqa: E402

HGS = ROOT / "external" / "HGS-CVRP-aos" / "hgs_aos"
RES = ROOT / "results" / "aos_study"
BRACOS = ["fixed", "unif", "ucb1", "ts", "pm", "hg"]
# pontos de controle (geracoes) em que se le a melhor solucao viavel; o ultimo e o orcamento total
CHECKS = [250, 500, 1000, 2000, 4000]
CAMPOS = ["conjunto", "instancia", "n_customers", "braco", "seed", "nmax", "ref", "viavel", "custo_final",
          "gap_final"] + [f"gap_{c}" for c in CHECKS] + ["cpu_s", "tentativas", "sucessos"]


def instancias(conjuntos):
    out = []
    for c in conjuntos:
        for i in load_all(ROOT / "instances" / c):
            out.append((c, i.name))
    return out


def executar(args):
    conjunto, nome, braco, seed, nmax = args
    inst = next(i for i in load_all(ROOT / "instances" / conjunto) if i.name == nome)
    ids = list(inst.coords)
    assert ids[0] == inst.depot_id
    with tempfile.TemporaryDirectory(prefix="hgsaos_") as tmp:
        sol, st = Path(tmp) / "out.sol", Path(tmp) / "stats.csv"
        cmd = [str(HGS), str(inst.source_vrp), str(sol), "-it", "10000000", "-maxit", str(nmax), "-seed", str(seed),
               "-round", "1" if inst.round_distances else "0", "-log", "0", "-aos", braco]
        t0 = os.times()
        subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "HGS_AOS_STATS": str(st)})
        t1 = os.times()
        cpu = (t1.children_user + t1.children_system) - (t0.children_user + t0.children_system)
        if not sol.exists():
            raise RuntimeError(f"sem solucao: {nome} {braco} {seed}")
        rotas = _ler_solucao(sol, ids)
        pg = Path(str(sol) + ".PG.csv")
        prog = []
        for linha in pg.read_text().splitlines():
            p = linha.split(";")
            prog.append((int(p[4]), float(p[2])))          # (geracao, custo da melhor solucao viavel ate ali)
        tent, suc = ("", "")
        if st.exists():
            linhas = st.read_text().splitlines()[1:]
            tent = "|".join(l.split(",")[1] for l in linhas)
            suc = "|".join(l.split(",")[2] for l in linhas)
    viavel, _ = inst.is_feasible(rotas)
    custo = inst.solution_cost(rotas)
    # o HGS imprime o custo com 6 algarismos significativos (CMT usa distancias reais): tolerancia relativa de 1e-5
    assert not prog or abs(prog[-1][1] - custo) <= 1e-5 * max(1.0, abs(custo)) or not viavel, (nome, braco, seed, prog[-1], custo)
    ref = inst.optimum

    def gap_em(g):
        c = [cc for (it, cc) in prog if it <= g]
        return 100 * (min(c) - ref) / ref if c else float("nan")

    linha = dict(conjunto=conjunto, instancia=nome, n_customers=inst.n_customers, braco=braco, seed=seed, nmax=nmax,
                 ref=ref, viavel=viavel, custo_final=custo, gap_final=100 * (custo - ref) / ref, cpu_s=cpu,
                 tentativas=tent, sucessos=suc)
    for c in CHECKS:
        linha[f"gap_{c}"] = gap_em(c) if c <= nmax else float("nan")
    return linha


def rodar(tarefas, out_csv, workers):
    feitos = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            feitos = {(r["conjunto"], r["instancia"], r["braco"], int(r["seed"])) for r in csv.DictReader(f)}
    pend = [t for t in tarefas if (t[0], t[1], t[2], t[3]) not in feitos]
    print(f"{len(tarefas)} tarefas, {len(feitos)} ja feitas, {len(pend)} pendentes", flush=True)
    novo = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(out_csv, "a", newline="", encoding="utf-8") as f, mp.Pool(workers) as pool:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if novo:
            w.writeheader()
        for i, linha in enumerate(pool.imap_unordered(executar, pend, chunksize=1), 1):
            w.writerow(linha)
            if i % 50 == 0:
                f.flush()
                print(f"  {i}/{len(pend)}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    print("concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--calib", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--nmax", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    if a.calib:
        insts = [("X", "X-n101-k25"), ("X", "X-n153-k22"), ("X", "X-n186-k15"), ("CMT", "CMT2"), ("SYN_TUNE", None)]
        base = instancias(["SYN_TUNE"])
        insts = [i for i in insts if i[1]] + [base[0], base[len(base) // 2], base[-1]]
        tarefas = [(c, n, "fixed", 9001 + s, a.nmax) for (c, n) in insts for s in range(2)]
        rodar(tarefas, RES / "etapa7_calib.csv", a.workers)
    elif a.pilot:
        insts = [("X", "X-n101-k25"), ("X", "X-n153-k22"), ("CMT", "CMT1")]
        tarefas = [(c, n, b, 9001, 500) for (c, n) in insts for b in BRACOS]
        rodar(tarefas, RES / "etapa7_pilot.csv", a.workers)
    else:
        insts = instancias(["X", "CMT", "SYN_VAL"])
        tarefas = [(c, n, b, 1000 + s, a.nmax) for (c, n) in sorted(insts, key=lambda t: t[1], reverse=True)
                   for b in BRACOS for s in range(a.seeds)]
        rodar(tarefas, RES / "etapa7_full.csv", a.workers)


if __name__ == "__main__":
    main()
