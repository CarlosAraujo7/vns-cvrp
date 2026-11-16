#!/usr/bin/env python3
"""
gerar_variantes_capacidade.py
=============================
Gera o conjunto SYN_CAP do Estudo 27 (diario, Entradas 145 e 147): o MESMO leiaute e as MESMAS demandas de instancias de SYN_REP (as 20 de 100 e as 20 de 200 clientes), com a CAPACIDADE mudada para
obter tamanhos medios de rota alvo de 4, 10 e 25 clientes (o numero de rotas K cai de cerca de n/4 a cerca de n/25) sem mexer em coordenadas nem demandas. Capacidade da variante = max(maior demanda,
ceil(r x soma das demandas / n)) (a formula do gerador de Uchoa; com demandas unitarias, floor(r)). Cada variante recebe o custo de referencia do HGS-CVRP (melhor de 2 execucoes, seeds 1 e 2, 0,2 s por
cliente, distancias arredondadas), gravado como `.sol`. Nomes: X-synq{bb}-n{dimensao}-r{r}, bb = indice da instancia-base em SYN_REP. Roda no WSL; retomavel (pula o que ja tem .sol).
Uso: HGS_THREADS=3 python3 -u estudo_aos/gerar_variantes_capacidade.py
"""
import csv
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
sys.path.insert(0, str(AQUI))
from gerar_instancias_sinteticas import hgs_melhor  # noqa: E402

BASE = ROOT / "instances" / "SYN_REP"
DEST = ROOT / "instances" / "SYN_CAP"
TAMANHOS = (100, 200)
ALVOS = (4, 10, 25)


def ler_vrp(path):
    linhas = path.read_text(encoding="utf-8").splitlines()
    dem = []
    i = linhas.index("DEMAND_SECTION") + 1
    while not linhas[i].startswith("DEPOT_SECTION"):
        dem.append(int(linhas[i].split()[1]))
        i += 1
    return linhas, dem[1:]                                                    # sem o deposito


def variante(linhas, demandas, alvo, nome, comentario):
    n = len(demandas)
    soma, dmax = sum(demandas), max(demandas)
    cap = math.floor(alvo) if soma == n else max(dmax, math.ceil(alvo * soma / n))
    out = []
    for l in linhas:
        if l.startswith("NAME"):
            l = f"NAME : {nome}"
        elif l.startswith("COMMENT"):
            l = f'COMMENT : "{comentario}"'
        elif l.startswith("CAPACITY"):
            l = f"CAPACITY : {int(cap)}"
        out.append(l)
    return "\n".join(out) + "\n", int(cap), math.ceil(soma / cap)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    itens = []
    for p in sorted(BASE.glob("*.vrp")):
        m = re.fullmatch(r"X-synr(\d+)-n(\d+)", p.stem)
        if not m or int(m.group(2)) - 1 not in TAMANHOS:
            continue
        linhas, dem = ler_vrp(p)
        for alvo in ALVOS:
            nome = f"X-synq{m.group(1)}-n{m.group(2)}-r{alvo:02d}"
            txt, cap, kmin = variante(linhas, dem, alvo, nome, f"Capacity variant of {p.stem} (SYN_REP): target mean route size {alvo}; coordinates and demands unchanged")
            vrp = DEST / f"{nome}.vrp"
            if not vrp.exists():
                vrp.write_text(txt, encoding="utf-8")
            itens.append((nome, p.stem, len(dem), alvo, cap, kmin, vrp))
    print(f"{len(itens)} variantes escritas ({len(itens) // len(ALVOS)} instancias-base); rodando HGS de referencia...", flush=True)

    def ref(item):
        nome, base, n, alvo, cap, kmin, vrp = item
        sol = vrp.with_suffix(".sol")
        if not sol.exists():
            custo, texto = hgs_melhor(vrp, n)
            sol.write_text(texto)
        t = sol.read_text()
        custo = [float(l.split()[1]) for l in t.splitlines() if l.startswith("Cost")][0]
        return nome, base, n, alvo, cap, kmin, custo, t.count("Route #")

    with ThreadPoolExecutor(max_workers=int(os.environ.get("HGS_THREADS", "3"))) as ex:
        res = list(ex.map(ref, itens))
    with open(DEST / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nome", "base", "n_clientes", "rota_alvo", "capacidade", "k_min", "custo_ref_hgs", "rotas_ref"])
        for r in res:
            w.writerow(r)
            print(f"  {r[0]:22s} n={r[2]:3d} Q={r[4]:5d} Kmin={r[5]:3d} ref={r[6]:.0f} K_ref={r[7]}", flush=True)
    print("SYN_CAP pronto", flush=True)


if __name__ == "__main__":
    main()
