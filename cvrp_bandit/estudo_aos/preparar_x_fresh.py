#!/usr/bin/env python3
"""
preparar_x_fresh.py
===================
Prepara o conjunto X_FRESH do Estudo 26 (diario, Entradas 145 e 146): as 38 instancias X de Uchoa et al. (2017) com 200 a 401 clientes (X-n200-k36 a X-n401-k29), que nenhum braco dos Estudos 1 a 25 usou
(o conjunto X do projeto tem as 21 instancias de 100 a 195 clientes). Os arquivos .vrp vem do repositorio publico vidalt/HGS-CVRP (licenca MIT; pasta Instances/CVRP, as instancias originais da CVRPLIB);
o servidor da CVRPLIB (galgos.inf.puc-rio.br) estava fora do ar em 16/11/2026. O custo de REFERENCIA nao e o BKS da CVRPLIB (sem acesso): e o melhor de 2 execucoes do HGS-CVRP (seeds 1 e 2, t = 0,4 s por
cliente, distancias arredondadas), gravado como `.sol` no formato CVRPLIB (como nos conjuntos sinteticos; o custo de referencia NAO e otimo provado, e os gaps sao relativos a ele). Roda no WSL.
Uso: HGS_THREADS=3 python3 -u estudo_aos/preparar_x_fresh.py        (retomavel: pula o que ja tem .sol)
"""
import csv
import hashlib
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
sys.path.insert(0, str(AQUI))
from gerar_instancias_sinteticas import hgs_melhor  # noqa: E402

DEST = ROOT / "instances" / "X_FRESH"
BASE = "https://raw.githubusercontent.com/vidalt/HGS-CVRP/master/Instances/CVRP/"
API = "https://api.github.com/repos/vidalt/HGS-CVRP/contents/Instances/CVRP"
N_MIN, N_MAX = 200, 401
FATOR = 0.4


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pesquisa-cvrp-leitura"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def lista():
    nomes = []
    for x in json.loads(get(API)):
        m = re.fullmatch(r"(X-n(\d+)-k(\d+))\.vrp", x["name"])
        if m and N_MIN <= int(m.group(2)) <= N_MAX:
            nomes.append((m.group(1), int(m.group(2)) - 1))     # n clientes = dimensao - 1 = n - 1 (o n do nome e a dimensao)
    return sorted(nomes, key=lambda t: t[1])


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    itens = lista()
    print(f"{len(itens)} instancias X de {N_MIN} a {N_MAX} (dimensao) no repositorio", flush=True)
    origem = []
    for nome, _ in itens:
        f = DEST / f"{nome}.vrp"
        if not f.exists():
            f.write_bytes(get(BASE + f"{nome}.vrp"))
        origem.append((nome, f.stat().st_size, hashlib.sha256(f.read_bytes()).hexdigest()))
    (DEST / "ORIGEM.txt").write_text("Origem: https://github.com/vidalt/HGS-CVRP/tree/master/Instances/CVRP (licenca MIT), baixado em 16/11/2026; as instancias X originais da CVRPLIB (Uchoa et al., 2017).\n"
                                     "Referencia: melhor de 2 execucoes do HGS-CVRP (seeds 1 e 2, t = 0,4 s por cliente, -round 1), NAO o BKS da CVRPLIB.\n"
                                     + "\n".join(f"{n}.vrp {t} bytes sha256 {h}" for n, t, h in origem) + "\n", encoding="utf-8")

    def ref(item):
        nome, n = item
        vrp = DEST / f"{nome}.vrp"
        sol = vrp.with_suffix(".sol")
        if sol.exists():
            return nome, n, None, None
        custo, texto = hgs_melhor(vrp, n, fator=FATOR)
        sol.write_text(texto)
        return nome, n, custo, texto.count("Route #")

    with ThreadPoolExecutor(max_workers=int(os.environ.get("HGS_THREADS", "3"))) as ex:
        for nome, n, custo, k in ex.map(ref, itens):
            print(f"  {nome:16s} n={n:3d} " + ("(ja existia)" if custo is None else f"ref={custo:.0f} K={k}"), flush=True)
    with open(DEST / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nome", "n_clientes", "custo_ref_hgs", "rotas_ref"])
        for nome, n in itens:
            sol = (DEST / f"{nome}.sol").read_text()
            custo = [float(l.split()[1]) for l in sol.splitlines() if l.startswith("Cost")][0]
            w.writerow([nome, n, custo, sol.count("Route #")])
    print("X_FRESH pronto", flush=True)


if __name__ == "__main__":
    main()
