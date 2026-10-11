#!/usr/bin/env python3
"""
gerar_instancias_sinteticas.py
==============================
Gera os conjuntos SINTETICOS da Etapa 4 (diario, Entrada 83 -- pre-registro),
disjuntos das 74 instancias da CVRPLIB, com o gerador estilo Uchoa JA
VALIDADO byte-a-byte contra o script oficial (direcao1_local/
uchoa_gerador_core.py, achado #32).

  SYN_TUNE: 24 instancias (8 de n=50, 8 de n=100, 8 de n=150), semente 20261008
  SYN_VAL : 30 instancias (10 de n=100, 10 de n=200, 10 de n=300), semente 20261009

Cada instancia recebe um custo de REFERENCIA do HGS-CVRP (melhor de 2
execucoes, seeds 1 e 2, t = 0,2 s por cliente, distancias arredondadas), gravado
como `.sol` no formato CVRPLIB -- assim `instance.load_instance` funciona sem
alteracao. Nomes comecam com "X" => o numero de rotas e um limite inferior (K do
HGS), como no conjunto X real. O custo de referencia NAO e otimo provado
(declarado no manuscrito); os gaps sao relativos a ele.

Roda no WSL (binario do HGS e Linux):
    python3 -u gerar_instancias_sinteticas.py
"""
import csv
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "direcao1_local"))
from uchoa_gerador_core import gerar_instancia_uchoa, amostrar_combinacao  # noqa: E402

HGS = ROOT / "external" / "HGS-CVRP" / "hgs"
CONJUNTOS = {
    "SYN_TUNE": dict(tag="t", semente=20261008, tamanhos=[50] * 8 + [100] * 8 + [150] * 8),
    "SYN_VAL": dict(tag="v", semente=20261009, tamanhos=[100] * 10 + [200] * 10 + [300] * 10),
    # Estudo 10 (instancias GRANDES; diario, Entrada 100): ajuste e validacao em tamanhos maiores que os dos Estudos 4-7
    "SYN_TUNE_L": dict(tag="tl", semente=20261011, tamanhos=[200] * 4 + [300] * 4 + [400] * 4),
    "SYN_VAL_L": dict(tag="vl", semente=20261012, tamanhos=[200] * 5 + [300] * 5 + [400] * 5 + [500] * 5),
}


def escrever_vrp(path, nome, comentario, depot, clientes, demandas, cap):
    n = len(clientes)
    linhas = [f"NAME : {nome}", f'COMMENT : "{comentario}"', "TYPE : CVRP", f"DIMENSION : {n + 1}",
              "EDGE_WEIGHT_TYPE : EUC_2D", f"CAPACITY : {cap}", "NODE_COORD_SECTION",
              f"1 {depot[0]} {depot[1]}"]
    linhas += [f"{i + 2} {x} {y}" for i, (x, y) in enumerate(clientes)]
    linhas += ["DEMAND_SECTION", "1 0"] + [f"{i + 2} {d}" for i, d in enumerate(demandas)]
    linhas += ["DEPOT_SECTION", " 1", " -1", "EOF"]
    path.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def custo_do_sol(sol_path):
    for linha in sol_path.read_text().splitlines():
        if linha.startswith("Cost"):
            return float(linha.split()[1])
    raise RuntimeError(f"sem linha Cost em {sol_path}")


def hgs_melhor(vrp_path, n):
    melhor = None
    with tempfile.TemporaryDirectory(prefix="hgsref_") as tmp:
        for seed in (1, 2):
            out = Path(tmp) / f"s{seed}.sol"
            subprocess.run([str(HGS), str(vrp_path), str(out), "-t", f"{0.2 * n:.1f}", "-seed", str(seed),
                            "-round", "1", "-log", "0"], capture_output=True, text=True)
            if out.exists():
                c = custo_do_sol(out)
                if melhor is None or c < melhor[0]:
                    melhor = (c, out.read_text())
    if melhor is None:
        raise RuntimeError(f"HGS nao produziu solucao para {vrp_path}")
    return melhor


def main():
    # Uso: python3 gerar_instancias_sinteticas.py [CONJUNTO ...]   (sem argumentos: todos). Um conjunto que ja tem
    # metadata.csv NAO e regenerado (os conjuntos dos Estudos 4-7 sao fixos; o HGS de referencia depende do tempo).
    escolhidos = sys.argv[1:] or list(CONJUNTOS)
    for nome_conj, cfg in CONJUNTOS.items():
        if nome_conj not in escolhidos:
            continue
        destino = ROOT / "instances" / nome_conj
        if (destino / "metadata.csv").exists():
            print(f"{nome_conj}: ja existe (metadata.csv); nao regenerado", flush=True)
            continue
        destino.mkdir(parents=True, exist_ok=True)
        rng = random.Random(cfg["semente"])
        itens = []
        for idx, n in enumerate(cfg["tamanhos"], start=1):
            while True:
                dp, cp, dt, rs = amostrar_combinacao(rng)
                try:
                    depot, clientes, demandas, cap = gerar_instancia_uchoa(n, dp, cp, dt, rs, rng=rng)
                    break
                except ValueError as e:
                    if "clusters" not in str(e):
                        raise
            nome = f"X-syn{cfg['tag']}{idx:02d}-n{n + 1}"
            vrp = destino / f"{nome}.vrp"
            escrever_vrp(vrp, nome, f"Synthetic Uchoa-style, validated generator, seed {cfg['semente']}, "
                         f"dp={dp} cp={cp} dt={dt} rs={rs}", depot, clientes, demandas, cap)
            itens.append((nome, n, vrp, (dp, cp, dt, rs), cap))
        print(f"{nome_conj}: {len(itens)} instancias escritas; rodando HGS de referencia...", flush=True)

        def ref(item):
            nome, n, vrp, comb, cap = item
            custo, texto = hgs_melhor(vrp, n)
            vrp.with_suffix(".sol").write_text(texto)
            return nome, n, comb, cap, custo, texto.count("Route #")

        with ThreadPoolExecutor(max_workers=6) as ex:
            resultados = list(ex.map(ref, itens))
        with open(destino / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["nome", "n_clientes", "depot_pos", "cust_pos", "demand_type", "avg_route_size",
                        "capacidade", "custo_ref_hgs", "rotas_ref"])
            for nome, n, comb, cap, custo, k in resultados:
                w.writerow([nome, n, *comb, cap, custo, k])
                print(f"  {nome:18s} n={n:3d} comb={comb} cap={cap:5d} ref={custo:.0f} K={k}", flush=True)


if __name__ == "__main__":
    main()
