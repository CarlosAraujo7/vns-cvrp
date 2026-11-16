#!/usr/bin/env python3
"""
run_aos_etapa27.py
==================
Estudo 27 da Direcao A (diario, Entradas 145 e 147 -- pre-registro): O NUMERO DE ROTAS MODERA A VANTAGEM DOS PESOS ADAPTATIVOS? (manipulacao a leiaute e demandas fixos). Nas 51 instancias de avaliacao
do Estudo 21 a vantagem RP - U acompanha o numero de rotas K (Spearman -0,40; parcial dado n -0,36), mas K e o tamanho se confundem e A, B e CMT (K de 7) nao mostram vantagem (Entrada 145). Aqui o
numero de rotas e MANIPULADO: 40 leiautes de SYN_REP (20 de 100 e 20 de 200 clientes) com a capacidade mudada para obter rotas medias de 4, 10 e 25 clientes (K de cerca de n/4 a cerca de n/25), sem
mexer em coordenadas nem demandas (`gerar_variantes_capacidade.py`, conjunto SYN_CAP, 120 instancias). Bracos congelados do Estudo 21 (P21, recozimento, 1.500 n^2): U e RP (rho 0,1, segmento 100).
Seeds 1000 e 1001. Modos: --instancias | --eval (retomavel por chave). Uso (WSL): python3 -u run_aos_etapa27.py --eval --workers 11
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa9 as r9  # noqa: E402
import run_aos_etapa4 as r4  # noqa: E402
import run_aos_etapa23 as r23  # noqa: E402
import run_aos_etapa26 as r26  # noqa: E402

RES = r23.RES
CONJ27 = "SYN_CAP"
N_INST = 120
SEEDS27 = 2
SEMENTES_AVAL = 1000
BRACOS27 = ("U", "RP")


def specs27():
    sp = r26.specs26()
    out = {}
    for a in BRACOS27:
        s = dict(sp[a])
        s["nome"] = f"s27_{a}"
        s["config_id"] = "e27"
        out[a] = s
    return out


def instancias27():
    out = [(CONJ27, n, r9._instancia(CONJ27, n)[0].n_customers) for n in r4.nomes(CONJ27)]
    return sorted(out, key=lambda t: -t[2])                                  # as maiores primeiro (balanceia a carga)


def modo_instancias():
    nomes = r4.nomes(CONJ27)
    tam = []
    for n in nomes:
        inst, cws = r9._instancia(CONJ27, n)
        assert inst.optimum and inst.optimum > 0 and sum(len(r) for r in cws) == inst.n_customers, n
        ok, motivo = inst.is_feasible(cws)
        assert ok, (n, motivo)
        tam.append(inst.n_customers)
    assert len(nomes) == N_INST, len(nomes)
    print(f"{CONJ27}: {len(nomes)} instancias; clientes de {min(tam)} a {max(tam)}", flush=True)
    print("INSTANCIAS: OK", flush=True)


def modo_eval(workers):
    sp = specs27()
    tar = [(s, 1, c, n, SEMENTES_AVAL + sd) for c, n, _nc in instancias27() for s in sp.values() for sd in range(SEEDS27)]
    print(f"eval Estudo 27 (P21): {len(tar)} execucoes ({', '.join(sp)}) em {len(set(t[3] for t in tar))} instancias", flush=True)
    with mp.Pool(workers) as pool:
        r23.rodar23(tar, RES / "etapa27_eval.csv", pool)
    print("eval Estudo 27 concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--instancias", action="store_true")
    g.add_argument("--eval", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.instancias:
        return modo_instancias()
    return modo_eval(a.workers)


if __name__ == "__main__":
    main()
