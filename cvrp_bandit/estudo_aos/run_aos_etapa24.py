#!/usr/bin/env python3
"""
run_aos_etapa24.py
==================
Estudo 24 da Direcao A (diario, Entrada 142 -- pre-registro): QUAIS POLITICAS DE SELECAO APROVEITAM O CONJUNTO GRANDE? O Estudo 21 mostrou, no ALNS do CVRP com 21 operadores (P21), os pesos
adaptativos de Ropke e Pisinger (RP) superando a uniforme e as misturas ajustadas; o Estudo 23, que a vantagem sobrevive a pontuacao fiel (RPh). Aqui as outras familias de politicas do Estudo 9 rodam em
P21, SEM ajuste, com a pontuacao FIEL (novidade = True) e os parametros do Estudo 9 (UCB1: C 2,0; EXP3: gamma 0,1; probability matching: gamma 0,05 e p_min = 0,05 x 7/21 = 0,0167, que mantem a massa
minima de 0,35 do conjunto de 7 operadores; amostragem de Thompson e UCB de eficiencia, com os padroes). Avaliacao: as 51 instancias e seeds do Estudo 21 (360 execucoes por braco).
Uso (WSL): python3 -u run_aos_etapa24.py --eval --workers 11     (retomavel por chave)
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa21 as r21  # noqa: E402
import run_aos_etapa23 as r23  # noqa: E402

RES = r21.RES
ARMS24 = {"PM": ("pm", {"gamma": 0.05, "p_min": round(0.05 * 7 / 21, 4)}), "EXP3": ("exp3", {}), "UCB1": ("ucb1", {"C": 2.0}), "TS": ("ts", {}), "EFF": ("eff", {})}


def specs():
    return {a: r23.com_id(r23.spec23(f"s24_P21_{a}", pol, params=params, novidade=True), a) for a, (pol, params) in ARMS24.items()}


def modo_eval(workers):
    sp = specs()
    tar = [(s, 1, c, n, r23.SEMENTES_AVAL + sd) for c, n, ns, _nc in r21.instancias_aval() for s in sp.values() for sd in range(ns)]
    print(f"eval Estudo 24 (P21): {len(tar)} execucoes ({', '.join(sp)})", flush=True)
    with mp.Pool(workers) as pool:
        r23.rodar23(tar, RES / "etapa24_eval_P21.csv", pool)
    print("eval Estudo 24 concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", required=True)
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    return modo_eval(a.workers)


if __name__ == "__main__":
    main()
