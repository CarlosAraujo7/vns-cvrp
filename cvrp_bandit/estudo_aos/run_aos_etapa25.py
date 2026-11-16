#!/usr/bin/env python3
"""
run_aos_etapa25.py
==================
Estudo 25 da Direcao A (diario, Entrada 144 -- pre-registro): DESLOCAMENTOS DE IMPLANTACAO EM INSTANCIAS COM FOLGA. No Estudo 22, nas instancias A, B e CMT (30 a 120 clientes), todos os bracos terminam a menos
de 0,07 pp uns dos outros (gap medio de 0,4% em 1.500 n^2 e de 0,9% em 375 n^2): nao ha folga para qualquer politica, e os deslocamentos de regime la nao testam a vantagem do Estudo 21. A vantagem do RP
sobre a uniforme cresce com o gap da uniforme (Spearman -0,41 em P21, exploratorio). Aqui as mesmas configuracoes CONGELADAS de P21 (U, RP padrao, S65 = mistura de maior esforco) sao implantadas nas 51
instancias de avaliacao do Estudo 21 (X e SYN_VAL, 100 a 300 clientes, gap de 1,4%), em dois regimes: acc (record-to-record travel com o desvio delta0 do Estudo 22, 1.500 n^2) e short (recozimento, 375 n^2).
Seeds: 1000-1004 nas X e 1000-1002 nas SYN_VAL (menos que as 10 e 5 do Estudo 21, para o custo; a unidade de analise e a media por instancia). A referencia de cada contraste e o proprio Estudo 21 (mesmas instancias,
1.500 n^2, recozimento).
Uso (WSL): python3 -u run_aos_etapa25.py --eval {acc,short} --workers 11     (retomavel por chave)
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa21 as r21  # noqa: E402
import run_aos_etapa22 as r22  # noqa: E402

RES = r21.RES
SEEDS25 = {"X": 5, "SYN_VAL": 3}
REG25 = {"acc": dict(nivel=1.0, aceit="rrt"), "short": dict(nivel=0.25, aceit="sa")}
BRACOS25 = ("U", "RP", "S65")


def specs(reg):
    fr = r22.congeladas()
    delta0 = r22.delta0_escolhido() if REG25[reg]["aceit"] == "rrt" else 0.0
    out = {}
    for a in BRACOS25:
        pol, params, pd_, pr = fr[a]
        out[a] = r22.spec22(f"s25_{reg}_{a}", pol, params=params, pesos_d=pd_, pesos_r=pr, aceit=REG25[reg]["aceit"], delta0=delta0)
    return out


def modo_eval(reg, workers):
    sp = specs(reg)
    cfg = REG25[reg]
    tar = [(s, cfg["nivel"], c, n, r22.SEMENTES_AVAL + sd) for c, n, _ns, _nc in r21.instancias_aval() for s in sp.values() for sd in range(SEEDS25[c])]
    print(f"eval Estudo 25 {reg}: {len(tar)} execucoes ({', '.join(sp)})", flush=True)
    with mp.Pool(workers) as pool:
        r22.rodar22(tar, RES / f"etapa25_eval_{reg}.csv", pool)
    print(f"eval Estudo 25 {reg} concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", choices=list(REG25), required=True)
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    return modo_eval(a.eval, a.workers)


if __name__ == "__main__":
    main()
