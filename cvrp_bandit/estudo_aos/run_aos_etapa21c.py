#!/usr/bin/env python3
"""
run_aos_etapa21c.py
===================
DIAGNOSTICO EXPLORATORIO do Estudo 21 (diario, Entrada 140, declarado ANTES de rodar): o braco S_RP, a mistura ESTATICA de P21 cujos pesos sao o uso medio de operadores dos pesos adaptativos
(RP, hiperparametros padrao) nas execucoes de VALIDACAO nas instancias de AJUSTE (seeds 3003-3005; nenhum dado de avaliacao entra nos pesos). Responde: a vantagem de RP sobre as misturas
ajustadas vem do QUE ele seleciona em media (uma alocacao boa, que a busca offline nao achou) ou de COMO seleciona (a dinamica dentro da execucao)? Se S_RP ~ RP, a adaptacao funciona como
ajuste automatico da alocacao; se S_RP e pior, a dinamica importa. Avaliacao: as 51 instancias e seeds do Estudo 21 (360 execucoes); arquivo etapa21c_eval_P21.csv. Modulo separado.
Uso (WSL): python3 -u run_aos_etapa21c.py --eval --workers 11
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa21 as r21  # noqa: E402

RES = r21.RES
POOL = "P21"


def pesos_srp():
    import pandas as pd
    v = pd.read_csv(RES / f"etapa21_validacao_{POOL}.csv")
    v = v[(v.braco == f"s21_{POOL}_RP") & v.viavel]
    assert len(v) == 72, len(v)
    out = []
    for col in ("uso_d", "uso_r"):
        m = pd.DataFrame(v[col].str.split("|").apply(lambda s: [float(x) for x in s]).tolist()).mean()
        out.append([round(float(x), 4) for x in m / m.sum()])
    return out


def modo_eval(workers):
    pd_, pr = pesos_srp()
    (RES / "etapa21c_srp_pesos.json").write_text(json.dumps({"pesos_d": pd_, "pesos_r": pr}, indent=1))
    nome = f"s21_{POOL}_SRP"
    s = r21.spec21(POOL, nome, "static", pesos_d=pd_, pesos_r=pr)
    tar = [(s, 1, c, n, r21.SEMENTES_AVAL + sd) for c, n, ns, _nc in r21.instancias_aval() for sd in range(ns)]
    print(f"eval diagnostico S_RP ({POOL}): {len(tar)} execucoes; pesos_d {pd_} pesos_r {pr}", flush=True)
    with mp.Pool(workers) as mpool:
        r21.rodar21(tar, RES / f"etapa21c_eval_{POOL}.csv", mpool)
    print("eval S_RP concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", required=True)
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    return modo_eval(a.workers)


if __name__ == "__main__":
    main()
