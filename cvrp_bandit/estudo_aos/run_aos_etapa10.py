#!/usr/bin/env python3
"""
run_aos_etapa10.py
==================
Runner do Estudo 10 da Direcao A: o valor da selecao de operadores (mistura estatica aprendida offline e politicas
online sensiveis a custo) CRESCE com o tamanho da instancia? PROTOCOLO PRE-REGISTRADO no diario (Entrada com este
titulo); leia la antes de mexer.

Estrutura fixa (a dos Estudos 4 e 6): P10, termino R5, aceitacao "best", shake dos 5 operadores originais,
orcamento E1 = 600 n^2 em avaliacoes de distancia, solucao inicial fixada em disco.
  ajuste     : SYN_TUNE_L (12 instancias, n=200/300/400), 3 seeds (3000-3002); busca (1+4)-ES da mistura estatica
               (15 geracoes, 61 avaliacoes; o mesmo protocolo do Estudo 4)
  avaliacao  : SYN_VAL_L (20 instancias, n=200/300/400/500), 5 seeds (1000-1004)

Modos (cada um resumivel por chave): --tuning-static, --selecionar, --eval, --pilot
Roda no WSL (venv de direcao1_local).
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa4 as r4  # noqa: E402
from run_aos_etapa3 import POOL_P10  # noqa: E402
from instance import load_all  # noqa: E402

RES = ROOT / "results" / "aos_study"
SEEDS_TUNE_L = [3000, 3001, 3002]
SEEDS_EVAL_L = [1000, 1001, 1002, 1003, 1004]
ES_BASE_SEED = 20261110


def nomes_ord(conj):
    """nomes ordenados do MAIOR para o menor n (as execucoes grandes entram primeiro no pool de processos)"""
    insts = load_all(ROOT / "instances" / conj)
    return [i.name for i in sorted(insts, key=lambda i: -i.n_customers)]


def tarefas_tune_L(sp):
    return [(sp, 1, "SYN_TUNE_L", n, s) for n in nomes_ord("SYN_TUNE_L") for s in SEEDS_TUNE_L]


def modo_tuning(workers):
    out = RES / "etapa10_tuning_static.csv"
    estado_p = RES / "etapa10_tuning_static_estado.json"
    k = len(POOL_P10)
    n_ref = len(nomes_ord("SYN_TUNE_L")) * len(SEEDS_TUNE_L)
    estado = (json.loads(estado_p.read_text()) if estado_p.exists()
              else {"geracao": 0, "theta": [0.0] * k, "sigma": r4.ES_SIGMA0, "melhor": None})

    def pesos(theta):
        w = np.exp(np.array(theta))
        return (w / w.sum()).tolist()

    with mp.Pool(workers) as pool:
        def avaliar(theta, cid):
            sp = r4.spec("tune_static_L", "static", custo=False,
                         params={"pesos": pesos(theta), "theta": list(map(float, theta))}, config_id=cid)
            r4.rodar(tarefas_tune_L(sp), out, pool)
            m, n = r4.media_gap(out, "tune_static_L", cid)
            assert n == n_ref, (cid, n, n_ref)
            return m

        if estado["melhor"] is None:
            estado["melhor"] = avaliar(estado["theta"], "gen00")
            estado_p.write_text(json.dumps(estado))
            print(f"gen 0: gap {estado['melhor']:.4f}", flush=True)
        while estado["geracao"] < r4.ES_GERACOES:
            g = estado["geracao"] + 1
            rng = np.random.default_rng(ES_BASE_SEED + g)
            filhos = [np.array(estado["theta"]) + estado["sigma"] * rng.normal(size=k) for _ in range(r4.ES_FILHOS)]
            gaps = [avaliar(th, f"gen{g:02d}c{i}") for i, th in enumerate(filhos)]
            i = int(np.argmin(gaps))
            if gaps[i] < estado["melhor"]:
                estado["theta"], estado["melhor"] = filhos[i].tolist(), gaps[i]
            else:
                estado["sigma"] *= 0.9
            estado["geracao"] = g
            estado_p.write_text(json.dumps(estado))
            print(f"gen {g}: melhor filho {min(gaps):.4f}  atual {estado['melhor']:.4f}  sigma {estado['sigma']:.3f}",
                  flush=True)
    print("busca estatica (SYN_TUNE_L) concluida", flush=True)


def modo_selecionar():
    import pandas as pd
    st = pd.read_csv(RES / "etapa10_tuning_static.csv")
    st = st[st.viavel].groupby(["config_id", "params"]).gap_pct.mean().reset_index().sort_values("gap_pct")
    top = st.iloc[0]
    par = json.loads(top.params)
    u0 = st[st.config_id == "gen00"].gap_pct.iloc[0]     # pesos iguais
    (RES / "etapa10_tuning_escolhida.json").write_text(json.dumps(
        {"params": par, "gap_tune": float(top.gap_pct), "gap_pesos_iguais": float(u0), "config_id": top.config_id,
         "n_configs": int(len(st))}, indent=1))
    print(f"static_L: melhor {top.config_id} gap_tune_L={top.gap_pct:.4f} (pesos iguais {u0:.4f}); pesos="
          f"{[round(x, 3) for x in par['pesos']]}")


def specs_eval():
    esc4 = json.loads((RES / "etapa4_tuning_escolhidas.json").read_text())
    escL = json.loads((RES / "etapa10_tuning_escolhida.json").read_text())
    return {
        "s10_U": r4.spec("s10_U", "U", custo=False, config_id="e10"),
        "s10_static_S": r4.spec("s10_static_S", "static", custo=False, params=esc4["static"]["params"],
                                config_id="e10"),
        "s10_static_L": r4.spec("s10_static_L", "static", custo=False, params=escL["params"], config_id="e10"),
        "s10_ucb1c": r4.spec("s10_ucb1c", "ucb1", custo=True, params={"C": 2.0}, config_id="e10"),
        "s10_pmc": r4.spec("s10_pmc", "pm", custo=True, params={"gamma": 0.1, "p_min": 0.05}, config_id="e10"),
        "s10_hgucbc": r4.spec("s10_hgucbc", "hgucb", custo=True, params={"alpha": 0.5, "C": 2.0}, config_id="e10"),
        "s10_ts": r4.spec("s10_ts", "ts", custo=False, config_id="e10"),
        "s10_eff": r4.spec("s10_eff", "eff", custo=False, params={"C": 1.0}, config_id="e10"),
    }


def modo_eval(workers):
    sp = specs_eval()
    nm = nomes_ord("SYN_VAL_L")
    tarefas = [(s, 1, "SYN_VAL_L", n, sd) for n in nm for s in sp.values() for sd in SEEDS_EVAL_L]
    print(f"eval: {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        r4.rodar(tarefas, RES / "etapa10_eval.csv", pool)
    print("eval concluido", flush=True)


def modo_pilot(workers):
    """infraestrutura: 2 bracos x 2 instancias de SYN_TUNE_L x 1 seed (nao entra em inferencia)"""
    sp = [r4.spec("pilot10_U", "U", custo=False, config_id="p"),
          r4.spec("pilot10_ucb1c", "ucb1", custo=True, params={"C": 2.0}, config_id="p")]
    nm = nomes_ord("SYN_TUNE_L")[-2:]
    tarefas = [(s, 1, "SYN_TUNE_L", n, 3000) for n in nm for s in sp]
    with mp.Pool(workers) as pool:
        r4.rodar(tarefas, RES / "etapa10_pilot.csv", pool)
    print("pilot concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("tuning-static", "selecionar", "eval", "pilot"):
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.tuning_static:
        modo_tuning(a.workers)
    elif a.selecionar:
        modo_selecionar()
    elif a.eval:
        modo_eval(a.workers)
    else:
        modo_pilot(a.workers)


if __name__ == "__main__":
    main()
