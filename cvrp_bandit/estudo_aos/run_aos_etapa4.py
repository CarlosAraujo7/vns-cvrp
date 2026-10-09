#!/usr/bin/env python3
"""
run_aos_etapa4.py
=================
Runner da Etapa 4 da Direcao A (AOS ajustadas pelo mesmo protocolo, mistura
estatica ajustada offline, validacao fora da amostra). PROTOCOLO
PRE-REGISTRADO em diario_pesquisa.md, Entrada 83 -- leia la antes de mexer.

Estrutura fixa: P10, termino R5, aceitacao "best", shake com os 5 operadores
originais, orcamento em avaliacoes de distancia E1=600 n^2, E2=5 E1.

Modos (cada um e resumivel por chave):
  --tuning-grid     grade de UCB1/PM/HG-UCB + referencia U em SYN_TUNE
  --tuning-static   busca (1+4)-ES da mistura estatica em SYN_TUNE
  --selecionar      escolhe as configuracoes e grava etapa4_tuning_escolhidas.json
  --eval74          bracos ajustados + U instrumentado nas 74 (E1 e E2)
  --val             bracos nas 30 instancias sinteticas de validacao (E1)
  --pilot           verificacao rapida da infraestrutura (nao entra em inferencia)

Roda no WSL (venv de direcao1_local):  python3 -u run_aos_etapa4.py --MODO
"""
import argparse
import csv
import itertools
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_aos_study import _instancia, todas_instancias  # noqa: E402
from run_aos_etapa3 import (ARMS as ARMS3, POOL_P10, POOL_P5, FOC_ORDEM, PACIENCIA,  # noqa: E402
                            KAPPA, _proxy)
from neighborhoods import NEIGHBORHOODS  # noqa: E402
from instance import load_all  # noqa: E402
from aos_vnd import vnd_generico, construir_politica  # noqa: E402

RES = ROOT / "results" / "aos_study"
SEEDS_TUNE = [3000 + i for i in range(5)]
SEEDS_TEST = [1000 + i for i in range(10)]
SEEDS_VAL = [1000 + i for i in range(5)]
GRADE_UCB1 = [{"C": c} for c in (0.1, 0.3, 1.0, 2.0, 4.0)]
GRADE_PM = [{"gamma": g, "p_min": p} for g in (0.05, 0.1, 0.3) for p in (0.02, 0.05, 0.1)]
GRADE_HG = [{"alpha": a, "C": c} for a in (0.1, 0.5, 2.0) for c in (1.0, 2.0, 4.0)]
PADRAO = {"ucb1": {"C": 2.0}, "pm": {"gamma": 0.1, "p_min": 0.05}, "hgucb": {"alpha": 0.5, "C": 2.0}}
ES_GERACOES, ES_FILHOS, ES_SIGMA0 = 15, 4, 0.5

CAMPOS = ["braco", "config_id", "params", "nivel", "conjunto", "instancia", "n_customers", "seed",
          "orcamento", "custo", "ref", "gap_pct", "viavel", "n_rotas", "calls_usadas", "max_calls_app",
          "apps", "shakes", "paciencia_fim", "uso_operadores", "calls_por_operador",
          "suc_por_operador", "ganho_por_operador", "tempo_s"]


def spec(nome, pol, termino="R", custo=True, params=None, config_id="padrao", pool="P10"):
    return {"nome": nome, "pol": pol, "termino": termino, "custo": custo, "params": params or {},
            "config_id": config_id, "pool": pool}


def spec_de_arm3(nome):
    pool, pol, custo, term = ARMS3[nome]
    return spec(nome, pol, term, custo, None, "etapa3", pool)


def executar4(sp, nivel, set_name, nome_inst, seed):
    inst, cws = _instancia(set_name, nome_inst)
    proxy = _proxy(set_name, nome_inst)
    proxy.chamadas = 0
    if isinstance(sp["pool"], list):          # Etapa 5: subconjunto do pool de 10 por indices
        pool = [POOL_P10[i] for i in sp["pool"]]
    else:
        pool = POOL_P10 if sp["pool"] == "P10" else POOL_P5
    k = len(pool)
    uso, calls_op, maximo = [0] * k, [0] * k, [0]
    suc, gan = [0] * k, [0.0] * k

    def envolver(idx, bf):
        def f(_inst, routes):
            c0 = proxy.chamadas
            uso[idx] += 1
            res = bf(proxy, routes)
            d = proxy.chamadas - c0
            calls_op[idx] += d
            if d > maximo[0]:
                maximo[0] = d
            return res
        return f

    def on_aplicacao(j, melhora, ganho):
        if melhora:
            suc[j] += 1
            gan[j] += ganho

    nbhs = [(nm, envolver(i, bf), rf) for i, (nm, bf, rf) in enumerate(pool)]
    rng = random.Random(seed)
    politica, modo = construir_politica(sp["pol"], k, rng, custo=sp["custo"],
                                        ordem=FOC_ORDEM if sp["pol"] == "FOC" else None,
                                        params=sp["params"])
    orcamento = nivel * KAPPA * inst.n_customers ** 2
    t0 = time.perf_counter()
    rotas, info = vnd_generico(inst, [list(r) for r in cws], rng, orcamento, nbhs, politica, "best",
                               sp["termino"], modo_recompensa=modo, neighborhoods_shake=NEIGHBORHOODS,
                               medida=lambda: proxy.chamadas, paciencia=PACIENCIA,
                               on_aplicacao=on_aplicacao)
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(rotas)
    viavel, _ = inst.is_feasible(rotas)
    return {
        "braco": sp["nome"], "config_id": sp["config_id"], "params": json.dumps(sp["params"], sort_keys=True),
        "nivel": nivel, "conjunto": set_name, "instancia": inst.name, "n_customers": inst.n_customers,
        "seed": seed, "orcamento": orcamento, "custo": custo, "ref": inst.optimum,
        "gap_pct": 100 * (custo - inst.optimum) / inst.optimum, "viavel": viavel,
        "n_rotas": len([r for r in rotas if r]), "calls_usadas": proxy.chamadas,
        "max_calls_app": maximo[0], "apps": info["usado"], "shakes": info["shakes"],
        "paciencia_fim": info["paciencia_fim"], "uso_operadores": "|".join(map(str, uso)),
        "calls_por_operador": "|".join(map(str, calls_op)), "suc_por_operador": "|".join(map(str, suc)),
        "ganho_por_operador": "|".join(f"{g:.6f}" for g in gan), "tempo_s": dt,
    }


def _worker(args):
    return executar4(*args)


def chave(r):
    return (r["braco"], r["config_id"], int(r["nivel"]), r["instancia"], int(r["seed"]))


def rodar(tarefas, out_csv, pool_mp):
    feitos = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            feitos = {chave(r) for r in csv.DictReader(f)}
    pend = [t for t in tarefas if (t[0]["nome"], t[0]["config_id"], t[1], t[3], t[4]) not in feitos]
    if not pend:
        return 0
    novo = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "a", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=CAMPOS)
        if novo:
            w.writeheader()
        t0 = time.perf_counter()
        feitas = 0
        for linha in pool_mp.imap_unordered(_worker, pend, chunksize=1):
            w.writerow(linha)
            feitas += 1
            if feitas % 100 == 0:
                f_out.flush()
                print(f"    {feitas}/{len(pend)}  {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    return len(pend)


def nomes(conjunto):
    return sorted(i.name for i in load_all(ROOT / "instances" / conjunto))


def media_gap(out_csv, nome, config_id):
    import pandas as pd
    d = pd.read_csv(out_csv)
    d = d[(d.braco == nome) & (d.config_id == config_id) & d.viavel]
    return d.gap_pct.mean(), len(d)


def tarefas_tune(sp):
    return [(sp, 1, "SYN_TUNE", n, s) for n in nomes("SYN_TUNE") for s in SEEDS_TUNE]


def modo_grid(workers):
    out = RES / "etapa4_tuning_grid.csv"
    specs = [spec("tune_U", "U", custo=False, config_id="U")]
    specs += [spec("tune_ucb1t", "ucb1", params=p, config_id=json.dumps(p, sort_keys=True)) for p in GRADE_UCB1]
    specs += [spec("tune_pmt", "pm", params=p, config_id=json.dumps(p, sort_keys=True)) for p in GRADE_PM]
    specs += [spec("tune_hgucbt", "hgucb", params=p, config_id=json.dumps(p, sort_keys=True)) for p in GRADE_HG]
    tarefas = [t for sp in specs for t in tarefas_tune(sp)]
    print(f"grade: {len(specs)} configuracoes, {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        n = rodar(tarefas, out, pool)
    print(f"grade concluida ({n} execucoes novas)", flush=True)


def modo_static(workers):
    out = RES / "etapa4_tuning_static.csv"
    estado_p = RES / "etapa4_tuning_static_estado.json"
    k = len(POOL_P10)
    if estado_p.exists():
        estado = json.loads(estado_p.read_text())
    else:
        estado = {"geracao": 0, "theta": [0.0] * k, "sigma": ES_SIGMA0, "melhor": None}

    def pesos(theta):
        w = np.exp(np.array(theta))
        return (w / w.sum()).tolist()

    with mp.Pool(workers) as pool:
        def avaliar(theta, cid):
            sp = spec("tune_static", "static", custo=False,
                      params={"pesos": pesos(theta), "theta": list(map(float, theta))}, config_id=cid)
            rodar(tarefas_tune(sp), out, pool)
            m, n = media_gap(out, "tune_static", cid)
            assert n == 24 * 5, (cid, n)
            return m

        if estado["melhor"] is None:
            estado["melhor"] = avaliar(estado["theta"], "gen00")
            estado_p.write_text(json.dumps(estado))
            print(f"gen 0: gap {estado['melhor']:.4f}", flush=True)
        while estado["geracao"] < ES_GERACOES:
            g = estado["geracao"] + 1
            rng = np.random.default_rng(20261010 + g)
            filhos = [np.array(estado["theta"]) + estado["sigma"] * rng.normal(size=k) for _ in range(ES_FILHOS)]
            gaps = [avaliar(th, f"gen{g:02d}c{i}") for i, th in enumerate(filhos)]
            i = int(np.argmin(gaps))
            if gaps[i] < estado["melhor"]:
                estado["theta"], estado["melhor"] = filhos[i].tolist(), gaps[i]
            else:
                estado["sigma"] *= 0.9
            estado["geracao"] = g
            estado_p.write_text(json.dumps(estado))
            print(f"gen {g}: melhor filho {min(gaps):.4f}  atual {estado['melhor']:.4f}  sigma {estado['sigma']:.3f}", flush=True)
    print("busca estatica concluida", flush=True)


def modo_selecionar():
    import pandas as pd
    g = pd.read_csv(RES / "etapa4_tuning_grid.csv")
    g = g[g.viavel]
    escolhidas = {}
    ref_u = g[g.braco == "tune_U"].gap_pct.mean()
    print(f"U (referencia, SYN_TUNE): {ref_u:.4f}")
    for fam, nome in (("ucb1", "tune_ucb1t"), ("pm", "tune_pmt"), ("hgucb", "tune_hgucbt")):
        d = g[g.braco == nome].groupby(["config_id"]).gap_pct.agg(["mean", "count"]).sort_values("mean")
        padrao_id = json.dumps(PADRAO[fam], sort_keys=True)
        melhor_id = d.index[0]
        gap_padrao = d.loc[padrao_id, "mean"]
        if d.loc[melhor_id, "mean"] >= gap_padrao - 1e-12:   # empate ou pior que o padrao -> padrao
            melhor_id = padrao_id
        escolhidas[fam] = {"params": json.loads(melhor_id), "gap_tune": float(d.loc[melhor_id, "mean"]),
                           "gap_padrao": float(gap_padrao), "n_configs": int(len(d))}
        print(f"{fam}: escolhida {melhor_id}  gap_tune={d.loc[melhor_id,'mean']:.4f}  (padrao {gap_padrao:.4f}; "
              f"pior config {d['mean'].max():.4f})")
    st = pd.read_csv(RES / "etapa4_tuning_static.csv")
    st = st[st.viavel].groupby(["config_id", "params"]).gap_pct.mean().reset_index().sort_values("gap_pct")
    top = st.iloc[0]
    par = json.loads(top.params)
    escolhidas["static"] = {"params": par, "gap_tune": float(top.gap_pct), "n_configs": int(len(st))}
    print(f"static: melhor {top.config_id} gap_tune={top.gap_pct:.4f}; pesos="
          f"{[round(x, 3) for x in par['pesos']]}")
    (RES / "etapa4_tuning_escolhidas.json").write_text(json.dumps(escolhidas, indent=1))
    print("gravado etapa4_tuning_escolhidas.json")


def specs_tuned():
    esc = json.loads((RES / "etapa4_tuning_escolhidas.json").read_text())
    return {
        "s4_R5_ucb1t": spec("s4_R5_ucb1t", "ucb1", params=esc["ucb1"]["params"], config_id="escolhida"),
        "s4_R5_pmt": spec("s4_R5_pmt", "pm", params=esc["pm"]["params"], config_id="escolhida"),
        "s4_R5_hgucbt": spec("s4_R5_hgucbt", "hgucb", params=esc["hgucb"]["params"], config_id="escolhida"),
        "s4_R5_static": spec("s4_R5_static", "static", custo=False, params=esc["static"]["params"],
                             config_id="escolhida"),
    }


def modo_eval74(workers):
    sp = specs_tuned()
    insts = todas_instancias()
    tarefas = [(s, nv, c, n, sd) for nv in (1, 5) for (c, n, nc) in sorted(insts, key=lambda t: -t[2])
               for s in sp.values() for sd in SEEDS_TEST]
    instr = spec("s4_R5_U_instr", "U", custo=False, config_id="instr")
    tarefas += [(instr, 1, c, n, sd) for (c, n, nc) in sorted(insts, key=lambda t: -t[2]) for sd in SEEDS_TEST]
    print(f"eval74: {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tarefas, RES / "etapa4_eval74.csv", pool)
    print("eval74 concluido", flush=True)


def modo_val(workers):
    sp = list(specs_tuned().values()) + [spec_de_arm3(n) for n in (
        "s3_P5_LO_U", "s3_LO_U", "s3_R5_U", "s3_R5_FOC", "s3_R5_ucb1c", "s3_R5_pmc", "s3_R5_hgucbc")]
    nm = nomes("SYN_VAL")
    tamanho = {i.name: i.n_customers for i in load_all(ROOT / "instances" / "SYN_VAL")}
    tarefas = [(s, 1, "SYN_VAL", n, sd) for n in sorted(nm, key=lambda x: -tamanho[x]) for s in sp
               for sd in SEEDS_VAL]
    print(f"val: {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tarefas, RES / "etapa4_val.csv", pool)
    print("val concluida", flush=True)


def modo_pilot(workers):
    sp = [spec("pilot_ucb1", "ucb1", params={"C": 0.3}, config_id="C=0.3"),
          spec("pilot_static", "static", custo=False, params={"pesos": [0.1] * 10}, config_id="unif"),
          spec("pilot_U_instr", "U", custo=False, config_id="instr")]
    nm = nomes("SYN_TUNE")[:3]
    tarefas = [(s, 1, "SYN_TUNE", n, 3000) for n in nm for s in sp]
    controle = [(spec("pilot_U_instr", "U", custo=False, config_id="instr"), 1, c, n, sd)
                for (c, n) in (("A", "A-n32-k5"), ("B", "B-n41-k6"), ("CMT", "CMT1")) for sd in (1000, 1001)]
    with mp.Pool(workers) as pool:
        rodar(tarefas + controle, RES / "etapa4_pilot.csv", pool)
    print("pilot concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("tuning-grid", "tuning-static", "selecionar", "eval74", "val", "pilot"):
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.tuning_grid:
        modo_grid(a.workers)
    elif a.tuning_static:
        modo_static(a.workers)
    elif a.selecionar:
        modo_selecionar()
    elif a.eval74:
        modo_eval74(a.workers)
    elif a.val:
        modo_val(a.workers)
    else:
        modo_pilot(a.workers)


if __name__ == "__main__":
    main()
