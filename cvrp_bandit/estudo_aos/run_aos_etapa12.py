#!/usr/bin/env python3
"""
run_aos_etapa12.py
==================
Runner do Estudo 12 da Direcao A: CONTROLE DE MISTURA ESTATICA NO ALNS. O Estudo 9 mostrou que politicas adaptativas
(RP, PM, EXP3, HG-UCB) superam a selecao uniforme no ALNS; resta saber se o ganho vem da ADAPTACAO ou apenas de pesar melhor
os operadores (o regret-2 e o melhor reparo), o que um vetor FIXO de probabilidades ajustado offline tambem capturaria.
PROTOCOLO PRE-REGISTRADO no diario (Entrada 109); leia la antes de mexer.

ALNS, orcamento, temperatura, contabilidade (n + R por iteracao, Entrada 107), instancias e seeds: exatamente os do Estudo 9.
Politica estatica: destruicao e reparo sorteados com probabilidades FIXAS p_d (4) e p_r (3) = softmax(theta[:4]), softmax(theta[4:]).
  ajuste   : SYN_TUNE (24 instancias), seeds 3000-3002, busca (1+4)-ES de theta (R^7; 8 geracoes x 4 filhos = 33 avaliacoes,
             sigma0=0,5, mesma regra de sigma do Estudo 4, sementes 20261112+g); a de menor gap medio vira `s12_static`
  avaliacao: X (21, seeds 1000-1009) e SYN_VAL (30, seeds 1000-1004); bracos novos s12_static e s12_reg2 (reparo SEMPRE regret-2,
             destruicao uniforme); as referencias U, RP, PM, EXP3, HG-UCB, UCB1 vem de etapa9_eval.csv (mesmo codigo/dados)
Modos: --tuning-static, --selecionar, --eval. Resumivel por chave. Roda no WSL (venv de direcao1_local).
"""
import argparse
import csv
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
from run_aos_study import _instancia  # noqa: E402
from run_aos_etapa3 import _proxy  # noqa: E402
import run_aos_etapa4 as r4  # noqa: E402
import run_aos_etapa9 as r9  # noqa: E402
from alns import DESTRUIR, REPARAR  # noqa: E402
from alns_loop import rodar_alns  # noqa: E402
from aos_vnd import EstaticaPolitica  # noqa: E402

RES = ROOT / "results" / "aos_study"
SEEDS_TUNE12 = [3000, 3001, 3002]
ES_GERACOES12, ES_FILHOS12, ES_SIGMA0 = 8, 4, 0.5
ES_BASE_SEED = 20261112


def spec12(nome, theta=None, pesos_d=None, pesos_r=None, config_id="e12"):
    return {"nome": nome, "theta": theta, "pesos_d": pesos_d, "pesos_r": pesos_r, "config_id": config_id,
            "kappa": r9.KAPPA9, "w0": r9.W09, "tf": r9.TF9, "params": {}}


def pesos_de_theta(theta):
    w = np.exp(np.array(theta, float))
    pd_, pr = w[:4] / w[:4].sum(), w[4:] / w[4:].sum()
    return pd_.tolist(), pr.tolist()


def executar12(sp, nivel, set_name, nome_inst, seed):
    inst, cws = _instancia(set_name, nome_inst)
    px = _proxy(set_name, nome_inst)
    px.chamadas = 0
    rng = random.Random(seed)
    pesos_d, pesos_r = sp["pesos_d"], sp["pesos_r"]
    if sp["theta"] is not None:
        pesos_d, pesos_r = pesos_de_theta(sp["theta"])
    pol_d = EstaticaPolitica(pesos_d, rng=random.Random(seed * 31 + 1))
    pol_r = EstaticaPolitica(pesos_r, rng=random.Random(seed * 31 + 2))
    orcamento = int(nivel * sp["kappa"] * inst.n_customers ** 2)
    max_rotas = inst.n_vehicles_ref if inst.vehicles_exact else None
    t0 = time.perf_counter()
    best, info = rodar_alns(px, [list(r) for r in cws], rng, orcamento, pol_d, "score", pol_r, "score",
                            lambda: px.chamadas, w0=sp["w0"], tf_fator=sp["tf"], max_rotas=max_rotas)
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(best)
    viavel, _ = inst.is_feasible(best)
    j = lambda v: "|".join(f"{x:g}" for x in v)  # noqa: E731
    params = {"pesos_d": [round(x, 4) for x in pesos_d], "pesos_r": [round(x, 4) for x in pesos_r]}
    return {"braco": sp["nome"], "config_id": sp["config_id"], "params": json.dumps(params, sort_keys=True),
            "nivel": nivel, "conjunto": set_name, "instancia": inst.name, "n_customers": inst.n_customers,
            "seed": seed, "orcamento": orcamento, "custo": custo, "ref": inst.optimum,
            "gap_pct": 100 * (custo - inst.optimum) / inst.optimum, "viavel": viavel,
            "n_rotas": len([r for r in best if r]), "calls_usadas": px.chamadas, "iteracoes": info["iteracoes"],
            "inviaveis": info["inviaveis"], "uso_d": j(info["uso_d"]), "uso_r": j(info["uso_r"]),
            "score_d": j(info["score_d"]), "score_r": j(info["score_r"]), "calls_d": j(info["calls_d"]),
            "calls_r": j(info["calls_r"]), "tempo_s": dt}


def _worker12(args):
    return executar12(*args)


def rodar12(tarefas, out_csv, pool_mp):
    feitos = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            feitos = {r4.chave(r) for r in csv.DictReader(f)}
    pend = [t for t in tarefas if (t[0]["nome"], t[0]["config_id"], t[1], t[3], t[4]) not in feitos]
    if not pend:
        return 0
    novo = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=r9.CAMPOS9)
        if novo:
            w.writeheader()
        t0 = time.perf_counter()
        feitas = 0
        for linha in pool_mp.imap_unordered(_worker12, pend, chunksize=1):
            w.writerow(linha)
            feitas += 1
            if feitas % 25 == 0:
                f_out.flush()
                print(f"    {feitas}/{len(pend)}  {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    return len(pend)


def tarefas_tune(sp):
    return [(sp, 1, "SYN_TUNE", n, s) for n in sorted(r4.nomes("SYN_TUNE"), reverse=True) for s in SEEDS_TUNE12]


def modo_tuning(workers):
    out = RES / "etapa12_tuning_static.csv"
    estado_p = RES / "etapa12_tuning_static_estado.json"
    k = 7
    n_ref = len(r4.nomes("SYN_TUNE")) * len(SEEDS_TUNE12)
    estado = (json.loads(estado_p.read_text()) if estado_p.exists()
              else {"geracao": 0, "theta": [0.0] * k, "sigma": ES_SIGMA0, "melhor": None})
    with mp.Pool(workers) as pool:
        def avaliar(theta, cid):
            sp = spec12("tune_static_A", theta=list(map(float, theta)), config_id=cid)
            rodar12(tarefas_tune(sp), out, pool)
            m, n = r4.media_gap(out, "tune_static_A", cid)
            assert n == n_ref, (cid, n, n_ref)
            return m

        if estado["melhor"] is None:
            estado["melhor"] = avaliar(estado["theta"], "gen00")
            estado_p.write_text(json.dumps(estado))
            print(f"gen 0: gap {estado['melhor']:.4f}", flush=True)
        while estado["geracao"] < ES_GERACOES12:
            g = estado["geracao"] + 1
            rng = np.random.default_rng(ES_BASE_SEED + g)
            filhos = [np.array(estado["theta"]) + estado["sigma"] * rng.normal(size=k) for _ in range(ES_FILHOS12)]
            gaps = [avaliar(th, f"gen{g:02d}c{i}") for i, th in enumerate(filhos)]
            i = int(np.argmin(gaps))
            if gaps[i] < estado["melhor"]:
                estado["theta"], estado["melhor"] = filhos[i].tolist(), gaps[i]
            else:
                estado["sigma"] *= 0.9
            estado["geracao"] = g
            estado_p.write_text(json.dumps(estado))
            print(f"gen {g}: melhor filho {min(gaps):.4f}  atual {estado['melhor']:.4f}  sigma {estado['sigma']:.3f}", flush=True)
    print("busca estatica (ALNS) concluida", flush=True)


def modo_selecionar():
    import pandas as pd
    st = pd.read_csv(RES / "etapa12_tuning_static.csv")
    st = st[st.viavel].groupby("config_id").agg(gap=("gap_pct", "mean"), params=("params", "first")).reset_index().sort_values("gap")
    top = st.iloc[0]
    par = json.loads(top.params)
    u0 = st[st.config_id == "gen00"].gap.iloc[0]
    (RES / "etapa12_tuning_escolhida.json").write_text(json.dumps(
        {"pesos_d": par["pesos_d"], "pesos_r": par["pesos_r"], "gap_tune": float(top.gap), "gap_pesos_iguais": float(u0),
         "config_id": top.config_id, "n_configs": int(len(st))}, indent=1))
    print(f"static_A: {top.config_id} gap_tune={top.gap:.4f} (pesos iguais {u0:.4f}); destruir {par['pesos_d']} reparar {par['pesos_r']}")


def modo_eval(workers):
    from instance import load_all
    esc = json.loads((RES / "etapa12_tuning_escolhida.json").read_text())
    sp = [spec12("s12_static", pesos_d=esc["pesos_d"], pesos_r=esc["pesos_r"]),
          spec12("s12_reg2", pesos_d=[0.25] * 4, pesos_r=[0.0, 1.0, 0.0])]
    insts = []
    for conj in ("X", "SYN_VAL"):
        insts += [(conj, i.name, i.n_customers) for i in load_all(ROOT / "instances" / conj)]
    insts.sort(key=lambda t: -t[2])
    tarefas = [(s, 1, c, n, 1000 + sd) for (c, n, nc) in insts for s in sp for sd in range(10 if c == "X" else 5)]
    print(f"eval: {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar12(tarefas, RES / "etapa12_eval.csv", pool)
    print("eval concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("tuning-static", "selecionar", "eval"):
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.tuning_static:
        modo_tuning(a.workers)
    elif a.selecionar:
        modo_selecionar()
    else:
        modo_eval(a.workers)


if __name__ == "__main__":
    main()
