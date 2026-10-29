#!/usr/bin/env python3
"""
run_aos_etapa14.py
==================
Estudo 14 (generalidade: CVRPTW, regime VND determinista). Estrutura dos Estudos 4-13 (R5, aceitacao "best", shake com os 5 operadores de
P5, orcamento em avaliacoes de distancia E1 = 600 n^2, E2 = 5 E1, E3 = 25 E1), pool P9 de src/tw_neighborhoods.py. Instancias: as 56 de
Solomon (CVRPLIB; avaliacao, 10 seeds) e SYN_TUNE_W (24 sinteticas; ajuste). Retomavel (cada modo pula o que ja esta no CSV).

Modos: --pilot-v, --tuning-static-v, --selecionar-v, --eval-v, --eval-pc
Uso (WSL): python3 -u run_aos_etapa14.py --eval-v --workers 11
"""
import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tw_inst import ler_solomon, construcao_inicial  # noqa: E402
from tw_neighborhoods import P9, P5  # noqa: E402
from aos_vnd import vnd_generico, construir_politica  # noqa: E402
from operadores_fracos import dud  # noqa: E402

RES = ROOT / "results" / "aos_study"
INST = ROOT / "instances"
CAMPOS_V = ["braco", "config_id", "params", "nivel", "conjunto", "instancia", "n_customers", "seed",
            "orcamento", "custo", "ref", "gap_pct", "viavel", "n_rotas", "calls_usadas", "max_calls_app",
            "apps", "shakes", "paciencia_fim", "uso_operadores", "calls_por_operador",
            "suc_por_operador", "ganho_por_operador", "tempo_s"]

KAPPA_V = 600
PACIENCIA = 5
NIVEIS_V = (1, 5, 25)
SEMENTES_AVAL = 1000
SEEDS_PILOTO = [3000, 3001, 3002]
SEEDS_TUNE_V = [3000, 3001, 3002, 3003, 3004]
ES_GERACOES, ES_FILHOS, ES_SIGMA0, ES_SEED_V = 15, 4, 0.5, 20261114

# Ordens fixas (indices de P9 = Move(1), Swap(1,1), Relocate(1,0), Relocate(2,0), 2-opt*, Or-opt(2-3), Swap-intra, CROSS(<=3), CROSS(4-5))
ORDEM_IDX = list(range(9))
ORDEM_CUSTO = None        # preenchida depois do piloto (custo medio por aplicacao crescente), ver ordem_custo_fixada.json
ORDEM_REV = None

POOLS = {"P9": (P9, []), "P5": (P5, []),
         "P9+D4x2": (P9, [2.0] * 4), "P9+D4x4": (P9, [4.0] * 4), "P9+D4x8": (P9, [8.0] * 4), "P9+D4x16": (P9, [16.0] * 4), "P9+D4x32": (P9, [32.0] * 4), "P9+D4x64": (P9, [64.0] * 4)}


def _ordens():
    p = RES / "etapa14_ordem_custo_fixada.json"
    if p.exists():
        o = json.loads(p.read_text())["ordem"]
        return o, o[::-1]
    return None, None


# ----------------------------------------------------------------------------------------------- instancias
@lru_cache(maxsize=None)
def _refs(conj):
    return {r["instancia"]: float(r["ref"]) for r in csv.DictReader(open(INST / conj / "ref.csv"))}


def nomes(conj):
    return sorted(_refs(conj))


@lru_cache(maxsize=None)
def carregar(conj, nome):
    inst = ler_solomon(INST / conj / f"{nome}.txt", ref=_refs(conj)[nome])
    return inst, construcao_inicial(inst)


# ----------------------------------------------------------------------------------------------- execucao
def spec_v(nome, pol, custo=False, params=None, pool="P9", config_id="e14", ordem=None):
    return {"nome": nome, "pol": pol, "custo": custo, "params": params or {}, "config_id": config_id, "pool": pool,
            "regime": "v", "ordem": ordem}


def executar_v(sp, nivel, conj, nome_inst, seed):
    inst, x0 = carregar(conj, nome_inst)
    inst.chamadas = 0
    base, duds = POOLS[sp["pool"]]
    pool = list(base) + [dud(f) for f in duds]
    k = len(pool)
    uso, calls_op, maximo = [0] * k, [0] * k, [0]
    suc, gan = [0] * k, [0.0] * k

    def envolver(idx, bf):
        def f(_inst, routes):
            c0 = inst.chamadas
            uso[idx] += 1
            res = bf(inst, routes)
            d = inst.chamadas - c0
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
    politica, modo = construir_politica(sp["pol"], k, rng, custo=sp["custo"], params=sp["params"], ordem=sp.get("ordem"))
    orcamento = nivel * KAPPA_V * inst.n_customers ** 2
    t0 = time.perf_counter()
    rotas, info = vnd_generico(inst, [list(r) for r in x0], rng, orcamento, nbhs, politica, "best", "R", modo_recompensa=modo,
                               neighborhoods_shake=P5, medida=lambda: inst.chamadas, paciencia=PACIENCIA,
                               on_aplicacao=on_aplicacao)
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(rotas)
    viavel, _ = inst.is_feasible(rotas)
    return {
        "braco": sp["nome"], "config_id": sp["config_id"], "params": json.dumps(sp["params"], sort_keys=True),
        "nivel": nivel, "conjunto": conj, "instancia": inst.nome, "n_customers": inst.n_customers,
        "seed": seed, "orcamento": orcamento, "custo": custo, "ref": inst.ref,
        "gap_pct": 100 * (custo - inst.ref) / inst.ref, "viavel": viavel,
        "n_rotas": len([r for r in rotas if r]), "calls_usadas": inst.chamadas, "max_calls_app": maximo[0],
        "apps": info["usado"], "shakes": info["shakes"], "paciencia_fim": info["paciencia_fim"],
        "uso_operadores": "|".join(map(str, uso)), "calls_por_operador": "|".join(map(str, calls_op)),
        "suc_por_operador": "|".join(map(str, suc)), "ganho_por_operador": "|".join(f"{g:.6f}" for g in gan),
        "tempo_s": dt,
    }


def _worker(args):
    sp, nivel, conj, nome, seed = args
    return executar_v(sp, nivel, conj, nome, seed)


def chave(r):
    return (r["braco"], r["config_id"], float(r["nivel"]), r["instancia"], int(r["seed"]))


def rodar(tarefas, out_csv, pool_mp, campos, cada=25):
    feitos = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            feitos = {chave(r) for r in csv.DictReader(f)}
    pend = [t for t in tarefas if (t[0]["nome"], t[0]["config_id"], float(t[1]), t[3], t[4]) not in feitos]
    if not pend:
        return 0
    novo = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "a", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=campos)
        if novo:
            w.writeheader()
        t0 = time.perf_counter()
        feitas = 0
        for linha in pool_mp.imap_unordered(_worker, pend, chunksize=1):
            w.writerow(linha)
            feitas += 1
            if feitas % cada == 0:
                f_out.flush()
                print(f"    {feitas}/{len(pend)}  {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    return len(pend)


def media_gap(out_csv, nome, config_id):
    import pandas as pd
    d = pd.read_csv(out_csv)
    d = d[(d.braco == nome) & (d.config_id == config_id) & d.viavel]
    return d.gap_pct.mean(), len(d)


def _softmax(theta):
    w = np.exp(np.array(theta, float))
    return w / w.sum()


def _es(out, estado_p, k, avaliar_fn, geracoes, base_seed, rotulo):
    estado = (json.loads(estado_p.read_text()) if estado_p.exists()
              else {"geracao": 0, "theta": [0.0] * k, "sigma": ES_SIGMA0, "melhor": None})
    if estado["melhor"] is None:
        estado["melhor"] = avaliar_fn(estado["theta"], "gen00")
        estado_p.write_text(json.dumps(estado))
        print(f"gen 0: gap {estado['melhor']:.4f}", flush=True)
    while estado["geracao"] < geracoes:
        g = estado["geracao"] + 1
        rng = np.random.default_rng(base_seed + g)
        filhos = [np.array(estado["theta"]) + estado["sigma"] * rng.normal(size=k) for _ in range(ES_FILHOS)]
        gaps = [avaliar_fn(th, f"gen{g:02d}c{i}") for i, th in enumerate(filhos)]
        i = int(np.argmin(gaps))
        if gaps[i] < estado["melhor"]:
            estado["theta"], estado["melhor"] = filhos[i].tolist(), gaps[i]
        else:
            estado["sigma"] *= 0.9
        estado["geracao"] = g
        estado_p.write_text(json.dumps(estado))
        print(f"gen {g}: melhor filho {min(gaps):.4f}  atual {estado['melhor']:.4f}  sigma {estado['sigma']:.3f}", flush=True)
    print(f"busca estatica ({rotulo}) concluida", flush=True)


# ----------------------------------------------------------------------------------------------- modos
def piloto_v(workers):
    """Calibracao em SYN_TUNE_W (NAO e dado do estudo): gap em E1, E2, E3 de U(P9) e U(P5), custo medio por aplicacao de cada operador
    (para fixar a ordem de custo crescente e o tamanho dos operadores inuteis), efeito dos duds e algumas politicas."""
    out = RES / "etapa14_piloto_v.csv"
    sp = [spec_v("p_U", "U"), spec_v("p_U_P5", "U", pool="P5"), spec_v("p_ucb1c", "ucb1", custo=True, params={"C": 2.0}),
          spec_v("p_ts", "ts"), spec_v("p_eff", "eff", params={"C": 1.0}),
          spec_v("p_FOC_idx", "FOC", ordem=ORDEM_IDX)]
    sp += [spec_v(f"p_U_{p}", "U", pool=p) for p in ("P9+D4x2", "P9+D4x4", "P9+D4x8", "P9+D4x16", "P9+D4x32", "P9+D4x64")]
    tarefas = [(s, nv, "SYN_TUNE_W", n, sd) for nv in NIVEIS_V for n in sorted(nomes("SYN_TUNE_W"), reverse=True)
               for s in sp for sd in SEEDS_PILOTO]
    print(f"piloto VND: {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tarefas, out, pool, CAMPOS_V, cada=100)
    print("piloto VND concluido", flush=True)


def modo_tuning_static_v(workers):
    out, estado_p = RES / "etapa14_tuning_static_v.csv", RES / "etapa14_tuning_static_v_estado.json"
    n_ref = len(nomes("SYN_TUNE_W")) * len(SEEDS_TUNE_V)
    with mp.Pool(workers) as pool:
        def avaliar(theta, cid):
            sp = spec_v("tune_static_W", "static", params={"pesos": _softmax(theta).tolist(), "theta": list(map(float, theta))},
                        config_id=cid)
            tar = [(sp, 1, "SYN_TUNE_W", n, s) for n in sorted(nomes("SYN_TUNE_W"), reverse=True) for s in SEEDS_TUNE_V]
            rodar(tar, out, pool, CAMPOS_V, cada=200)
            m, n = media_gap(out, "tune_static_W", cid)
            assert n == n_ref, (cid, n, n_ref)
            return m
        _es(out, estado_p, len(P9), avaliar, ES_GERACOES, ES_SEED_V, "VND P9 CVRPTW")


def modo_selecionar_v():
    import pandas as pd
    st = pd.read_csv(RES / "etapa14_tuning_static_v.csv")
    st = st[st.viavel].groupby(["config_id", "params"]).gap_pct.mean().reset_index().sort_values("gap_pct")
    top = st.iloc[0]
    u0 = st[st.config_id == "gen00"].gap_pct.iloc[0]
    par = json.loads(top.params)
    (RES / "etapa14_tuning_escolhida_v.json").write_text(json.dumps(
        {"params": par, "gap_tune": float(top.gap_pct), "gap_pesos_iguais": float(u0), "config_id": top.config_id,
         "n_configs": int(len(st))}, indent=1))
    print(f"static_W: melhor {top.config_id} gap_tune={top.gap_pct:.4f} (pesos iguais {u0:.4f}); pesos={[round(x, 3) for x in par['pesos']]}")


def specs_eval_v():
    esc = json.loads((RES / "etapa14_tuning_escolhida_v.json").read_text())
    o_custo, o_rev = _ordens()
    sp = {
        "s14v_U": spec_v("s14v_U", "U"),
        "s14v_ucb1c": spec_v("s14v_ucb1c", "ucb1", custo=True, params={"C": 2.0}),
        "s14v_pmc": spec_v("s14v_pmc", "pm", custo=True, params={"gamma": 0.1, "p_min": 0.05}),
        "s14v_hgucbc": spec_v("s14v_hgucbc", "hgucb", custo=True, params={"alpha": 0.5, "C": 2.0}),
        "s14v_ts": spec_v("s14v_ts", "ts"),
        "s14v_exp3": spec_v("s14v_exp3", "exp3", params={"gamma": 0.1}),
        "s14v_eff": spec_v("s14v_eff", "eff", params={"C": 1.0}),
        "s14v_static": spec_v("s14v_static", "static", params=esc["params"]),
        "s14v_P5_U": spec_v("s14v_P5_U", "U", pool="P5"),
        "s14v_FOC_idx": spec_v("s14v_FOC_idx", "FOC", ordem=ORDEM_IDX),
    }
    if o_custo is not None:
        sp["s14v_FOC_custo"] = spec_v("s14v_FOC_custo", "FOC", ordem=o_custo)
        sp["s14v_FOC_rev"] = spec_v("s14v_FOC_rev", "FOC", ordem=o_rev)
    return sp


def _instancias_aval():
    return sorted([("SOLOMON", n, 100, 10) for n in nomes("SOLOMON")], key=lambda t: t[1])


def modo_eval_v(workers):
    sp = specs_eval_v()
    tar = []
    for conj, n, _nc, ns in _instancias_aval():
        for nv in NIVEIS_V:
            tar += [(s, nv, conj, n, SEMENTES_AVAL + sd) for s in sp.values() for sd in range(ns)]
    print(f"eval VND: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tar, RES / "etapa14_eval_v.csv", pool, CAMPOS_V, cada=200)
    print("eval VND concluido", flush=True)


def modo_eval_pc(workers):
    """Controle positivo (analogo aos Estudos 8 e 13): quatro operadores inuteis de custo X n^2 plantados no pool P9, X dos pools D_LO e D_HI
    (fixados no piloto; ver POOLS_PC), em Solomon a E1 com 10 seeds; as 7 politicas sem a estatica."""
    base = specs_eval_v()
    arms = {k: base[f"s14v_{k}"] for k in ("U", "ucb1c", "pmc", "hgucbc", "ts", "exp3", "eff")}
    pcs = json.loads((RES / "etapa14_pools_pc.json").read_text())      # {"D_LO": "P9+D4x16", "D_HI": "P9+D4x32"}
    tar = []
    for tag, pool_nome in pcs.items():
        sp = [dict(a, nome=f"s14pc_{tag}_{k}", pool=pool_nome) for k, a in arms.items()]
        for n in nomes("SOLOMON"):
            tar += [(s, 1, "SOLOMON", n, SEMENTES_AVAL + sd) for s in sp for sd in range(10)]
    print(f"eval controle positivo: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tar, RES / "etapa14_eval_pc.csv", pool, CAMPOS_V, cada=200)
    print("eval controle positivo concluido", flush=True)


MODOS = {"pilot-v": piloto_v, "tuning-static-v": modo_tuning_static_v, "eval-v": modo_eval_v, "eval-pc": modo_eval_pc}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in list(MODOS) + ["selecionar-v"]:
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.selecionar_v:
        modo_selecionar_v()
    else:
        for m, f in MODOS.items():
            if getattr(a, m.replace("-", "_")):
                f(a.workers)


if __name__ == "__main__":
    main()
