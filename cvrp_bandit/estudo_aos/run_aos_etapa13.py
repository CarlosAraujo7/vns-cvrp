#!/usr/bin/env python3
"""
run_aos_etapa13.py
==================
Runner do Estudo 13 da Direcao A: GENERALIDADE -- o aparato dos Estudos 1-12 aplicado a um segundo problema, o TSP simetrico
euclidiano (src/tsp_inst.py, src/tsp_neighborhoods.py, src/tsp_alns.py; referencia LKH-3). Dois regimes, como no CVRP:
  VND  (operadores deterministas de melhor melhora, pool P9; laco generico aos_vnd.vnd_generico, termino R5, aceitacao
        "best", shake com os 5 operadores originais P5, orcamento E1 = KAPPA_V n^2 avaliacoes de distancia, E2 = 5 E1)
  ALNS (operadores estocasticos: 4 de destruicao x 3 de reparo, recozimento por fracao do orcamento; alns_loop.rodar_alns)
Conjuntos: TSPLIB (32 instancias EUC_2D, 51-226 nos, otimo publicado), SYN_TUNE_T (24, ajuste) e SYN_VAL_T (30, avaliacao).
Solucao inicial: vizinho mais proximo a partir do no 0 (determinista, igual para todos os bracos).

Este arquivo, no estado atual, serve aos PILOTOS de calibracao em SYN_TUNE_T (instancias de ajuste, disjuntas das de
avaliacao); o desenho confirmatorio so e fixado no diario DEPOIS deles. Uso (WSL, venv de direcao1_local):
    python3 -u run_aos_etapa13.py --pilot-v --workers 11
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
from tsp_inst import ler_tsplib, vizinho_mais_proximo  # noqa: E402
from tsp_neighborhoods import P9, P5  # noqa: E402
from tsp_alns import DESTRUIR_TSP, REPARAR_TSP, MAX_ROTAS_TSP  # noqa: E402
from aos_vnd import vnd_generico, construir_politica, EstaticaPolitica  # noqa: E402
from alns_loop import rodar_alns, construir_politica_alns  # noqa: E402
from operadores_fracos import dud  # noqa: E402

RES = ROOT / "results" / "aos_study"
INST = ROOT / "instances"
CAMPOS_V = ["braco", "config_id", "params", "nivel", "conjunto", "instancia", "n_customers", "seed",
            "orcamento", "custo", "ref", "gap_pct", "viavel", "n_rotas", "calls_usadas", "max_calls_app",
            "apps", "shakes", "paciencia_fim", "uso_operadores", "calls_por_operador",
            "suc_por_operador", "ganho_por_operador", "tempo_s"]
CAMPOS_A = ["braco", "config_id", "params", "nivel", "conjunto", "instancia", "n_customers", "seed", "orcamento",
            "custo", "ref", "gap_pct", "viavel", "n_rotas", "calls_usadas", "iteracoes", "inviaveis", "uso_d", "uso_r",
            "score_d", "score_r", "calls_d", "calls_r", "tempo_s"]

KAPPA_V = 600          # E1 = 600 n^2 (como nos Estudos 3-11); E2 = 5 E1
PACIENCIA = 5
KAPPA_A = 1500         # orcamento do ALNS = 1500 n^2 (como no Estudo 9)
W0_A = 0.02            # temperatura inicial = W0 * custo / ln 2 (valor do Estudo 9; reconferido no piloto)
TF_A = 1e-3

POOLS13 = {"P9": (P9, []), "P5": (P5, []),
           "P9+D4x4": (P9, [4.0] * 4), "P9+D4x8": (P9, [8.0] * 4), "P9+D4x16": (P9, [16.0] * 4),
           "P9+D4x32": (P9, [32.0] * 4)}


# ----------------------------------------------------------------------------------------------- instancias
@lru_cache(maxsize=None)
def _refs(conj):
    if conj == "TSPLIB":
        return {r["instancia"]: float(r["otimo"]) for r in csv.DictReader(open(INST / "TSPLIB" / "optimos_tsplib.csv"))}
    return {r["instancia"]: float(r["ref"]) for r in csv.DictReader(open(INST / conj / "ref.csv"))}


def nomes(conj):
    return sorted(_refs(conj))


@lru_cache(maxsize=None)
def carregar(conj, nome):
    inst = ler_tsplib(INST / conj / f"{nome}.tsp", ref=_refs(conj)[nome])
    return inst, vizinho_mais_proximo(inst)


# ----------------------------------------------------------------------------------------------- regime VND
def spec_v(nome, pol, custo=False, params=None, pool="P9", config_id="e13"):
    return {"nome": nome, "pol": pol, "custo": custo, "params": params or {}, "config_id": config_id, "pool": pool,
            "regime": "v"}


def executar_v(sp, nivel, conj, nome_inst, seed):
    inst, nn = carregar(conj, nome_inst)
    inst.chamadas = 0
    base, duds = POOLS13[sp["pool"]]
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
    rotas, info = vnd_generico(inst, [list(nn)], rng, orcamento, nbhs, politica, "best", "R", modo_recompensa=modo,
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
        "n_rotas": 1, "calls_usadas": inst.chamadas, "max_calls_app": maximo[0],
        "apps": info["usado"], "shakes": info["shakes"], "paciencia_fim": info["paciencia_fim"],
        "uso_operadores": "|".join(map(str, uso)), "calls_por_operador": "|".join(map(str, calls_op)),
        "suc_por_operador": "|".join(map(str, suc)), "ganho_por_operador": "|".join(f"{g:.6f}" for g in gan),
        "tempo_s": dt,
    }


# ----------------------------------------------------------------------------------------------- regime ALNS
def spec_a(nome, pol, params=None, pesos_d=None, pesos_r=None, kappa=KAPPA_A, w0=W0_A, tf=TF_A, config_id="e13"):
    return {"nome": nome, "pol": pol, "params": params or {}, "pesos_d": pesos_d, "pesos_r": pesos_r,
            "kappa": kappa, "w0": w0, "tf": tf, "config_id": config_id, "regime": "a"}


def executar_a(sp, nivel, conj, nome_inst, seed):
    inst, nn = carregar(conj, nome_inst)
    inst.chamadas = 0
    rng = random.Random(seed)
    if sp["pol"] == "static":
        pol_d = EstaticaPolitica(sp["pesos_d"], rng=random.Random(seed * 31 + 1))
        pol_r = EstaticaPolitica(sp["pesos_r"], rng=random.Random(seed * 31 + 2))
        md = mr = "score"
    else:
        pol_d, md = construir_politica_alns(sp["pol"], len(DESTRUIR_TSP), random.Random(seed * 31 + 1), sp["params"])
        pol_r, mr = construir_politica_alns(sp["pol"], len(REPARAR_TSP), random.Random(seed * 31 + 2), sp["params"])
    orcamento = int(nivel * sp["kappa"] * inst.n_customers ** 2)
    t0 = time.perf_counter()
    best, info = rodar_alns(inst, [list(nn)], rng, orcamento, pol_d, md, pol_r, mr, lambda: inst.chamadas,
                            w0=sp["w0"], tf_fator=sp["tf"], max_rotas=MAX_ROTAS_TSP,
                            destruir=DESTRUIR_TSP, reparar=REPARAR_TSP)
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(best)
    viavel, _ = inst.is_feasible(best)
    j = lambda v: "|".join(f"{x:g}" for x in v)  # noqa: E731
    params = dict(sp["params"])
    if sp["pol"] == "static":
        params = {"pesos_d": [round(x, 4) for x in sp["pesos_d"]], "pesos_r": [round(x, 4) for x in sp["pesos_r"]]}
        if "theta" in sp["params"]:
            params["theta"] = sp["params"]["theta"]                  # o ES guarda theta (precisao total) para a selecao
    return {"braco": sp["nome"], "config_id": sp["config_id"], "params": json.dumps(params, sort_keys=True),
            "nivel": nivel, "conjunto": conj, "instancia": inst.nome, "n_customers": inst.n_customers,
            "seed": seed, "orcamento": orcamento, "custo": custo, "ref": inst.ref,
            "gap_pct": 100 * (custo - inst.ref) / inst.ref, "viavel": viavel,
            "n_rotas": 1, "calls_usadas": inst.chamadas, "iteracoes": info["iteracoes"],
            "inviaveis": info["inviaveis"], "uso_d": j(info["uso_d"]), "uso_r": j(info["uso_r"]),
            "score_d": j(info["score_d"]), "score_r": j(info["score_r"]), "calls_d": j(info["calls_d"]),
            "calls_r": j(info["calls_r"]), "tempo_s": dt}


# ----------------------------------------------------------------------------------------------- execucao em paralelo
def _worker(args):
    sp, nivel, conj, nome, seed = args
    return (executar_v if sp["regime"] == "v" else executar_a)(sp, nivel, conj, nome, seed)


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


# ----------------------------------------------------------------------------------------------- pilotos (SYN_TUNE_T)
SEEDS_PILOTO = [3000, 3001, 3002]


def piloto_v(workers):
    """Calibracao do regime VND em SYN_TUNE_T: gap em varios niveis de orcamento (multiplos de KAPPA_V n^2), custo por
    operador e tamanho dos duds. As execucoes sao baratas (operadores vetorizados), ao contrario das do CVRP."""
    out = RES / "etapa13_piloto_v.csv"
    sp = [spec_v("p_U", "U", pool="P9"), spec_v("p_U_P5", "U", pool="P5")]
    sp += [spec_v(f"p_U_{p}", "U", pool=p) for p in ("P9+D4x4", "P9+D4x8", "P9+D4x16", "P9+D4x32")]
    sp += [spec_v("p_ucb1c", "ucb1", custo=True, params={"C": 2.0}), spec_v("p_ts", "ts"), spec_v("p_eff", "eff", params={"C": 1.0})]
    tarefas = [(s, nv, "SYN_TUNE_T", n, sd) for nv in (1, 5, 25, 125) for n in sorted(nomes("SYN_TUNE_T"), reverse=True)
               for s in sp for sd in SEEDS_PILOTO]
    print(f"piloto VND: {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tarefas, out, pool, CAMPOS_V, cada=200)
    print("piloto VND concluido", flush=True)


def piloto_a(workers):
    """Calibracao do ALNS em SYN_TUNE_T: orcamento (kappa) e temperatura inicial; U x RP x reparo fixo regret-2."""
    out = RES / "etapa13_piloto_a.csv"
    sp = []
    for kappa in (25, 100, 400):
        for pol, kw in (("U", {}), ("RP", {}), ("static", dict(pesos_d=[0.25] * 4, pesos_r=[0.0, 1.0, 0.0]))):
            sp.append(spec_a(f"pa_{pol if pol != 'static' else 'reg2'}_k{kappa}", pol, kappa=kappa, config_id=f"k{kappa}", **kw))
    for w0 in (0.005, 0.05):
        for pol in ("U", "RP"):
            sp.append(spec_a(f"pa_{pol}_k100_w{w0:g}", pol, kappa=100, w0=w0, config_id=f"k100w{w0:g}"))
    tarefas = [(s, 1, "SYN_TUNE_T", n, sd) for n in sorted(nomes("SYN_TUNE_T"), reverse=True)
               for s in sp for sd in SEEDS_PILOTO]
    print(f"piloto ALNS: {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tarefas, out, pool, CAMPOS_A, cada=50)
    print("piloto ALNS concluido", flush=True)


def piloto_a2(workers):
    """2o piloto do ALNS: o 1o (kappa 25-400) mostrou so ~80-300 iteracoes por execucao -- no TSP cada insercao varre o
    passeio inteiro (n avaliacoes), logo o orcamento por iteracao e ~n vezes maior que no CVRP, e o RP (segmentos de 100
    iteracoes) nao chega a adaptar. Varre kappa maiores."""
    out = RES / "etapa13_piloto_a2.csv"
    sp = []
    for kappa in (800, 1500, 3000):
        for pol, kw in (("U", {}), ("RP", {}), ("static", dict(pesos_d=[0.25] * 4, pesos_r=[0.0, 1.0, 0.0]))):
            sp.append(spec_a(f"pa_{pol if pol != 'static' else 'reg2'}_k{kappa}", pol, kappa=kappa, config_id=f"k{kappa}", **kw))
    tarefas = [(s, 1, "SYN_TUNE_T", n, sd) for n in sorted(nomes("SYN_TUNE_T"), reverse=True)
               for s in sp for sd in SEEDS_PILOTO]
    print(f"piloto ALNS 2: {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tarefas, out, pool, CAMPOS_A, cada=50)
    print("piloto ALNS 2 concluido", flush=True)


# ----------------------------------------------------------------------------------------------- DESENHO CONFIRMATORIO
# (fixado no diario, Entrada 113, ANTES de qualquer dado de ajuste ou avaliacao; os pilotos acima so usaram SYN_TUNE_T)
ES_GERACOES, ES_FILHOS, ES_SIGMA0 = 15, 4, 0.5        # VND: (1+4)-ES do Estudo 4 (61 avaliacoes)
ES_GERACOES_A = 8                                     # ALNS: 8 geracoes x 4 filhos (33 avaliacoes), como o Estudo 12
SEEDS_TUNE_V = [3000, 3001, 3002, 3003, 3004]
SEEDS_TUNE_A = [3000, 3001, 3002]
ES_SEED_V, ES_SEED_A = 20261113, 20261114
GRADES_A = {"ucb1": [{"C": c} for c in (0.1, 0.3, 1.0, 2.0, 4.0)],
            "pm": [{"gamma": g, "p_min": p} for g in (0.05, 0.1, 0.3) for p in (0.02, 0.05, 0.1)],
            "hgucb": [{"alpha": a, "C": c} for a in (0.1, 0.5, 2.0) for c in (1.0, 2.0, 4.0)],
            "RP": [{"rho": r, "segmento": s} for r in (0.05, 0.1, 0.3) for s in (50, 100, 200)]}
PADRAO_A = {"ucb1": {"C": 2.0}, "pm": {"gamma": 0.1, "p_min": 0.05}, "hgucb": {"alpha": 0.5, "C": 2.0},
            "RP": {"rho": 0.1, "segmento": 100}}
NIVEIS_V_TSPLIB = (1, 5, 25)                          # E1, E2 e E3 = 25 E1 (E3 exploratorio)
NIVEL_REF = KAPPA_A / KAPPA_V                         # 2,5: o VND ao mesmo orcamento do ALNS
SEMENTES_AVAL = 1000


def _softmax(theta):
    w = np.exp(np.array(theta, float))
    return (w / w.sum())


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


def modo_tuning_static_v(workers):
    out, estado_p = RES / "etapa13_tuning_static_v.csv", RES / "etapa13_tuning_static_v_estado.json"
    n_ref = len(nomes("SYN_TUNE_T")) * len(SEEDS_TUNE_V)
    with mp.Pool(workers) as pool:
        def avaliar(theta, cid):
            sp = spec_v("tune_static_T", "static", params={"pesos": _softmax(theta).tolist(), "theta": list(map(float, theta))},
                        config_id=cid)
            tar = [(sp, 1, "SYN_TUNE_T", n, s) for n in sorted(nomes("SYN_TUNE_T"), reverse=True) for s in SEEDS_TUNE_V]
            rodar(tar, out, pool, CAMPOS_V, cada=200)
            m, n = media_gap(out, "tune_static_T", cid)
            assert n == n_ref, (cid, n, n_ref)
            return m
        _es(out, estado_p, len(P9), avaliar, ES_GERACOES, ES_SEED_V, "VND P9")


def modo_selecionar_v():
    import pandas as pd
    st = pd.read_csv(RES / "etapa13_tuning_static_v.csv")
    st = st[st.viavel].groupby(["config_id", "params"]).gap_pct.mean().reset_index().sort_values("gap_pct")
    top = st.iloc[0]
    u0 = st[st.config_id == "gen00"].gap_pct.iloc[0]
    par = json.loads(top.params)
    (RES / "etapa13_tuning_escolhida_v.json").write_text(json.dumps(
        {"params": par, "gap_tune": float(top.gap_pct), "gap_pesos_iguais": float(u0), "config_id": top.config_id,
         "n_configs": int(len(st))}, indent=1))
    print(f"static_T: melhor {top.config_id} gap_tune={top.gap_pct:.4f} (pesos iguais {u0:.4f}); "
          f"pesos={[round(x, 3) for x in par['pesos']]}")


def modo_tuning_a(workers):
    """Grade das familias (UCB1, PM, HG-UCB, RP), reparo unico (3 configs) e busca estatica do ALNS, todos em SYN_TUNE_T."""
    out_g, out_s = RES / "etapa13_tuning_grid_a.csv", RES / "etapa13_tuning_static_a.csv"
    estado_p = RES / "etapa13_tuning_static_a_estado.json"
    insts = sorted(nomes("SYN_TUNE_T"), reverse=True)
    specs = [spec_a("tune13_U", "U", config_id="U")]
    for fam, grade in GRADES_A.items():
        specs += [spec_a(f"tune13_{fam}", fam, params=p, config_id=json.dumps(p, sort_keys=True)) for p in grade]
    for j in range(3):
        pr = [1.0 if i == j else 0.0 for i in range(3)]
        specs.append(spec_a("tune13_fixrep", "static", pesos_d=[0.25] * 4, pesos_r=pr, config_id=f"r{j}"))
    n_ref = len(insts) * len(SEEDS_TUNE_A)
    with mp.Pool(workers) as pool:
        tar = [(s, 1, "SYN_TUNE_T", n, sd) for n in insts for s in specs for sd in SEEDS_TUNE_A]
        print(f"grade ALNS: {len(specs)} configuracoes, {len(tar)} execucoes", flush=True)
        rodar(tar, out_g, pool, CAMPOS_A, cada=100)

        def avaliar(theta, cid):
            w = np.exp(np.array(theta, float))
            pd_, pr = (w[:4] / w[:4].sum()).tolist(), (w[4:] / w[4:].sum()).tolist()
            sp = spec_a("tune13_static_A", "static", pesos_d=pd_, pesos_r=pr, config_id=cid)
            sp["params"] = {"theta": list(map(float, theta))}
            rodar([(sp, 1, "SYN_TUNE_T", n, s) for n in insts for s in SEEDS_TUNE_A], out_s, pool, CAMPOS_A, cada=100)
            m, n = media_gap(out_s, "tune13_static_A", cid)
            assert n == n_ref, (cid, n, n_ref)
            return m
        _es(out_s, estado_p, 7, avaliar, ES_GERACOES_A, ES_SEED_A, "ALNS")


def modo_selecionar_a():
    import pandas as pd
    g = pd.read_csv(RES / "etapa13_tuning_grid_a.csv")
    g = g[g.viavel]
    esc = {}
    print(f"U (referencia, SYN_TUNE_T): {g[g.braco == 'tune13_U'].gap_pct.mean():.4f}")
    for fam in GRADES_A:
        d = g[g.braco == f"tune13_{fam}"].groupby("config_id").gap_pct.agg(["mean", "count"]).sort_values("mean")
        padrao_id = json.dumps(PADRAO_A[fam], sort_keys=True)
        melhor_id = d.index[0]
        if d.loc[melhor_id, "mean"] >= d.loc[padrao_id, "mean"] - 1e-12:      # empate ou pior que o padrao -> padrao
            melhor_id = padrao_id
        esc[fam] = {"params": json.loads(melhor_id), "gap_tune": float(d.loc[melhor_id, "mean"]),
                    "gap_padrao": float(d.loc[padrao_id, "mean"]), "n_configs": int(len(d))}
        print(f"{fam}: escolhida {melhor_id} gap_tune={d.loc[melhor_id, 'mean']:.4f} (padrao {d.loc[padrao_id, 'mean']:.4f})")
    fr = g[g.braco == "tune13_fixrep"].groupby("config_id").gap_pct.mean().sort_values()
    esc["fixrep"] = {"j": int(fr.index[0][1:]), "gap_tune": float(fr.iloc[0]), "gaps": {k: float(v) for k, v in fr.items()}}
    print(f"reparo unico: {fr.to_dict()}")
    s = pd.read_csv(RES / "etapa13_tuning_static_a.csv")
    s = s[s.viavel].groupby(["config_id", "params"]).gap_pct.mean().reset_index().sort_values("gap_pct")
    top = s.iloc[0]
    th = np.array(json.loads(top.params)["theta"])
    w = np.exp(th)
    esc["static"] = {"pesos_d": (w[:4] / w[:4].sum()).tolist(), "pesos_r": (w[4:] / w[4:].sum()).tolist(),
                     "gap_tune": float(top.gap_pct), "gap_pesos_iguais": float(s[s.config_id == "gen00"].gap_pct.iloc[0]),
                     "config_id": top.config_id, "n_configs": int(len(s))}
    print(f"static_A: {top.config_id} gap_tune={top.gap_pct:.4f} pesos_d={np.round(esc['static']['pesos_d'], 3)} "
          f"pesos_r={np.round(esc['static']['pesos_r'], 3)}")
    (RES / "etapa13_tuning_escolhidas_a.json").write_text(json.dumps(esc, indent=1))


def _instancias_aval():
    """(conjunto, nome, n_customers, n_seeds): TSPLIB (10 seeds) e SYN_VAL_T (5 seeds), das maiores as menores."""
    out = []
    for conj, ns in (("TSPLIB", 10), ("SYN_VAL_T", 5)):
        out += [(conj, n, carregar(conj, n)[0].n_customers, ns) for n in nomes(conj)]
    return sorted(out, key=lambda t: -t[2])


def specs_eval_v():
    esc = json.loads((RES / "etapa13_tuning_escolhida_v.json").read_text())
    return {
        "s13v_U": spec_v("s13v_U", "U"),
        "s13v_ucb1c": spec_v("s13v_ucb1c", "ucb1", custo=True, params={"C": 2.0}),
        "s13v_pmc": spec_v("s13v_pmc", "pm", custo=True, params={"gamma": 0.1, "p_min": 0.05}),
        "s13v_hgucbc": spec_v("s13v_hgucbc", "hgucb", custo=True, params={"alpha": 0.5, "C": 2.0}),
        "s13v_ts": spec_v("s13v_ts", "ts"),
        "s13v_exp3": spec_v("s13v_exp3", "exp3", params={"gamma": 0.1}),
        "s13v_eff": spec_v("s13v_eff", "eff", params={"C": 1.0}),
        "s13v_static": spec_v("s13v_static", "static", params=esc["params"]),
        "s13v_P5_U": spec_v("s13v_P5_U", "U", pool="P5"),
    }


def modo_eval_v(workers):
    sp = specs_eval_v()
    tar = []
    for conj, n, _nc, ns in _instancias_aval():
        for nv in (NIVEIS_V_TSPLIB if conj == "TSPLIB" else (1,)):
            tar += [(s, nv, conj, n, SEMENTES_AVAL + sd) for s in sp.values() for sd in range(ns)]
    print(f"eval VND: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tar, RES / "etapa13_eval_v.csv", pool, CAMPOS_V, cada=200)
    print("eval VND concluido", flush=True)


def modo_eval_pc(workers):
    """Controle positivo (analogo ao Estudo 8): quatro operadores inuteis de custo X n^2 plantados no pool P9, X = 8 e 16,
    em TSPLIB a E1 com 10 seeds; politicas com hiperparametros padrao (os mesmos de specs_eval_v)."""
    base = specs_eval_v()
    arms = {"U": base["s13v_U"], "ucb1c": base["s13v_ucb1c"], "pmc": base["s13v_pmc"], "hgucbc": base["s13v_hgucbc"],
            "ts": base["s13v_ts"], "exp3": base["s13v_exp3"], "eff": base["s13v_eff"]}
    tar = []
    for pool_nome, tag in (("P9+D4x8", "D8"), ("P9+D4x16", "D16")):
        sp = [dict(a, nome=f"s13pc_{tag}_{k}", pool=pool_nome) for k, a in arms.items()]
        for n in nomes("TSPLIB"):
            tar += [(s, 1, "TSPLIB", n, SEMENTES_AVAL + sd) for s in sp for sd in range(10)]
    tar.sort(key=lambda t: -carregar(t[2], t[3])[0].n_customers)
    print(f"eval controle positivo: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tar, RES / "etapa13_eval_pc.csv", pool, CAMPOS_V, cada=200)
    print("eval controle positivo concluido", flush=True)


def specs_eval_a():
    esc = json.loads((RES / "etapa13_tuning_escolhidas_a.json").read_text())
    jr = esc["fixrep"]["j"]
    return {
        "s13a_U": spec_a("s13a_U", "U"),
        "s13a_RP": spec_a("s13a_RP", "RP", params=esc["RP"]["params"]),
        "s13a_ucb1": spec_a("s13a_ucb1", "ucb1", params=esc["ucb1"]["params"]),
        "s13a_pm": spec_a("s13a_pm", "pm", params=esc["pm"]["params"]),
        "s13a_hgucb": spec_a("s13a_hgucb", "hgucb", params=esc["hgucb"]["params"]),
        "s13a_ts": spec_a("s13a_ts", "ts"),
        "s13a_exp3": spec_a("s13a_exp3", "exp3", params={"gamma": 0.1}),
        "s13a_eff": spec_a("s13a_eff", "eff", params={"C": 1.0}),
        "s13a_static": spec_a("s13a_static", "static", pesos_d=esc["static"]["pesos_d"], pesos_r=esc["static"]["pesos_r"]),
        "s13a_fixrep": spec_a("s13a_fixrep", "static", pesos_d=[0.25] * 4, pesos_r=[1.0 if i == jr else 0.0 for i in range(3)]),
    }


def modo_eval_a(workers):
    sp = specs_eval_a()
    tar = [(s, 1, conj, n, SEMENTES_AVAL + sd) for conj, n, _nc, ns in _instancias_aval() for s in sp.values()
           for sd in range(ns)]
    print(f"eval ALNS: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tar, RES / "etapa13_eval_a.csv", pool, CAMPOS_A, cada=100)
    print("eval ALNS concluido", flush=True)


def modo_eval_ref(workers):
    """Referencias VND ao MESMO orcamento do ALNS (nivel 2,5 = 1500 n^2): uniforme em P9, uniforme em P5 e a mistura
    estatica do VND; mesmas instancias e seeds do ALNS (a escolha de estrutura: ruin-and-recreate x descida)."""
    b = specs_eval_v()
    sp = [dict(b["s13v_U"], nome="s13r_U_P9"), dict(b["s13v_P5_U"], nome="s13r_U_P5"), dict(b["s13v_static"], nome="s13r_static")]
    tar = [(s, NIVEL_REF, conj, n, SEMENTES_AVAL + sd) for conj, n, _nc, ns in _instancias_aval() for s in sp for sd in range(ns)]
    print(f"eval referencias VND: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tar, RES / "etapa13_eval_ref.csv", pool, CAMPOS_V, cada=100)
    print("eval referencias concluido", flush=True)


def modo_explor_ordem(workers):
    """POS-HOC, exploratorio (NAO pre-registrado; escrito depois de ver a Entrada 114): ucb1c e hgucbc ganham de U em E1 com uso quase
    uniforme dos operadores; testa se o ganho vem da ORDEM inicial (a rodada de inicializacao dos bandits percorre o pool na ordem
    do indice, aproximadamente de custo crescente): VND classico de ordem FIXA do P9 (FOC) em ordem crescente e decrescente de indice.
    TSPLIB (E1, E2, E3) e SYN_VAL_T (E1), mesmas sementes da avaliacao."""
    sp = [spec_v("s13x_FOC_asc", "FOC"), spec_v("s13x_FOC_desc", "FOC")]
    sp[0]["ordem"] = list(range(9))
    sp[1]["ordem"] = list(range(8, -1, -1))
    tar = []
    for conj, n, _nc, ns in _instancias_aval():
        for nv in (NIVEIS_V_TSPLIB if conj == "TSPLIB" else (1,)):
            tar += [(s, nv, conj, n, SEMENTES_AVAL + sd) for s in sp for sd in range(ns)]
    print(f"explor ordem: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tar, RES / "etapa13_explor_ordem.csv", pool, CAMPOS_V, cada=100)
    print("explor ordem concluido", flush=True)


MODOS = {"pilot-v": piloto_v, "pilot-a": piloto_a, "pilot-a2": piloto_a2,
         "tuning-static-v": modo_tuning_static_v, "tuning-a": modo_tuning_a,
         "eval-v": modo_eval_v, "eval-pc": modo_eval_pc, "eval-a": modo_eval_a, "eval-ref": modo_eval_ref,
         "explor-ordem": modo_explor_ordem}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in list(MODOS) + ["selecionar-v", "selecionar-a"]:
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.selecionar_v:
        modo_selecionar_v()
    elif a.selecionar_a:
        modo_selecionar_a()
    else:
        for m, f in MODOS.items():
            if getattr(a, m.replace("-", "_")):
                f(a.workers)


if __name__ == "__main__":
    main()
