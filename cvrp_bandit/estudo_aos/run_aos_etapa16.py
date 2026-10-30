#!/usr/bin/env python3
"""
run_aos_etapa16.py
==================
Estudo 16 da Direcao A: o REGIME ESTOCASTICO (ALNS) no CVRPTW -- o aparato dos Estudos 9, 12 e 13 (4 destruicoes x 3 reparos, recozimento por
fracao do orcamento, politica adaptativa de Ropke e Pisinger e as demais familias, controle de mistura estatica e de reparo fixo) com
janelas de tempo (src/tw_alns.py). Instancias do Estudo 14: as 56 de Solomon (avaliacao, 10 seeds) e SYN_TUNE_W (24; ajuste e pilotos).
Referencias do regime VND (as do Estudo 14: U em P9, U em P5, descida de ordem fixa de custo crescente, mistura estatica do VND) ao MESMO
orcamento do ALNS, para o contraste de estrutura. Retomavel (cada modo pula o que ja esta no CSV).

Modos: --pilot-a, --tuning-a, --selecionar-a, --eval-a, --eval-ref
Uso (WSL, venv de direcao1_local): python3 -u run_aos_etapa16.py --eval-a --workers 11
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
import run_aos_etapa14 as r14  # noqa: E402
from tw_alns import TWAlnsInst, EstadoTW, DESTRUIR_TW, REPARAR_TW  # noqa: E402
from aos_vnd import EstaticaPolitica  # noqa: E402
from alns_loop import rodar_alns, construir_politica_alns  # noqa: E402

RES = ROOT / "results" / "aos_study"
CAMPOS_A = ["braco", "config_id", "params", "nivel", "conjunto", "instancia", "n_customers", "seed", "orcamento",
            "custo", "ref", "gap_pct", "viavel", "n_rotas", "calls_usadas", "iteracoes", "inviaveis", "uso_d", "uso_r",
            "score_d", "score_r", "calls_d", "calls_r", "tempo_s"]

KAPPA_A = 1500         # orcamento do ALNS = KAPPA_A n^2 avaliacoes de distancia (fixado depois dos pilotos; ver diario)
W0_A = 0.2             # temperatura inicial = W0 * custo / ln 2; regra: w0 que minimiza o gap de U em SYN_TUNE_W a kappa=1500 (pilotos 1-3: 0,2)
TF_A = 1e-3
KAPPA_V = r14.KAPPA_V  # 600
SEMENTES_AVAL = 1000
SEEDS_PILOTO = [3000, 3001, 3002]
SEEDS_TUNE_A = [3000, 3001, 3002]
ES_FILHOS, ES_SIGMA0, ES_GERACOES_A, ES_SEED_A = 4, 0.5, 8, 20261116


# ----------------------------------------------------------------------------------------------- instancias
@lru_cache(maxsize=None)
def carregar(conj, nome):
    inst, x0 = r14.carregar(conj, nome)
    return TWAlnsInst.desde(inst), [list(r) for r in x0]


# ----------------------------------------------------------------------------------------------- regime ALNS
def spec_a(nome, pol, params=None, pesos_d=None, pesos_r=None, kappa=None, w0=None, tf=None, config_id="e16"):
    return {"nome": nome, "pol": pol, "params": params or {}, "pesos_d": pesos_d, "pesos_r": pesos_r,
            "kappa": KAPPA_A if kappa is None else kappa, "w0": W0_A if w0 is None else w0, "tf": TF_A if tf is None else tf,
            "config_id": config_id, "regime": "a"}


def executar_a(sp, nivel, conj, nome_inst, seed):
    inst, x0 = carregar(conj, nome_inst)
    inst.chamadas = 0
    rng = random.Random(seed)
    if sp["pol"] == "static":
        pol_d = EstaticaPolitica(sp["pesos_d"], rng=random.Random(seed * 31 + 1))
        pol_r = EstaticaPolitica(sp["pesos_r"], rng=random.Random(seed * 31 + 2))
        md = mr = "score"
    else:
        pol_d, md = construir_politica_alns(sp["pol"], len(DESTRUIR_TW), random.Random(seed * 31 + 1), sp["params"])
        pol_r, mr = construir_politica_alns(sp["pol"], len(REPARAR_TW), random.Random(seed * 31 + 2), sp["params"])
    orcamento = int(nivel * sp["kappa"] * inst.n_customers ** 2)
    t0 = time.perf_counter()
    best, info = rodar_alns(inst, [list(r) for r in x0], rng, orcamento, pol_d, md, pol_r, mr, lambda: inst.chamadas,
                            w0=sp["w0"], tf_fator=sp["tf"], max_rotas=inst.nveic, destruir=DESTRUIR_TW, reparar=REPARAR_TW,
                            estado_cls=EstadoTW)
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(best)
    viavel, _ = inst.is_feasible(best)
    j = lambda v: "|".join(f"{x:g}" for x in v)  # noqa: E731
    params = dict(sp["params"])
    if sp["pol"] == "static":
        params = {"pesos_d": [round(x, 4) for x in sp["pesos_d"]], "pesos_r": [round(x, 4) for x in sp["pesos_r"]]}
        if "theta" in sp["params"]:
            params["theta"] = sp["params"]["theta"]
    return {"braco": sp["nome"], "config_id": sp["config_id"], "params": json.dumps(params, sort_keys=True),
            "nivel": nivel, "conjunto": conj, "instancia": inst.nome, "n_customers": inst.n_customers,
            "seed": seed, "orcamento": orcamento, "custo": custo, "ref": inst.ref,
            "gap_pct": 100 * (custo - inst.ref) / inst.ref, "viavel": viavel,
            "n_rotas": len([r for r in best if r]), "calls_usadas": inst.chamadas, "iteracoes": info["iteracoes"],
            "inviaveis": info["inviaveis"], "uso_d": j(info["uso_d"]), "uso_r": j(info["uso_r"]),
            "score_d": j(info["score_d"]), "score_r": j(info["score_r"]), "calls_d": j(info["calls_d"]),
            "calls_r": j(info["calls_r"]), "tempo_s": dt}


# ----------------------------------------------------------------------------------------------- execucao em paralelo
def _worker(args):
    sp, nivel, conj, nome, seed = args
    return (r14.executar_v if sp["regime"] == "v" else executar_a)(sp, nivel, conj, nome, seed)


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


# ----------------------------------------------------------------------------------------------- piloto (SYN_TUNE_W)
def piloto_a(workers, kappas=(100, 400, 1500), w0s=(0.005, 0.05), kappa_w0=400):
    """Calibracao do ALNS em SYN_TUNE_W (NAO e dado do estudo): orcamento (kappa) e temperatura inicial; U x RP x reparo fixo
    regret-2 x reparo fixo guloso."""
    out = RES / "etapa16_piloto_a.csv"
    sp = []
    for kappa in kappas:
        for tag, pol, kw in (("U", "U", {}), ("RP", "RP", {}),
                             ("reg2", "static", dict(pesos_d=[0.25] * 4, pesos_r=[0.0, 1.0, 0.0])),
                             ("gul", "static", dict(pesos_d=[0.25] * 4, pesos_r=[1.0, 0.0, 0.0]))):
            sp.append(spec_a(f"pa_{tag}_k{kappa}", pol, kappa=kappa, config_id=f"k{kappa}", **kw))
    for w0 in w0s:
        for tag in ("U", "RP"):
            sp.append(spec_a(f"pa_{tag}_k{kappa_w0}_w{w0:g}", tag, kappa=kappa_w0, w0=w0, config_id=f"k{kappa_w0}w{w0:g}"))
    tarefas = [(s, 1, "SYN_TUNE_W", n, sd) for n in sorted(r14.nomes("SYN_TUNE_W"), reverse=True)
               for s in sp for sd in SEEDS_PILOTO]
    print(f"piloto ALNS-TW: {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tarefas, out, pool, CAMPOS_A, cada=20)
    print("piloto ALNS-TW concluido", flush=True)


def piloto_a2(workers):
    """2o piloto (SYN_TUNE_W; o 1o mostrou que a temperatura inicial pesa mais que no TSP): w0 em kappa = 1500 para U, RP e o reparo
    fixo regret-2, para nao prejudicar a selecao uniforme com uma temperatura ruim."""
    out = RES / "etapa16_piloto_a2.csv"
    sp = []
    for w0 in (0.02, 0.05, 0.1, 0.2):
        for tag, pol, kw in (("U", "U", {}), ("RP", "RP", {}),
                             ("reg2", "static", dict(pesos_d=[0.25] * 4, pesos_r=[0.0, 1.0, 0.0]))):
            sp.append(spec_a(f"pb_{tag}_w{w0:g}", pol, kappa=1500, w0=w0, config_id=f"w{w0:g}", **kw))
    tarefas = [(s, 1, "SYN_TUNE_W", n, sd) for n in sorted(r14.nomes("SYN_TUNE_W"), reverse=True)
               for s in sp for sd in SEEDS_PILOTO]
    print(f"piloto ALNS-TW 2: {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tarefas, out, pool, CAMPOS_A, cada=20)
    print("piloto ALNS-TW 2 concluido", flush=True)


def piloto_a3(workers):
    """3o piloto (SYN_TUNE_W): o 2o mostrou o gap de U caindo monotonicamente com w0 ate 0,2 (a maior temperatura testada); estende a grade
    de w0 (0,3, 0,5, 1,0) em kappa = 1500 para U, RP e o reparo fixo regret-2."""
    out = RES / "etapa16_piloto_a3.csv"
    sp = []
    for w0 in (0.3, 0.5, 1.0):
        for tag, pol, kw in (("U", "U", {}), ("RP", "RP", {}),
                             ("reg2", "static", dict(pesos_d=[0.25] * 4, pesos_r=[0.0, 1.0, 0.0]))):
            sp.append(spec_a(f"pc_{tag}_w{w0:g}", pol, kappa=1500, w0=w0, config_id=f"w{w0:g}", **kw))
    tarefas = [(s, 1, "SYN_TUNE_W", n, sd) for n in sorted(r14.nomes("SYN_TUNE_W"), reverse=True)
               for s in sp for sd in SEEDS_PILOTO]
    print(f"piloto ALNS-TW 3: {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tarefas, out, pool, CAMPOS_A, cada=20)
    print("piloto ALNS-TW 3 concluido", flush=True)


# ----------------------------------------------------------------------------------------------- DESENHO CONFIRMATORIO
GRADES_A = {"ucb1": [{"C": c} for c in (0.1, 0.3, 1.0, 2.0, 4.0)],
            "pm": [{"gamma": g, "p_min": p} for g in (0.05, 0.1, 0.3) for p in (0.02, 0.05, 0.1)],
            "hgucb": [{"alpha": a, "C": c} for a in (0.1, 0.5, 2.0) for c in (1.0, 2.0, 4.0)],
            "RP": [{"rho": r, "segmento": s} for r in (0.05, 0.1, 0.3) for s in (50, 100, 200)]}
PADRAO_A = {"ucb1": {"C": 2.0}, "pm": {"gamma": 0.1, "p_min": 0.05}, "hgucb": {"alpha": 0.5, "C": 2.0},
            "RP": {"rho": 0.1, "segmento": 100}}


def _es(estado_p, k, avaliar_fn, geracoes, base_seed, rotulo):
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


def modo_tuning_a(workers):
    """Grade das familias (UCB1, PM, HG-UCB, RP), reparo unico (3 configs) e busca estatica do ALNS, todos em SYN_TUNE_W."""
    out_g, out_s = RES / "etapa16_tuning_grid_a.csv", RES / "etapa16_tuning_static_a.csv"
    estado_p = RES / "etapa16_tuning_static_a_estado.json"
    insts = sorted(r14.nomes("SYN_TUNE_W"), reverse=True)
    specs = [spec_a("tune16_U", "U", config_id="U")]
    for fam, grade in GRADES_A.items():
        specs += [spec_a(f"tune16_{fam}", fam, params=p, config_id=json.dumps(p, sort_keys=True)) for p in grade]
    for j in range(3):
        pr = [1.0 if i == j else 0.0 for i in range(3)]
        specs.append(spec_a("tune16_fixrep", "static", pesos_d=[0.25] * 4, pesos_r=pr, config_id=f"r{j}"))
    n_ref = len(insts) * len(SEEDS_TUNE_A)
    with mp.Pool(workers) as pool:
        tar = [(s, 1, "SYN_TUNE_W", n, sd) for n in insts for s in specs for sd in SEEDS_TUNE_A]
        print(f"grade ALNS-TW: {len(specs)} configuracoes, {len(tar)} execucoes", flush=True)
        rodar(tar, out_g, pool, CAMPOS_A, cada=100)

        def avaliar(theta, cid):
            w = np.exp(np.array(theta, float))
            pd_, pr = (w[:4] / w[:4].sum()).tolist(), (w[4:] / w[4:].sum()).tolist()
            sp = spec_a("tune16_static_A", "static", pesos_d=pd_, pesos_r=pr, config_id=cid)
            sp["params"] = {"theta": list(map(float, theta))}
            rodar([(sp, 1, "SYN_TUNE_W", n, s) for n in insts for s in SEEDS_TUNE_A], out_s, pool, CAMPOS_A, cada=100)
            m, n = media_gap(out_s, "tune16_static_A", cid)
            assert n == n_ref, (cid, n, n_ref)
            return m
        _es(estado_p, 7, avaliar, ES_GERACOES_A, ES_SEED_A, "ALNS-TW")


def modo_selecionar_a():
    import pandas as pd
    g = pd.read_csv(RES / "etapa16_tuning_grid_a.csv")
    g = g[g.viavel]
    esc = {}
    print(f"U (referencia, SYN_TUNE_W): {g[g.braco == 'tune16_U'].gap_pct.mean():.4f}")
    for fam in GRADES_A:
        d = g[g.braco == f"tune16_{fam}"].groupby("config_id").gap_pct.agg(["mean", "count"]).sort_values("mean")
        padrao_id = json.dumps(PADRAO_A[fam], sort_keys=True)
        melhor_id = d.index[0]
        if d.loc[melhor_id, "mean"] >= d.loc[padrao_id, "mean"] - 1e-12:      # empate ou pior que o padrao -> padrao
            melhor_id = padrao_id
        esc[fam] = {"params": json.loads(melhor_id), "gap_tune": float(d.loc[melhor_id, "mean"]),
                    "gap_padrao": float(d.loc[padrao_id, "mean"]), "n_configs": int(len(d))}
        print(f"{fam}: escolhida {melhor_id} gap_tune={d.loc[melhor_id, 'mean']:.4f} (padrao {d.loc[padrao_id, 'mean']:.4f})")
    fr = g[g.braco == "tune16_fixrep"].groupby("config_id").gap_pct.mean().sort_values()
    esc["fixrep"] = {"j": int(fr.index[0][1:]), "gap_tune": float(fr.iloc[0]), "gaps": {k: float(v) for k, v in fr.items()}}
    print(f"reparo unico: {fr.to_dict()}")
    s = pd.read_csv(RES / "etapa16_tuning_static_a.csv")
    s = s[s.viavel].groupby(["config_id", "params"]).gap_pct.mean().reset_index().sort_values("gap_pct")
    top = s.iloc[0]
    th = np.array(json.loads(top.params)["theta"])
    w = np.exp(th)
    esc["static"] = {"pesos_d": (w[:4] / w[:4].sum()).tolist(), "pesos_r": (w[4:] / w[4:].sum()).tolist(),
                     "gap_tune": float(top.gap_pct), "gap_pesos_iguais": float(s[s.config_id == "gen00"].gap_pct.iloc[0]),
                     "config_id": top.config_id, "n_configs": int(len(s))}
    print(f"static_A: {top.config_id} gap_tune={top.gap_pct:.4f} pesos_d={np.round(esc['static']['pesos_d'], 3)} "
          f"pesos_r={np.round(esc['static']['pesos_r'], 3)}")
    (RES / "etapa16_tuning_escolhidas_a.json").write_text(json.dumps(esc, indent=1))


def _instancias_aval():
    """(conjunto, nome, n_customers, n_seeds): as 56 de Solomon, 10 seeds."""
    return [("SOLOMON", n, carregar("SOLOMON", n)[0].n_customers, 10) for n in r14.nomes("SOLOMON")]


def specs_eval_a():
    esc = json.loads((RES / "etapa16_tuning_escolhidas_a.json").read_text())
    jr = esc["fixrep"]["j"]
    return {
        "s16a_U": spec_a("s16a_U", "U"),
        "s16a_RP": spec_a("s16a_RP", "RP", params=esc["RP"]["params"]),
        "s16a_ucb1": spec_a("s16a_ucb1", "ucb1", params=esc["ucb1"]["params"]),
        "s16a_pm": spec_a("s16a_pm", "pm", params=esc["pm"]["params"]),
        "s16a_hgucb": spec_a("s16a_hgucb", "hgucb", params=esc["hgucb"]["params"]),
        "s16a_ts": spec_a("s16a_ts", "ts"),
        "s16a_exp3": spec_a("s16a_exp3", "exp3", params={"gamma": 0.1}),
        "s16a_eff": spec_a("s16a_eff", "eff", params={"C": 1.0}),
        "s16a_static": spec_a("s16a_static", "static", pesos_d=esc["static"]["pesos_d"], pesos_r=esc["static"]["pesos_r"]),
        "s16a_fixrep": spec_a("s16a_fixrep", "static", pesos_d=[0.25] * 4, pesos_r=[1.0 if i == jr else 0.0 for i in range(3)]),
    }


def modo_eval_a(workers):
    sp = specs_eval_a()
    tar = [(s, 1, conj, n, SEMENTES_AVAL + sd) for conj, n, _nc, ns in _instancias_aval() for s in sp.values()
           for sd in range(ns)]
    tar.sort(key=lambda t: t[3])
    print(f"eval ALNS-TW: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tar, RES / "etapa16_eval_a.csv", pool, CAMPOS_A, cada=100)
    print("eval ALNS-TW concluido", flush=True)


def modo_eval_ref(workers):
    """Referencias do regime VND (Estudo 14) ao MESMO orcamento do ALNS (nivel KAPPA_A / KAPPA_V): U em P9, U em P5, descida de ordem fixa de
    custo crescente e mistura estatica do VND; mesmas instancias e seeds do ALNS (a escolha de estrutura: ruin-and-recreate x descida)."""
    b = r14.specs_eval_v()
    nivel = KAPPA_A / KAPPA_V
    sp = [dict(b["s14v_U"], nome="s16r_U_P9"), dict(b["s14v_P5_U"], nome="s16r_U_P5"),
          dict(b["s14v_FOC_custo"], nome="s16r_FOC_custo"), dict(b["s14v_static"], nome="s16r_static")]
    tar = [(s, nivel, conj, n, SEMENTES_AVAL + sd) for conj, n, _nc, ns in _instancias_aval() for s in sp for sd in range(ns)]
    print(f"eval referencias VND: {len(tar)} execucoes (nivel {nivel})", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tar, RES / "etapa16_eval_ref.csv", pool, r14.CAMPOS_V, cada=100)
    print("eval referencias concluido", flush=True)


MODOS = {"pilot-a": piloto_a, "pilot-a2": piloto_a2, "pilot-a3": piloto_a3, "tuning-a": modo_tuning_a, "eval-a": modo_eval_a, "eval-ref": modo_eval_ref}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in list(MODOS) + ["selecionar-a"]:
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.selecionar_a:
        modo_selecionar_a()
    else:
        for m, f in MODOS.items():
            if getattr(a, m.replace("-", "_")):
                f(a.workers)


if __name__ == "__main__":
    main()
