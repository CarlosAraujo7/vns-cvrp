#!/usr/bin/env python3
"""
run_aos_etapa17.py
==================
Runner do Estudo 17 (diario, Entradas 124 e seguintes): selecao adaptativa do CONJUNTO de operadores da busca local dentro do ILS do
PyVRP 0.13 (solver do estado da arte), CVRP. Conjuntos aninhados P1..P7 (src/pyvrp_aos.py); orcamento = TEMPO de parede por execucao,
T = nivel * n segundos (nivel = segundos por cliente). Cada execucao e um processo independente (11 em paralelo, 12 nucleos).

Uso (WSL, venv de direcao1_local; resumivel):
    python3 -u run_aos_etapa17.py --piloto --workers 11
"""
import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from instance import load_all  # noqa: E402
from pyvrp.stop import MaxRuntime  # noqa: E402
from pyvrp_aos import construir_dados, construir_pools, rodar_ils, rotas_em_ids  # noqa: E402

RES = ROOT / "results" / "aos_study"
CAMPOS = ["braco", "config_id", "params", "nivel", "conjunto", "instancia", "n_customers", "seed", "orcamento", "custo", "ref",
          "gap_pct", "viavel", "n_rotas", "iteracoes", "uso", "tempo_pool", "custo_pyvrp", "tempo_s"]
SEEDS_PILOTO = [3000, 3001, 3002]
SEEDS_TUNE = [3000, 3001, 3002]
SEMENTES_AVAL = 1000
ES_FILHOS, ES_SIGMA0, ES_GERACOES, ES_SEED = 4, 0.5, 8, 20261117
NIVEIS = (0.03, 0.1)    # segundos por cliente dos dois orcamentos; regra (so a curva do conjunto padrao P6 em SYN_TUNE): gap de P6 mais proximo de 1,5% e de 0,3%
K = 7     # numero de conjuntos (P1..P7)


# ----------------------------------------------------------------------------------------------- instancias
@lru_cache(maxsize=None)
def _lista(conj):
    return {i.name: i for i in load_all(ROOT / "instances" / conj)}


@lru_cache(maxsize=None)
def carregar(conj, nome):
    inst = _lista(conj)[nome]
    data, ids = construir_dados(inst)
    return inst, data, ids, construir_pools(data)


def nomes(conj):
    return sorted(_lista(conj))


# ----------------------------------------------------------------------------------------------- execucao
def spec17(nome, pol=None, params=None, fixo=None, config_id="e17"):
    """pol: U, RP, ucb1, pm, hgucb, ts, exp3, eff, static (params['pesos']) ou None com `fixo` = indice do conjunto (0 = P1)."""
    return {"nome": nome, "pol": pol, "params": params or {}, "fixo": fixo, "config_id": config_id}


def executar17(sp, nivel, conj, nome_inst, seed):
    inst, data, ids, pools = carregar(conj, nome_inst)
    tempo = nivel * inst.n_customers
    t0 = time.perf_counter()
    res, busca = rodar_ils(data, pools, seed, MaxRuntime(tempo), nome_pol=sp["pol"], params=sp["params"], fixo=sp["fixo"])
    dt = time.perf_counter() - t0
    rotas = rotas_em_ids(res.best, ids)
    viavel, _ = inst.is_feasible(rotas)
    viavel = bool(viavel and res.is_feasible())
    custo = inst.solution_cost(rotas)
    j = lambda v, f="g": "|".join(format(x, f) for x in v)  # noqa: E731
    return {"braco": sp["nome"], "config_id": sp["config_id"], "params": json.dumps(sp["params"], sort_keys=True),
            "nivel": nivel, "conjunto": conj, "instancia": inst.name, "n_customers": inst.n_customers, "seed": seed,
            "orcamento": tempo, "custo": custo, "ref": inst.optimum, "gap_pct": 100 * (custo - inst.optimum) / inst.optimum,
            "viavel": viavel, "n_rotas": len(rotas), "iteracoes": res.num_iterations, "uso": j(busca.uso),
            "tempo_pool": j(busca.tempo, ".4f"), "custo_pyvrp": res.cost(), "tempo_s": dt}


def _worker(args):
    return executar17(*args)


def chave(r):
    return (r["braco"], r["config_id"], float(r["nivel"]), r["conjunto"], r["instancia"], int(r["seed"]))


def rodar(tarefas, out_csv, pool_mp, cada=25):
    feitos = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            feitos = {chave(r) for r in csv.DictReader(f)}
    pend = [t for t in tarefas if (t[0]["nome"], t[0]["config_id"], float(t[1]), t[2], t[3], t[4]) not in feitos]
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
            if feitas % cada == 0:
                f_out.flush()
                print(f"    {feitas}/{len(pend)}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return len(pend)


# ----------------------------------------------------------------------------------------------- piloto (SYN_TUNE)
def modo_piloto(workers):
    """Calibracao do orcamento de tempo, SO em SYN_TUNE (ajuste; disjunto das instancias de avaliacao): conjuntos fixos P1..P7 e as
    politicas com parametros padrao, em tres orcamentos (segundos por cliente). Nao e resultado do estudo."""
    specs = [spec17(f"pil_P{j + 1}", fixo=j) for j in range(K)]
    specs += [spec17(f"pil_{p}", pol=p) for p in ("U", "RP", "hgucb", "ucb1", "eff")]
    tarefas = [(sp, nivel, "SYN_TUNE", nm, s) for nivel in (0.01, 0.03, 0.1) for sp in specs
               for nm in nomes("SYN_TUNE") for s in SEEDS_PILOTO]
    # embaralha de forma deterministica para equilibrar a carga entre processos (o tempo de parede e o orcamento)
    import random
    random.Random(17).shuffle(tarefas)
    with mp.Pool(workers) as pool_mp:
        n = rodar(tarefas, RES / "etapa17_piloto.csv", pool_mp)
    print(f"piloto concluido ({n} execucoes novas)", flush=True)


# ----------------------------------------------------------------------------------------------- DESENHO CONFIRMATORIO
GRADES = {"ucb1": [{"C": c} for c in (0.1, 0.3, 1.0, 2.0, 4.0)],
          "pm": [{"gamma": g, "p_min": p} for g in (0.05, 0.1, 0.3) for p in (0.02, 0.05, 0.1)],
          "hgucb": [{"alpha": a, "C": c} for a in (0.1, 0.5, 2.0) for c in (1.0, 2.0, 4.0)],
          "RP": [{"rho": r, "segmento": sg} for r in (0.05, 0.1, 0.3) for sg in (50, 100, 200)]}
PADRAO = {"ucb1": {"C": 2.0}, "pm": {"gamma": 0.1, "p_min": 0.05}, "hgucb": {"alpha": 0.5, "C": 2.0},
          "RP": {"rho": 0.1, "segmento": 100}}


def media_gap(out_csv, nome, config_id):
    import pandas as pd
    d = pd.read_csv(out_csv)
    d = d[(d.braco == nome) & (d.config_id == config_id) & d.viavel]
    return d.gap_pct.mean(), len(d)


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


def modo_tuning(workers):
    """Ajuste em SYN_TUNE no primeiro nivel de orcamento: U (referencia), grades das familias (UCB1, PM, HG-UCB, RP), conjuntos fixos
    P1..P7 e busca estatica (1+4)-ES de 8 geracoes sobre theta em R^7 (softmax sobre os 7 conjuntos)."""
    out_g, out_s = RES / "etapa17_tuning_grid.csv", RES / "etapa17_tuning_static.csv"
    estado_p = RES / "etapa17_tuning_static_estado.json"
    insts = nomes("SYN_TUNE")
    n1 = NIVEIS[0]
    specs = [spec17("tune17_U", "U", config_id="U")]
    for fam, grade in GRADES.items():
        specs += [spec17(f"tune17_{fam}", fam, params=p, config_id=json.dumps(p, sort_keys=True)) for p in grade]
    specs += [spec17("tune17_fixo", fixo=j, config_id=f"P{j + 1}") for j in range(K)]
    n_ref = len(insts) * len(SEEDS_TUNE)
    import random
    with mp.Pool(workers) as pool:
        tar = [(s, n1, "SYN_TUNE", n, sd) for n in insts for s in specs for sd in SEEDS_TUNE]
        random.Random(171).shuffle(tar)
        print(f"ajuste: {len(specs)} configuracoes, {len(tar)} execucoes", flush=True)
        rodar(tar, out_g, pool, cada=100)

        def avaliar(theta, cid):
            w = np.exp(np.array(theta, float))
            pesos = (w / w.sum()).tolist()
            sp = spec17("tune17_static", "static", params={"pesos": pesos, "theta": list(map(float, theta))}, config_id=cid)
            rodar([(sp, n1, "SYN_TUNE", n, sd) for n in insts for sd in SEEDS_TUNE], out_s, pool, cada=100)
            m, n = media_gap(out_s, "tune17_static", cid)
            assert n == n_ref, (cid, n, n_ref)
            return m
        _es(estado_p, K, avaliar, ES_GERACOES, ES_SEED, "conjuntos do PyVRP")


def modo_selecionar():
    import pandas as pd
    g = pd.read_csv(RES / "etapa17_tuning_grid.csv")
    g = g[g.viavel]
    esc = {}
    print(f"U (referencia, SYN_TUNE): {g[g.braco == 'tune17_U'].gap_pct.mean():.4f}")
    for fam in GRADES:
        d = g[g.braco == f"tune17_{fam}"].groupby("config_id").gap_pct.agg(["mean", "count"]).sort_values("mean")
        padrao_id = json.dumps(PADRAO[fam], sort_keys=True)
        melhor_id = d.index[0]
        if d.loc[melhor_id, "mean"] >= d.loc[padrao_id, "mean"] - 1e-12:      # empate ou pior que o padrao -> padrao
            melhor_id = padrao_id
        esc[fam] = {"params": json.loads(melhor_id), "gap_tune": float(d.loc[melhor_id, "mean"]),
                    "gap_padrao": float(d.loc[padrao_id, "mean"]), "n_configs": int(len(d))}
        print(f"{fam}: escolhida {melhor_id} gap_tune={d.loc[melhor_id, 'mean']:.4f} (padrao {d.loc[padrao_id, 'mean']:.4f})")
    fx = g[g.braco == "tune17_fixo"].groupby("config_id").gap_pct.mean().sort_values()
    esc["fixo"] = {"melhor": fx.index[0], "gaps": {k: float(v) for k, v in fx.items()}}
    print(f"conjuntos fixos (gap de ajuste): {fx.round(4).to_dict()}")
    s = pd.read_csv(RES / "etapa17_tuning_static.csv")
    s = s[s.viavel].groupby(["config_id", "params"]).gap_pct.mean().reset_index().sort_values("gap_pct")
    top = s.iloc[0]
    th = np.array(json.loads(top.params)["theta"])
    w = np.exp(th)
    esc["static"] = {"pesos": (w / w.sum()).tolist(), "gap_tune": float(top.gap_pct),
                     "gap_pesos_iguais": float(s[s.config_id == "gen00"].gap_pct.iloc[0]), "config_id": top.config_id,
                     "n_configs": int(len(s))}
    print(f"static: {top.config_id} gap_tune={top.gap_pct:.4f} pesos={np.round(esc['static']['pesos'], 3)}")
    (RES / "etapa17_tuning_escolhidas.json").write_text(json.dumps(esc, indent=1))


def _instancias_aval():
    """(conjunto, nome, n_customers, n_seeds): as 74 da CVRPLIB (A, B, CMT, X; 10 seeds) e as 30 sinteticas SYN_VAL (5 seeds)."""
    lista = [(c, n, 10) for c in ("A", "B", "CMT", "X") for n in nomes(c)] + [("SYN_VAL", n, 5) for n in nomes("SYN_VAL")]
    return [(c, n, _lista(c)[n].n_customers, ns) for c, n, ns in lista]


def specs_eval():
    esc = json.loads((RES / "etapa17_tuning_escolhidas.json").read_text())
    sp = {f"s17_P{j + 1}": spec17(f"s17_P{j + 1}", fixo=j) for j in range(2, K)}     # fixos P3..P7 (P1, P2 so no ajuste e como opcoes das politicas)
    sp.update({"s17_U": spec17("s17_U", "U"),
               "s17_RP": spec17("s17_RP", "RP", params=esc["RP"]["params"]),
               "s17_ucb1": spec17("s17_ucb1", "ucb1", params=esc["ucb1"]["params"]),
               "s17_pm": spec17("s17_pm", "pm", params=esc["pm"]["params"]),
               "s17_hgucb": spec17("s17_hgucb", "hgucb", params=esc["hgucb"]["params"]),
               "s17_ts": spec17("s17_ts", "ts"),
               "s17_exp3": spec17("s17_exp3", "exp3", params={"gamma": 0.1}),
               "s17_eff": spec17("s17_eff", "eff", params={"C": 1.0}),
               "s17_static": spec17("s17_static", "static", params={"pesos": esc["static"]["pesos"]})})
    return sp


def modo_eval(workers):
    import random
    sp = specs_eval()
    tar = [(s, nivel, conj, n, SEMENTES_AVAL + sd) for nivel in NIVEIS for conj, n, _nc, ns in _instancias_aval()
           for s in sp.values() for sd in range(ns)]
    random.Random(172).shuffle(tar)      # carga equilibrada entre processos: o orcamento e o tempo de parede
    print(f"avaliacao: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar(tar, RES / "etapa17_eval.csv", pool, cada=100)
    print("avaliacao concluida", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piloto", action="store_true")
    ap.add_argument("--tuning", action="store_true")
    ap.add_argument("--selecionar", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.piloto:
        modo_piloto(a.workers)
    if a.tuning:
        modo_tuning(a.workers)
    if a.selecionar:
        modo_selecionar()
    if a.eval:
        modo_eval(a.workers)


if __name__ == "__main__":
    main()
