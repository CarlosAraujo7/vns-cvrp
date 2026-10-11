#!/usr/bin/env python3
"""
run_aos_etapa9.py
=================
Runner do Estudo 9 da Direcao A: AOS no cenario CANONICO de operadores ESTOCASTICOS (ALNS para o CVRP;
src/alns.py, src/alns_loop.py). Este arquivo, no estado atual, serve a CALIBRACAO em SYN_TUNE (instancias de
ajuste, disjuntas das de avaliacao); o desenho confirmatorio so e fixado no diario DEPOIS dela.

Orcamento em avaliacoes de distancia: kappa * n^2 (kappa=ORC por nivel). Aceitacao por recozimento com temperatura
em funcao da fracao do orcamento. Destruicao (4) e reparo (3) escolhidos por duas instancias da mesma politica.

Uso (WSL, venv de direcao1_local):
    python3 -u run_aos_etapa9.py --calib --workers 11
"""
import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_aos_study import _instancia  # noqa: E402
from run_aos_etapa3 import _proxy  # noqa: E402
import run_aos_etapa4 as r4  # noqa: E402
from alns import DESTRUIR, REPARAR  # noqa: E402
from alns_loop import rodar_alns, construir_politica_alns  # noqa: E402

RES = ROOT / "results" / "aos_study"
CAMPOS9 = ["braco", "config_id", "params", "nivel", "conjunto", "instancia", "n_customers", "seed", "orcamento",
           "custo", "ref", "gap_pct", "viavel", "n_rotas", "calls_usadas", "iteracoes", "inviaveis", "uso_d", "uso_r",
           "score_d", "score_r", "calls_d", "calls_r", "tempo_s"]


def spec9(nome, pol, kappa, w0=0.05, tf=1e-3, params=None, config_id="e9"):
    return {"nome": nome, "pol": pol, "kappa": kappa, "w0": w0, "tf": tf, "params": params or {},
            "config_id": config_id}


def executar9(sp, nivel, set_name, nome_inst, seed):
    inst, cws = _instancia(set_name, nome_inst)
    px = _proxy(set_name, nome_inst)
    px.chamadas = 0
    rng = random.Random(seed)
    pol_d, md = construir_politica_alns(sp["pol"], len(DESTRUIR), random.Random(seed * 31 + 1), sp["params"])
    pol_r, mr = construir_politica_alns(sp["pol"], len(REPARAR), random.Random(seed * 31 + 2), sp["params"])
    orcamento = int(nivel * sp["kappa"] * inst.n_customers ** 2)
    max_rotas = inst.n_vehicles_ref if inst.vehicles_exact else None
    t0 = time.perf_counter()
    best, info = rodar_alns(px, [list(r) for r in cws], rng, orcamento, pol_d, md, pol_r, mr, lambda: px.chamadas,
                            w0=sp["w0"], tf_fator=sp["tf"], max_rotas=max_rotas)
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(best)
    viavel, _ = inst.is_feasible(best)
    j = lambda v: "|".join(f"{x:g}" for x in v)  # noqa: E731
    return {"braco": sp["nome"], "config_id": sp["config_id"], "params": json.dumps(sp["params"], sort_keys=True),
            "nivel": nivel, "conjunto": set_name, "instancia": inst.name, "n_customers": inst.n_customers,
            "seed": seed, "orcamento": orcamento, "custo": custo, "ref": inst.optimum,
            "gap_pct": 100 * (custo - inst.optimum) / inst.optimum, "viavel": viavel,
            "n_rotas": len([r for r in best if r]), "calls_usadas": px.chamadas, "iteracoes": info["iteracoes"],
            "inviaveis": info["inviaveis"], "uso_d": j(info["uso_d"]), "uso_r": j(info["uso_r"]),
            "score_d": j(info["score_d"]), "score_r": j(info["score_r"]), "calls_d": j(info["calls_d"]),
            "calls_r": j(info["calls_r"]), "tempo_s": dt}


def _worker9(args):
    return executar9(*args)


def rodar9(tarefas, out_csv, pool_mp):
    feitos = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            feitos = {r4.chave(r) for r in csv.DictReader(f)}
    pend = [t for t in tarefas if (t[0]["nome"], t[0]["config_id"], t[1], t[3], t[4]) not in feitos]
    if not pend:
        return 0
    novo = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "a", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=CAMPOS9)
        if novo:
            w.writeheader()
        t0 = time.perf_counter()
        feitas = 0
        for linha in pool_mp.imap_unordered(_worker9, pend, chunksize=1):
            w.writerow(linha)
            feitas += 1
            if feitas % 25 == 0:
                f_out.flush()
                print(f"    {feitas}/{len(pend)}  {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    return len(pend)


# ------------------------------------------------------------------ desenho confirmatorio (Entrada 101) ---------
KAPPA9, W09, TF9 = 1500, 0.02, 1e-3                       # fixados pela calibracao (Entrada 100)
SEEDS_TUNE9 = [3000, 3001, 3002]
GRADES9 = {"ucb1": [{"C": c} for c in (0.1, 0.3, 1.0, 2.0, 4.0)],
           "pm": [{"gamma": g, "p_min": p} for g in (0.05, 0.1, 0.3) for p in (0.02, 0.05, 0.1)],
           "hgucb": [{"alpha": a, "C": c} for a in (0.1, 0.5, 2.0) for c in (1.0, 2.0, 4.0)],
           "RP": [{"rho": r, "segmento": s} for r in (0.05, 0.1, 0.3) for s in (50, 100, 200)]}
PADRAO9 = {"ucb1": {"C": 2.0}, "pm": {"gamma": 0.1, "p_min": 0.05}, "hgucb": {"alpha": 0.5, "C": 2.0},
           "RP": {"rho": 0.1, "segmento": 100}}


def modo_tuning_grid(workers):
    out = RES / "etapa9_tuning_grid.csv"
    nm = sorted(r4.nomes("SYN_TUNE"), reverse=True)
    specs = [spec9("tune9_U", "U", KAPPA9, W09, TF9, config_id="U")]
    for fam, grade in GRADES9.items():
        specs += [spec9(f"tune9_{fam}", fam, KAPPA9, W09, TF9, params=p, config_id=json.dumps(p, sort_keys=True))
                  for p in grade]
    tarefas = [(sp, 1, "SYN_TUNE", n, s) for n in nm for sp in specs for s in SEEDS_TUNE9]
    print(f"grade: {len(specs)} configuracoes, {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar9(tarefas, out, pool)
    print("grade concluida", flush=True)


def modo_selecionar():
    import pandas as pd
    g = pd.read_csv(RES / "etapa9_tuning_grid.csv")
    g = g[g.viavel]
    escolhidas = {}
    print(f"U (referencia, SYN_TUNE): {g[g.braco == 'tune9_U'].gap_pct.mean():.4f}")
    for fam in GRADES9:
        d = g[g.braco == f"tune9_{fam}"].groupby("config_id").gap_pct.agg(["mean", "count"]).sort_values("mean")
        padrao_id = json.dumps(PADRAO9[fam], sort_keys=True)
        melhor_id = d.index[0]
        if d.loc[melhor_id, "mean"] >= d.loc[padrao_id, "mean"] - 1e-12:   # empate ou pior que o padrao -> padrao
            melhor_id = padrao_id
        escolhidas[fam] = {"params": json.loads(melhor_id), "gap_tune": float(d.loc[melhor_id, "mean"]),
                           "gap_padrao": float(d.loc[padrao_id, "mean"]), "n_configs": int(len(d))}
        print(f"{fam}: escolhida {melhor_id} gap_tune={d.loc[melhor_id, 'mean']:.4f} (padrao {d.loc[padrao_id, 'mean']:.4f}; "
              f"pior {d['mean'].max():.4f})")
    (RES / "etapa9_tuning_escolhidas.json").write_text(json.dumps(escolhidas, indent=1))


def specs_eval9():
    esc = json.loads((RES / "etapa9_tuning_escolhidas.json").read_text())
    s = lambda nome, pol, par: spec9(nome, pol, KAPPA9, W09, TF9, params=par, config_id="e9")  # noqa: E731
    return {"s9_U": s("s9_U", "U", {}), "s9_RP": s("s9_RP", "RP", esc["RP"]["params"]),
            "s9_ucb1": s("s9_ucb1", "ucb1", esc["ucb1"]["params"]), "s9_pm": s("s9_pm", "pm", esc["pm"]["params"]),
            "s9_hgucb": s("s9_hgucb", "hgucb", esc["hgucb"]["params"]), "s9_ts": s("s9_ts", "ts", {}),
            "s9_exp3": s("s9_exp3", "exp3", {"gamma": 0.1}), "s9_eff": s("s9_eff", "eff", {"C": 1.0})}


def _worker_vns(args):
    sp, nivel, c, n, sd = args
    return r4.executar4(sp, nivel, c, n, sd)


def modo_vns(workers):
    """Braços VNS ao MESMO orcamento do ALNS (kappa=1500 = 2,5 x E1), nas mesmas instancias e seeds do Estudo 9:
    U (P10, R5) e a mistura estatica do Estudo 4. Sao a referencia 'melhor VNS' contra o ALNS."""
    esc4 = json.loads((RES / "etapa4_tuning_escolhidas.json").read_text())
    sp = [r4.spec("s9_vns_U", "U", termino="R", custo=False, config_id="e9v", pool="P10"),
          r4.spec("s9_vns_static", "static", termino="R", custo=False, params=esc4["static"]["params"],
                  config_id="e9v", pool="P10")]
    insts = []
    for conj in ("X", "SYN_VAL"):
        from instance import load_all
        insts += [(conj, i.name, i.n_customers) for i in load_all(ROOT / "instances" / conj)]
    insts.sort(key=lambda t: -t[2])
    nivel = KAPPA9 / 600.0                                   # 2,5: orcamento = 1500 n^2
    tarefas = [(s, nivel, c, n, 1000 + sd) for (c, n, nc) in insts for s in sp for sd in range(10 if c == "X" else 5)]
    out = RES / "etapa9_vns.csv"
    feitos = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            feitos = {(r["braco"], r["config_id"], float(r["nivel"]), r["instancia"], int(r["seed"]))
                      for r in csv.DictReader(f)}
    pend = [t for t in tarefas if (t[0]["nome"], t[0]["config_id"], float(t[1]), t[3], t[4]) not in feitos]
    print(f"vns: {len(tarefas)} execucoes ({len(pend)} pendentes)", flush=True)
    novo = not out.exists()
    with mp.Pool(workers) as pool, open(out, "a", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=r4.CAMPOS)
        if novo:
            w.writeheader()
        for k, linha in enumerate(pool.imap_unordered(_worker_vns, pend, chunksize=1), 1):
            w.writerow(linha)
            if k % 50 == 0:
                f_out.flush()
                print(f"    {k}/{len(pend)}", flush=True)
    print("vns concluido", flush=True)


def modo_eval(workers):
    sp = specs_eval9()
    insts = []
    for conj in ("X", "SYN_VAL"):
        from instance import load_all
        insts += [(conj, i.name, i.n_customers) for i in load_all(ROOT / "instances" / conj)]
    insts.sort(key=lambda t: -t[2])
    tarefas = [(s, 1, c, n, 1000 + sd) for (c, n, nc) in insts for s in sp.values()
               for sd in range(10 if c == "X" else 5)]
    print(f"eval: {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar9(tarefas, RES / "etapa9_eval.csv", pool)
    print("eval concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--calib", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--tuning-grid", action="store_true")
    g.add_argument("--selecionar", action="store_true")
    g.add_argument("--eval", action="store_true")
    g.add_argument("--vns", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.tuning_grid:
        return modo_tuning_grid(a.workers)
    if a.selecionar:
        return modo_selecionar()
    if a.eval:
        return modo_eval(a.workers)
    if a.vns:
        return modo_vns(a.workers)
    nm = r4.nomes("SYN_TUNE")
    if a.pilot:
        # piloto das politicas (SYN_TUNE; exploratorio): configuracao do ALNS fixada pela calibracao (diario, Entrada 100)
        arms = [("p_U", "U", {}), ("p_RP", "RP", {}), ("p_ucb1", "ucb1", {"C": 2.0}), ("p_pm", "pm", {}),
                ("p_ts", "ts", {}), ("p_exp3", "exp3", {"gamma": 0.1}), ("p_hgucb", "hgucb", {"alpha": 0.5, "C": 2.0}),
                ("p_eff", "eff", {"C": 1.0})]
        tarefas = [(spec9(nome, pol, 1500, 0.02, 1e-3, params=par, config_id="pilot"), 1, "SYN_TUNE", n, s)
                   for n in sorted(nm, reverse=True) for (nome, pol, par) in arms for s in r4.SEEDS_TUNE]
        out = RES / "etapa9_pilot.csv"
        print(f"{len(tarefas)} tarefas (piloto das politicas em SYN_TUNE)", flush=True)
        with mp.Pool(a.workers) as pool:
            rodar9(tarefas, out, pool)
        print("concluido", flush=True)
        return
    seeds = r4.SEEDS_TUNE[:3]
    # calibracao: politica uniforme; kappa e temperatura (w0, tf) -- mesma para todos os bracos depois
    cfgs = [("c_k300_w05_t3", 300, 0.05, 1e-3), ("c_k1500_w05_t3", 1500, 0.05, 1e-3),
            ("c_k1500_w02_t3", 1500, 0.02, 1e-3), ("c_k1500_w10_t3", 1500, 0.10, 1e-3),
            ("c_k1500_w05_t2", 1500, 0.05, 1e-2), ("c_k1500_w05_t4", 1500, 0.05, 1e-4)]
    tarefas = [(spec9(nome, "U", k, w, t, config_id=nome), 1, "SYN_TUNE", n, s)
               for n in sorted(nm, reverse=True) for (nome, k, w, t) in cfgs for s in seeds]
    out = RES / "etapa9_calib.csv"
    print(f"{len(tarefas)} tarefas (calibracao em SYN_TUNE)", flush=True)
    with mp.Pool(a.workers) as pool:
        rodar9(tarefas, out, pool)
    print("concluido", flush=True)


if __name__ == "__main__":
    main()
