#!/usr/bin/env python3
"""
run_aos_etapa11.py
==================
Runner do Estudo 11 da Direcao A: os resultados centrais sobrevivem a um SWAP* EFICIENTE? O SWAP* ingenuo dos
Estudos 3-10 custa ~18x mais avaliacoes de distancia que o de Vidal (2022) (tests/test_swap_star_fast.py: mesmo melhor
movimento, razao de avaliacoes 4-34x) e seu custo cresce com n; este estudo troca-o por `swap_star_fast_best` e refaz o
nucleo da comparacao (politicas vs uniforme, mistura estatica ajustada, efeito do tamanho). PROTOCOLO PRE-REGISTRADO
no diario; este arquivo, no estado atual, so tem o ESQUELETO de execucao (o desenho confirmatorio vem na entrada).

Pool P10F = os 10 operadores do P10 com o SWAP* (indice 9) trocado pelo eficiente (mesma vizinhanca, mesma melhor
melhora; so muda a contabilidade de avaliacoes). Estrutura fixa: termino R5, aceitacao "best", shake dos 5 originais,
orcamento E1 = 600 n^2 (E2 = 5 E1), solucao inicial fixada em disco.
"""
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
from run_aos_etapa3 import POOL_P10, POOL_P5, PACIENCIA, KAPPA, _proxy  # noqa: E402
import run_aos_etapa4 as r4  # noqa: E402
from neighborhoods import NEIGHBORHOODS  # noqa: E402
from neighborhoods_ext import _nao_usado_random  # noqa: E402
from neighborhoods_fast import swap_star_fast_best  # noqa: E402
from aos_vnd import vnd_generico, construir_politica  # noqa: E402

RES = ROOT / "results" / "aos_study"
POOL_P10F = POOL_P10[:9] + [("SWAP*-rapido", swap_star_fast_best, _nao_usado_random)]


def executar11(sp, nivel, set_name, nome_inst, seed):
    inst, cws = _instancia(set_name, nome_inst)
    proxy = _proxy(set_name, nome_inst)
    proxy.chamadas = 0
    pool = POOL_P10F if sp.get("pool", "P10F") == "P10F" else POOL_P5
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
    politica, modo = construir_politica(sp["pol"], k, rng, custo=sp["custo"], params=sp["params"])
    orcamento = nivel * KAPPA * inst.n_customers ** 2
    t0 = time.perf_counter()
    rotas, info = vnd_generico(inst, [list(r) for r in cws], rng, orcamento, nbhs, politica, "best", "R",
                               modo_recompensa=modo, neighborhoods_shake=NEIGHBORHOODS,
                               medida=lambda: proxy.chamadas, paciencia=PACIENCIA, on_aplicacao=on_aplicacao)
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(rotas)
    viavel, _ = inst.is_feasible(rotas)
    return {
        "braco": sp["nome"], "config_id": sp["config_id"], "params": json.dumps(sp["params"], sort_keys=True),
        "nivel": nivel, "conjunto": set_name, "instancia": inst.name, "n_customers": inst.n_customers,
        "seed": seed, "orcamento": orcamento, "custo": custo, "ref": inst.optimum,
        "gap_pct": 100 * (custo - inst.optimum) / inst.optimum, "viavel": viavel,
        "n_rotas": len([r for r in rotas if r]), "calls_usadas": proxy.chamadas, "max_calls_app": maximo[0],
        "apps": info["usado"], "shakes": info["shakes"], "paciencia_fim": info["paciencia_fim"],
        "uso_operadores": "|".join(map(str, uso)), "calls_por_operador": "|".join(map(str, calls_op)),
        "suc_por_operador": "|".join(map(str, suc)), "ganho_por_operador": "|".join(f"{g:.6f}" for g in gan),
        "tempo_s": dt,
    }


def _worker11(args):
    return executar11(*args)


def rodar11(tarefas, out_csv, pool_mp):
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
        w = csv.DictWriter(f_out, fieldnames=r4.CAMPOS)
        if novo:
            w.writeheader()
        t0 = time.perf_counter()
        feitas = 0
        for linha in pool_mp.imap_unordered(_worker11, pend, chunksize=1):
            w.writerow(linha)
            feitas += 1
            if feitas % 50 == 0:
                f_out.flush()
                print(f"    {feitas}/{len(pend)}  {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    return len(pend)


import argparse  # noqa: E402
import numpy as np  # noqa: E402
from instance import load_all  # noqa: E402

SEEDS_TUNE11 = [3000, 3001, 3002, 3003, 3004]
ES_BASE_SEED = 20261111


def spec11(nome, pol, custo=False, params=None, pool="P10F", config_id="e11"):
    return {"nome": nome, "pol": pol, "custo": custo, "params": params or {}, "config_id": config_id, "pool": pool,
            "termino": "R"}


def tarefas_tune(sp):
    return [(sp, 1, "SYN_TUNE", n, s) for n in sorted(r4.nomes("SYN_TUNE"), reverse=True) for s in SEEDS_TUNE11]


def modo_tuning(workers):
    out = RES / "etapa11_tuning_static.csv"
    estado_p = RES / "etapa11_tuning_static_estado.json"
    k = len(POOL_P10F)
    n_ref = len(r4.nomes("SYN_TUNE")) * len(SEEDS_TUNE11)
    estado = (json.loads(estado_p.read_text()) if estado_p.exists()
              else {"geracao": 0, "theta": [0.0] * k, "sigma": r4.ES_SIGMA0, "melhor": None})

    def pesos(theta):
        w = np.exp(np.array(theta))
        return (w / w.sum()).tolist()

    with mp.Pool(workers) as pool:
        def avaliar(theta, cid):
            sp = spec11("tune_static_F", "static", params={"pesos": pesos(theta), "theta": list(map(float, theta))},
                        config_id=cid)
            rodar11(tarefas_tune(sp), out, pool)
            m, n = r4.media_gap(out, "tune_static_F", cid)
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
    print("busca estatica (P10F) concluida", flush=True)


def modo_selecionar():
    import pandas as pd
    st = pd.read_csv(RES / "etapa11_tuning_static.csv")
    st = st[st.viavel].groupby(["config_id", "params"]).gap_pct.mean().reset_index().sort_values("gap_pct")
    top = st.iloc[0]
    par = json.loads(top.params)
    u0 = st[st.config_id == "gen00"].gap_pct.iloc[0]
    (RES / "etapa11_tuning_escolhida.json").write_text(json.dumps(
        {"params": par, "gap_tune": float(top.gap_pct), "gap_pesos_iguais": float(u0), "config_id": top.config_id,
         "n_configs": int(len(st))}, indent=1))
    print(f"static_F: melhor {top.config_id} gap_tune={top.gap_pct:.4f} (pesos iguais {u0:.4f}); pesos="
          f"{[round(x, 3) for x in par['pesos']]}")


def specs_eval():
    esc = json.loads((RES / "etapa11_tuning_escolhida.json").read_text())
    return {
        "s11_U": spec11("s11_U", "U"),
        "s11_ucb1c": spec11("s11_ucb1c", "ucb1", custo=True, params={"C": 2.0}),
        "s11_pmc": spec11("s11_pmc", "pm", custo=True, params={"gamma": 0.1, "p_min": 0.05}),
        "s11_hgucbc": spec11("s11_hgucbc", "hgucb", custo=True, params={"alpha": 0.5, "C": 2.0}),
        "s11_ts": spec11("s11_ts", "ts"),
        "s11_exp3": spec11("s11_exp3", "exp3", params={"gamma": 0.1}),
        "s11_eff": spec11("s11_eff", "eff", params={"C": 1.0}),
        "s11_static_F": spec11("s11_static_F", "static", params=esc["params"]),
        "s11_P5_U": spec11("s11_P5_U", "U", pool="P5"),
    }


def modo_eval74(workers):
    from run_aos_study import todas_instancias
    sp = specs_eval()
    insts = sorted(todas_instancias(), key=lambda x: -x[2])
    tarefas = [(s, nv, c, n, 1000 + sd) for nv in (1, 5) for (c, n, nc) in insts for s in sp.values() for sd in range(10)]
    print(f"eval74: {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar11(tarefas, RES / "etapa11_eval74.csv", pool)
    print("eval74 concluido", flush=True)


def modo_evalL(workers):
    sp = {k: v for k, v in specs_eval().items() if k != "s11_P5_U"}
    nm = [i.name for i in sorted(load_all(ROOT / "instances" / "SYN_VAL_L"), key=lambda i: -i.n_customers)]
    tarefas = [(s, 1, "SYN_VAL_L", n, 1000 + sd) for n in nm for s in sp.values() for sd in range(5)]
    print(f"evalL: {len(tarefas)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar11(tarefas, RES / "etapa11_evalL.csv", pool)
    print("evalL concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("tuning-static", "selecionar", "eval74", "evalL"):
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.tuning_static:
        modo_tuning(a.workers)
    elif a.selecionar:
        modo_selecionar()
    elif a.eval74:
        modo_eval74(a.workers)
    else:
        modo_evalL(a.workers)


if __name__ == "__main__":
    main()
