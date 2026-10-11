#!/usr/bin/env python3
"""
run_aos_etapa8.py
=================
Runner do Estudo 8 da Direcao A: CONTROLE POSITIVO e "pool inflado" (operadores inuteis ou parciais no conjunto).
O desenho confirmatorio so e fixado no diario DEPOIS do piloto em SYN_TUNE (instancias de ajuste, disjuntas das de
avaliacao); este arquivo no estado atual serve ao PILOTO. Estrutura fixa (a mesma dos Estudos 4 e 6): termino R5,
aceitacao "best", shake com os 5 operadores originais, orcamento em avaliacoes de distancia E1 = 600 n^2 (E2 = 5 E1),
solucao inicial fixada em disco.

Pools:  P10      os 10 operadores originais+novos (referencia)
        P10+Dk   P10 + k operadores INUTEIS (gastam n^2 avaliacoes e nunca melhoram)  -- efeito plantado
        P10+R4   P10 + 4 versoes RESTRITAS (2 rotas sorteadas) de Swap(1,1), Reloc(1,0), CROSS e SWAP*

Uso (WSL, venv de direcao1_local):
    python3 -u run_aos_etapa8.py --pilot --workers 11
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
from run_aos_etapa3 import POOL_P10, PACIENCIA, KAPPA, _proxy  # noqa: E402
import run_aos_etapa4 as r4  # noqa: E402
from neighborhoods import NEIGHBORHOODS  # noqa: E402
from aos_vnd import vnd_generico, construir_politica  # noqa: E402
from operadores_fracos import dud, restrito  # noqa: E402

RES = ROOT / "results" / "aos_study"
POOLS = {
    "P10": {"duds": [], "restr": []},
    "P10+D2": {"duds": [1.0, 1.0], "restr": []},
    "P10+D5": {"duds": [1.0] * 5, "restr": []},
    "P10+R4": {"duds": [], "restr": [2, 3, 8, 9]},
    # piloto 2: efeito plantado MAIOR (o do piloto 1 era ~0,2 pp, dentro do ruido de 24 instancias x 5 seeds)
    "P10+D2x8": {"duds": [8.0, 8.0], "restr": []},
    "P10+D4x4": {"duds": [4.0] * 4, "restr": []},
    "P10+D4x8": {"duds": [8.0] * 4, "restr": []},
}
PILOTO2 = ["P10+D2x8", "P10+D4x4", "P10+D4x8"]
POOLS_FULL = ["P10", "P10+D4x4", "P10+D4x8"]            # conjuntos da avaliacao confirmatoria (Entrada 101)
# braco -> (politica, custo_sensivel, params)
POLITICAS = {
    "U": ("U", False, {}),
    "ucb1c": ("ucb1", True, {}),
    "pmc": ("pm", True, {}),
    "hgucbc": ("hgucb", True, {}),
    "ts": ("ts", False, {}),
    "exp3": ("exp3", False, {"gamma": 0.1}),
    "eff": ("eff", False, {"C": 1.0}),
}


def executar8(sp, nivel, set_name, nome_inst, seed):
    inst, cws = _instancia(set_name, nome_inst)
    proxy = _proxy(set_name, nome_inst)
    proxy.chamadas = 0
    cfg = POOLS[sp["pool"]]
    rng_ops = random.Random(seed * 7919 + 13)          # sorteio das rotas dos operadores restritos (nao consome o rng da busca)
    pool = list(POOL_P10) + [dud(f) for f in cfg["duds"]] + [restrito(POOL_P10[i], rng_ops) for i in cfg["restr"]]
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


def _worker8(args):
    return executar8(*args)


def spec8(pool, arm):
    pol, custo, par = POLITICAS[arm]
    return {"nome": f"s8_{pool}_{arm}", "pol": pol, "custo": custo, "params": par, "config_id": "e8", "pool": pool}


def rodar8(tarefas, out_csv, pool_mp):
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
        for linha in pool_mp.imap_unordered(_worker8, pend, chunksize=1):
            w.writerow(linha)
            feitas += 1
            if feitas % 50 == 0:
                f_out.flush()
                print(f"    {feitas}/{len(pend)}  {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    return len(pend)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--pilot2", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.full:
        # desenho confirmatorio (diario, Entrada 101): 74 instancias CVRPLIB, 10 seeds, E1, 3 pools x 7 politicas
        from run_aos_study import todas_instancias
        insts = sorted(todas_instancias(), key=lambda x: -x[2])
        tarefas = [(spec8(p, arm), 1, c, n, 1000 + sd) for (c, n, nc) in insts for p in POOLS_FULL
                   for arm in POLITICAS for sd in range(10)]
        out = RES / "etapa8_full.csv"
        print(f"{len(tarefas)} tarefas (Estudo 8, avaliacao)", flush=True)
        with mp.Pool(a.workers) as pool:
            rodar8(tarefas, out, pool)
        print("concluido", flush=True)
        return
    nm = r4.nomes("SYN_TUNE")
    pools = PILOTO2 if a.pilot2 else [p for p in POOLS if p not in PILOTO2]
    tarefas = [(spec8(p, arm), 1, "SYN_TUNE", n, s) for n in sorted(nm, reverse=True) for p in pools
               for arm in POLITICAS for s in r4.SEEDS_TUNE]
    out = RES / "etapa8_pilot.csv"
    print(f"{len(tarefas)} tarefas (piloto em SYN_TUNE)", flush=True)
    with mp.Pool(a.workers) as pool:
        rodar8(tarefas, out, pool)
    print("concluido", flush=True)


if __name__ == "__main__":
    main()
