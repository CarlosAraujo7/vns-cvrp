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
from run_aos_etapa3 import POOL_P10, PACIENCIA, KAPPA, _proxy  # noqa: E402
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
    pool = POOL_P10F
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


if __name__ == "__main__":
    print("este arquivo so tem o esqueleto de execucao; o desenho confirmatorio do Estudo 11 ainda nao foi pre-registrado")
