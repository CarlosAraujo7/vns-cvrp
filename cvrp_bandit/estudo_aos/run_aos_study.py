#!/usr/bin/env python3
"""
run_aos_study.py
================
Runner do estudo controlado de selecao adaptativa de operadores (AOS) --
Etapa 1 da Direcao A. PROTOCOLO PRE-REGISTRADO em diario_pesquisa.md,
Entrada 64 (leia la antes de alterar qualquer coisa aqui).

Cada execucao = uma (politica, orcamento B em aplicacoes de operador,
instancia, seed). Orcamento medido por CONTAGEM DIRETA de chamadas a
`best_fn` (wrapper contador), nao por tempo.

Modos:
  --pilot   poucas instancias/seeds, B=800: valida infraestrutura (contagem
            exata de operacoes, viabilidade, regressao do `hgucb_norm`
            contra results/tempos_hgucb_74.csv, vazao). NAO entra em inferencia.
  --full    74 instancias x 10 seeds x 3 orcamentos x 8 politicas.

Resumivel: pula chaves (politica,B,instancia,seed) ja presentes no CSV.
Roda no WSL (venv de direcao1_local, que tem `river`):
    python3 -u run_aos_study.py --pilot
    python3 -u run_aos_study.py --full --workers 11
"""
import argparse
import csv
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from instance import load_all  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402,F401
from solucao_inicial import obter_inicial  # noqa: E402
from neighborhoods import NEIGHBORHOODS  # noqa: E402
from vns import bandit_vns  # noqa: E402
from bandit import HGUCBBandit, reward_positive_normalized  # noqa: E402
from aos_policies import (  # noqa: E402
    UniformePolitica, UCB1Politica, EpsGreedyPolitica, ProbMatchingPolitica,
    reward_relativa, reward_zero, gvns_budgeted,
)

MAX_INNER_ITER = 40
BUDGETS = (800, 4000, 20000)
SEED_BASE = 1000
CONJUNTOS = ("A", "B", "X", "CMT")
ARMS_CONFIRMATORIOS = ("gvns_classico", "uniforme", "ucb1", "eps_greedy",
                       "prob_matching", "hgucb_literal", "hgucb_norm")
ARMS_EXPLORATORIOS = ("hgucb_norm_a5",)
ARMS = ARMS_CONFIRMATORIOS + ARMS_EXPLORATORIOS

CAMPOS = ["politica", "B", "conjunto", "instancia", "n_customers", "seed", "custo", "otimo",
          "gap_pct", "viavel", "n_rotas", "n_ops_contadas", "uso_operadores", "tempo_s"]

_CACHE = {}


def _instancia(set_name, nome):
    if (set_name, nome) not in _CACHE:
        for i in load_all(ROOT / "instances" / set_name):
            if i.name == nome:
                # solucao inicial FIXADA em disco (diario, Entrada 88): o reparo com CP-SAT nao e deterministico
                cws = obter_inicial(i)
                _CACHE[(set_name, i.name)] = (i, cws)
                break
    return _CACHE[(set_name, nome)]


def _contar(contador, uso):
    """Envolve os best_fn de NEIGHBORHOODS com um contador de aplicacoes."""
    saida = []
    for idx, (nome, best_fn, rand_fn) in enumerate(NEIGHBORHOODS):
        def f(inst, routes, _b=best_fn, _i=idx):
            contador[0] += 1
            uso[_i] += 1
            return _b(inst, routes)
        saida.append((nome, f, rand_fn))
    return saida


def executar(politica, B, set_name, nome_inst, seed):
    inst, cws = _instancia(set_name, nome_inst)
    rng = random.Random(seed)
    contador, uso = [0], [0] * len(NEIGHBORHOODS)
    nbhs = _contar(contador, uso)
    k = len(nbhs)
    steps = B // MAX_INNER_ITER
    t0 = time.perf_counter()
    if politica == "gvns_classico":
        rotas = gvns_budgeted(inst, [list(r) for r in cws], rng, B, nbhs)
    else:
        kw = dict(steps_max=steps, neighborhoods=nbhs, max_inner_iter=MAX_INNER_ITER,
                  progressive_x_linha=True)
        if politica == "uniforme":
            bandit = UniformePolitica(k, rng=rng)
            kw.update(reward_positive_fn=reward_relativa, reward_negative_fn=reward_zero)
        elif politica == "ucb1":
            bandit = UCB1Politica(k, rng=rng)
            kw.update(reward_positive_fn=reward_relativa, reward_negative_fn=reward_zero)
        elif politica == "eps_greedy":
            bandit = EpsGreedyPolitica(k, rng=rng)
            kw.update(reward_positive_fn=reward_relativa, reward_negative_fn=reward_zero)
        elif politica == "prob_matching":
            bandit = ProbMatchingPolitica(k, rng=rng)
            kw.update(reward_positive_fn=reward_relativa, reward_negative_fn=reward_zero)
        elif politica == "hgucb_literal":      # Eq. 12/13 brutas, alpha do artigo
            bandit = HGUCBBandit(n_operators=k, rng=rng, p_hat_mode="q_values", alpha=5.0)
        elif politica == "hgucb_norm":         # configuracao de producao
            bandit = HGUCBBandit(n_operators=k, rng=rng, p_hat_mode="q_values", alpha=0.5)
            kw.update(reward_positive_fn=reward_positive_normalized)
        elif politica == "hgucb_norm_a5":      # exploratorio: isola a normalizacao
            bandit = HGUCBBandit(n_operators=k, rng=rng, p_hat_mode="q_values", alpha=5.0)
            kw.update(reward_positive_fn=reward_positive_normalized)
        else:
            raise ValueError(politica)
        rotas = bandit_vns(inst, [list(r) for r in cws], rng, bandit=bandit, **kw)
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(rotas)
    viavel, _ = inst.is_feasible(rotas)
    return {
        "politica": politica, "B": B, "conjunto": set_name, "instancia": inst.name,
        "n_customers": inst.n_customers, "seed": seed, "custo": custo, "otimo": inst.optimum,
        "gap_pct": 100 * (custo - inst.optimum) / inst.optimum, "viavel": viavel,
        "n_rotas": len([r for r in rotas if r]), "n_ops_contadas": contador[0],
        "uso_operadores": "|".join(map(str, uso)), "tempo_s": dt,
    }


def _worker(args):
    return executar(*args)


def todas_instancias():
    out = []
    for s in CONJUNTOS:
        for i in load_all(ROOT / "instances" / s):
            out.append((s, i.name, i.n_customers))
    return out


def rodar(tarefas, out_csv, workers):
    feitos = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            feitos = {(r["politica"], int(r["B"]), r["instancia"], int(r["seed"]))
                      for r in csv.DictReader(f)}
    pend = [t for t in tarefas if (t[0], t[1], t[3], t[4]) not in feitos]
    print(f"{len(tarefas)} tarefas, {len(feitos)} ja feitas, {len(pend)} pendentes, "
          f"{workers} workers", flush=True)
    novo = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f_out, fieldnames=CAMPOS)
    if novo:
        w.writeheader()
    t0 = time.perf_counter()
    feitas = 0
    with mp.Pool(workers) as pool:
        for linha in pool.imap_unordered(_worker, pend, chunksize=1):
            w.writerow(linha)
            feitas += 1
            if feitas % 50 == 0:
                f_out.flush()
                dt = time.perf_counter() - t0
                print(f"  {feitas}/{len(pend)}  {dt/60:.1f} min  "
                      f"(ETA {dt/feitas*(len(pend)-feitas)/60:.0f} min)", flush=True)
    f_out.close()
    print(f"Concluido em {(time.perf_counter()-t0)/60:.1f} min -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--exploratorio-b20000", dest="expl20k", action="store_true",
                   help="Subconjunto EXPLORATORIO de B=20000 (diario, Entrada 69): 3 politicas, "
                        "22 instancias por regra deterministica, 5 seeds.")
    ap.add_argument("--workers", type=int, default=11)
    args = ap.parse_args()
    insts = todas_instancias()
    if args.pilot:
        escolhidas = {"A-n32-k5", "B-n41-k6", "A-n80-k10", "CMT1", "X-n101-k25"}
        tarefas = [(a, 800, s, n, SEED_BASE + sd) for (s, n, _) in insts if n in escolhidas
                   for a in ARMS for sd in range(2)]
        rodar(tarefas, ROOT / "results" / "aos_study" / "pilot.csv", args.workers)
    elif args.expl20k:
        import numpy as np
        escolhidas = []
        for conj, k in (("A", 6), ("B", 6), ("X", 6), ("CMT", None)):
            ordenadas = sorted((nc, n, s_) for (s_, n, nc) in insts if s_ == conj)
            if k is None:
                pick = ordenadas
            else:
                idx = sorted(set(int(round(x)) for x in np.linspace(0, len(ordenadas) - 1, k)))
                pick = [ordenadas[i] for i in idx]
            escolhidas += [(s_, n, nc) for (nc, n, s_) in pick]
        assert len(escolhidas) == 22, len(escolhidas)
        tarefas = [(a, 20000, s_, n, SEED_BASE + sd)
                   for (s_, n, nc) in sorted(escolhidas, key=lambda t: -t[2])
                   for a in ("gvns_classico", "uniforme", "hgucb_norm") for sd in range(5)]
        rodar(tarefas, ROOT / "results" / "aos_study" / "exploratorio_b20000.csv", args.workers)
    else:
        # B crescente (resultados parciais de B menor chegam antes); dentro de cada B,
        # instancias maiores primeiro (balanceamento de carga)
        tarefas = [(a, B, s, n, SEED_BASE + sd)
                   for B in sorted(BUDGETS)
                   for (s, n, nc) in sorted(insts, key=lambda t: -t[2])
                   for a in ARMS for sd in range(10)]
        rodar(tarefas, ROOT / "results" / "aos_study" / "full.csv", args.workers)


if __name__ == "__main__":
    main()
