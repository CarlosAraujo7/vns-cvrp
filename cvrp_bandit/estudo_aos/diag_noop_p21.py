#!/usr/bin/env python3
"""
diag_noop_p21.py -- DIAGNOSTICO (instancias de AJUSTE SYN_TUNE; nao e dado de avaliacao): com que frequencia cada destruicao de P21 seguida de cada reparo devolve a MESMA solucao (mesmo custo)
de uma solucao corrente de boa qualidade (a do fim de uma corrida curta do ALNS com RP)? O ALNS pontua um candidato igual ao corrente como 'aceito pior' (sigma3 = 13), entao um operador que quase
sempre reconstroi a mesma solucao recebe escore alto sem nada fazer (o Estudo 21 mostrou o RP concentrando 30% das destruicoes em rota_pequena).
Uso (WSL): python3 diag_noop_p21.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa4 as r4  # noqa: E402
import run_aos_etapa9 as r9  # noqa: E402
import alns_ext as ax  # noqa: E402
from alns import Estado  # noqa: E402
from alns_loop import rodar_alns, construir_politica_alns, qrange  # noqa: E402

D, R = ax.POOLS["P21"]
nomes = sorted(r4.nomes("SYN_TUNE"))
tot = {}
for ni in (nomes[2], nomes[11], nomes[20]):
    inst, cws = r9._instancia("SYN_TUNE", ni)
    px = r9._proxy("SYN_TUNE", ni)
    px.chamadas = 0
    pd_, md = construir_politica_alns("RP", len(D), random.Random(1))
    pr_, mr = construir_politica_alns("RP", len(R), random.Random(2))
    best, _ = rodar_alns(px, [list(r) for r in cws], random.Random(3), 400 * px.n_customers ** 2, pd_, md, pr_, mr, lambda: px.chamadas, w0=0.02, tf_fator=1e-3,
                         max_rotas=inst.n_vehicles_ref if inst.vehicles_exact else None, destruir=D, reparar=R)
    cur_c = px.solution_cost(best)
    dmax = max(max(row) for row in px._D)
    qmin, qmax = qrange(px.n_customers)
    rng = random.Random(5)
    for jd, (nd, dfn) in enumerate(D):
        for jr, (nr, rfn) in enumerate(R):
            for rep in range(12):
                est = Estado(px, best, dmax)
                q = rng.randint(qmin, qmax)
                rem = dfn(est, q, rng) if jd != 3 else dfn(est, q, rng, qmax=qmax)
                n_rem = len(rem)
                ok = rfn(est, rem, rng, inst.n_vehicles_ref if inst.vehicles_exact else None)
                if not ok:
                    continue
                c = px.solution_cost(est.rotas)
                igual = abs(c - cur_c) < 1e-9
                viavel = px.is_feasible(est.rotas)[0]
                t = tot.setdefault(nd, [0, 0, 0, 0])
                t[0] += 1
                t[1] += int(igual)
                t[2] += int(viavel)
                t[3] += n_rem
import csv
linhas = []
print(f"{'destruicao':20s} {'aplicacoes':>10s} {'mesma solucao %':>16s} {'viavel %':>9s} {'clientes removidos (media)':>27s}")
for nd, (n, ig, vi, nr) in tot.items():
    print(f"{nd:20s} {n:10d} {100 * ig / n:16.1f} {100 * vi / n:9.1f} {nr / n:27.1f}")
    linhas.append(dict(destruicao=nd, aplicacoes=n, mesma_solucao_pct=100 * ig / n, viavel_pct=100 * vi / n, clientes_removidos=nr / n))
with open(ROOT / "results" / "aos_study" / "etapa21_diag_noop.csv", "w", newline="", encoding="utf-8") as f_:
    w_ = csv.DictWriter(f_, fieldnames=list(linhas[0]))
    w_.writeheader()
    w_.writerows(linhas)
