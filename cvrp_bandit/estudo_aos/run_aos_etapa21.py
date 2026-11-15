#!/usr/bin/env python3
"""
run_aos_etapa21.py
==================
Estudo 21 da Direcao A (diario, Entrada 136 -- pre-registro): TAMANHO DO CONJUNTO DE OPERADORES no ALNS do CVRP. Os Estudos 12, 18 e 20 mostraram que a mistura
estatica ajustada (7 logits) iguala ou supera os pesos adaptativos de Ropke e Pisinger; o ajuste de uma mistura, porem, fica mais dificil quando o conjunto
cresce, e a camada adaptativa e justamente a que se propoe a dispensar esse ajuste. Aqui o ALNS do Estudo 9 (mesmo laco, mesma aceitacao, mesmo orcamento
de 1.500 n^2 avaliacoes de distancia, mesmas instancias e seeds de avaliacao) roda com tres conjuntos ANINHADOS de operadores (src/alns_ext.py):
P7 (os do Estudo 9; referencia REUTILIZADA, sem nova execucao), P12 (4+3 -> 7+5) e P21 (7+5 -> 12+9 operadores de destruicao + reparo).

Bracos (por conjunto):
  U       selecao uniforme                                     RP     pesos de Ropke e Pisinger, hiperparametros PADRAO (rho 0,1; segmento 100)
  RPt     RP com (rho, segmento) ajustados por busca aleatoria: 33 avaliacoes (a padrao r00 + 32 sorteios)      [P12 e P21]
  S33     mistura estatica (softmax, theta em R^K) ajustada pelo (1+4)-ES de 8 geracoes = 33 avaliacoes         [P12 e P21]
  S65     a mesma busca continuada ate 16 geracoes = 65 avaliacoes                                              [P21]
  HG      HG-UCB com os parametros escolhidos no Estudo 9 (C 1,0; alpha 0,1), sem novo ajuste                    [P21]
Ajuste: SYN_TUNE (24 instancias), seeds 3000-3002 (72 execucoes por avaliacao); validacao (diagnostico) com seeds 3003-3005; avaliacao: X (10 seeds, 1000-1009)
e SYN_VAL (5 seeds, 1000-1004), as mesmas dos Estudos 9, 12 e 20.
Modos: --regressao | --ajuste {static,RP} | --selecionar | --validar | --eval, sempre com --pool P12|P21 (retomaveis por chave).
Uso (WSL): python3 -u run_aos_etapa21.py --pool P21 --ajuste static --workers 11
"""
import argparse
import csv
import json
import multiprocessing as mp
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa18 as r18  # noqa: E402
import run_aos_etapa9 as r9  # noqa: E402
import run_aos_etapa4 as r4  # noqa: E402
import run_aos_etapa20 as r20  # noqa: E402
import alns_ext as ax  # noqa: E402
from alns_loop import rodar_alns  # noqa: E402
from politicas_contexto import softmax_np  # noqa: E402

RES = r18.RES
CAMPOS = r18.CAMPOS_18
SEEDS_TUNE = r20.SEEDS_TUNE
SEEDS_VALID = r20.SEEDS_VALID
SEMENTES_AVAL = 1000
GER_S33, GER_S65 = 8, 16                        # 1 + 4 x 8 = 33 e 1 + 4 x 16 = 65 avaliacoes
SIGMA0 = r18.ES_SIGMA0_STATIC                   # 0,5, como o ES estatico dos Estudos 12, 18 e 20
GER_CURVA = (0, 2, 4, 8, 12, 16)                # incumbentes avaliados com seeds novas (curva de aprendizado do ajuste)
PARAMS_HG = {"C": 1.0, "alpha": 0.1}            # os do Estudo 9 (etapa9_tuning_escolhidas.json)
PADRAO_RP = {"rho": 0.1, "segmento": 100}
POOLS_NOVOS = ("P12", "P21")
PROB21 = {p: dict(destruir=ax.POOLS[p][0], reparar=ax.POOLS[p][1], kappa=r9.KAPPA9, w0=r9.W09, tf=r9.TF9, tune_conj="SYN_TUNE",
                  aval=[("X", 10), ("SYN_VAL", 5)]) for p in ("P7", "P12", "P21")}
SEMENTE_ES = {"P12": 20261401, "P21": 20261402}
SEMENTE_RS = {"P12": 20261411, "P21": 20261412}


# ------------------------------------------------------------------------------------------------ instancias e execucao
def nomes(conj):
    return sorted(r4.nomes(conj))


def spec21(pool, nome, pol, params=None, pesos_d=None, pesos_r=None, config_id="e21"):
    P = PROB21[pool]
    return {"nome": nome, "pol": pol, "params": params or {}, "pesos_d": pesos_d, "pesos_r": pesos_r, "kappa": P["kappa"], "w0": P["w0"],
            "tf": P["tf"], "config_id": config_id, "regime": "a", "problema": "CVRP", "pool": pool}


def executar_21(sp, nivel, conj, nome_inst, seed):
    """Mesmos caminhos de codigo dos Estudos 9, 12 e 20 (sementes de politica seed * 31 + 1 e + 2), com as tabelas de operadores do conjunto `pool`;
    registra o uso por terco do orcamento. Com pool = P7 reproduz exatamente as execucoes desses estudos (`--regressao`)."""
    P = PROB21[sp["pool"]]
    inst, cws = r9._instancia(conj, nome_inst)
    obj = r9._proxy(conj, nome_inst)
    max_rotas = inst.n_vehicles_ref if inst.vehicles_exact else None
    obj.chamadas = 0
    rng = random.Random(seed)
    kd, kr = len(P["destruir"]), len(P["reparar"])
    pol_d, md = r18.fabricar(sp, "d", kd, random.Random(seed * 31 + 1))
    pol_r, mr = r18.fabricar(sp, "r", kr, random.Random(seed * 31 + 2))
    orcamento = int(nivel * sp["kappa"] * inst.n_customers ** 2)
    fase_d = [[0] * kd for _ in range(3)]
    fase_r = [[0] * kr for _ in range(3)]

    def on_it(it, jd, jr, sigma):
        f = min(2, int(3 * obj.chamadas / orcamento))
        fase_d[f][jd] += 1
        fase_r[f][jr] += 1

    t0 = time.perf_counter()
    best, info = rodar_alns(obj, [list(r) for r in cws], rng, orcamento, pol_d, md, pol_r, mr, lambda: obj.chamadas, w0=sp["w0"],
                            tf_fator=sp["tf"], max_rotas=max_rotas, destruir=P["destruir"], reparar=P["reparar"], on_iteracao=on_it)
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(best)
    viavel, _ = inst.is_feasible(best)
    j = lambda v: "|".join(f"{x:g}" for x in v)  # noqa: E731
    jf = lambda m: ";".join(j(v) for v in m)      # noqa: E731
    params = dict(sp["params"])
    if sp["pol"] in ("static", "tilt"):
        params["pesos_d"] = [round(x, 4) for x in sp["pesos_d"]]
        params["pesos_r"] = [round(x, 4) for x in sp["pesos_r"]]
        if sp["pol"] == "static" and "theta" in sp["params"]:
            params = {"pesos_d": params["pesos_d"], "pesos_r": params["pesos_r"], "theta": sp["params"]["theta"]}
    return {"braco": sp["nome"], "config_id": sp["config_id"], "params": json.dumps(params, sort_keys=True), "nivel": nivel, "conjunto": conj,
            "instancia": inst.name, "n_customers": inst.n_customers, "seed": seed, "orcamento": orcamento, "custo": custo, "ref": inst.optimum,
            "gap_pct": 100 * (custo - inst.optimum) / inst.optimum, "viavel": viavel, "n_rotas": len([r for r in best if r]),
            "calls_usadas": obj.chamadas, "iteracoes": info["iteracoes"], "inviaveis": info["inviaveis"], "uso_d": j(info["uso_d"]),
            "uso_r": j(info["uso_r"]), "score_d": j(info["score_d"]), "score_r": j(info["score_r"]), "calls_d": j(info["calls_d"]),
            "calls_r": j(info["calls_r"]), "tempo_s": dt, "uso_d_fase": jf(fase_d), "uso_r_fase": jf(fase_r)}


def _worker21(args):
    return executar_21(*args)


def rodar21(tarefas, out_csv, pool, cada=100):
    feitos = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            feitos = {(r["braco"], r["config_id"], float(r["nivel"]), r["instancia"], int(r["seed"])) for r in csv.DictReader(f)}
    pend = [t for t in tarefas if (t[0]["nome"], t[0]["config_id"], float(t[1]), t[3], t[4]) not in feitos]
    if not pend:
        return 0
    novo = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "a", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=CAMPOS)
        if novo:
            w.writeheader()
        t0, feitas = time.perf_counter(), 0
        for linha in pool.imap_unordered(_worker21, pend, chunksize=1):
            w.writerow(linha)
            feitas += 1
            if feitas % cada == 0:
                f_out.flush()
                print(f"    {feitas}/{len(pend)}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return len(pend)


# ------------------------------------------------------------------------------------------------ ajuste
def tarefas_ajuste(pool, sp, seeds):
    conj = PROB21[pool]["tune_conj"]
    return [(sp, 1, conj, n, s) for n in sorted(nomes(conj), reverse=True) for s in seeds]


def pesos_de_theta(theta, kd):
    th = np.asarray(theta, float)
    return list(map(float, softmax_np(th[:kd]))), list(map(float, softmax_np(th[kd:])))


def pesos_eval(theta, kd):
    """Pesos de AVALIACAO: arredondados a 4 casas, como o braco static dos Estudos 12 e 20."""
    pd_, pr = pesos_de_theta(theta, kd)
    return [round(x, 4) for x in pd_], [round(x, 4) for x in pr]


def csv_ajuste(pool, metodo):
    return RES / f"etapa21_ajuste_{metodo}_{pool}.csv"


def estado_es(pool, sufixo=""):
    return RES / f"etapa21_ajuste_static_{pool}_estado{sufixo}.json"


def modo_ajuste_static(pool, workers):
    """(1+4)-ES (a do Estudo 12/18/20) de theta em R^K, K = kd + kr, a partir de theta = 0 (mistura uniforme), 16 geracoes; o estado da geracao 8 (33
    avaliacoes) e guardado a parte (S33); o final (65 avaliacoes) e S65."""
    P = PROB21[pool]
    kd, kr = len(P["destruir"]), len(P["reparar"])
    out = csv_ajuste(pool, "static")
    nome = f"tune21_{pool}_static"
    n_ref = len(nomes(P["tune_conj"])) * len(SEEDS_TUNE)
    with mp.Pool(workers) as mpool:
        def avaliar(theta, cid):
            pd_, pr = pesos_de_theta(theta, kd)                       # precisao total, como na busca original
            sp = spec21(pool, nome, "static", pesos_d=pd_, pesos_r=pr, config_id=cid)
            sp["params"] = {"theta": list(map(float, theta))}
            rodar21(tarefas_ajuste(pool, sp, SEEDS_TUNE), out, mpool)
            m, n = r18.r16.media_gap(out, nome, cid)
            assert n == n_ref, (pool, cid, n, n_ref)
            return m

        g8 = estado_es(pool, "_gen08")
        if not g8.exists():
            r18.ES_GERACOES_18 = GER_S33
            r18.es18(estado_es(pool), [0.0] * (kd + kr), SIGMA0, SEMENTE_ES[pool], avaliar, f"{pool}/static")
            assert json.loads(estado_es(pool).read_text())["geracao"] == GER_S33
            shutil.copy(estado_es(pool), g8)
        r18.ES_GERACOES_18 = GER_S65
        r18.es18(estado_es(pool), [0.0] * (kd + kr), SIGMA0, SEMENTE_ES[pool], avaliar, f"{pool}/static")


def trajetoria_es(pool):
    """Incumbentes do ES por geracao, reconstruidos do CSV de ajuste: [(geracao, theta, gap de ajuste, n avaliacoes)]."""
    out = csv_ajuste(pool, "static")
    nome = f"tune21_{pool}_static"
    K = len(PROB21[pool]["destruir"]) + len(PROB21[pool]["reparar"])
    import pandas as pd
    d = pd.read_csv(out)
    d = d[(d.braco == nome) & d.viavel]
    g = d.groupby("config_id").agg(gap=("gap_pct", "mean"), params=("params", "first"), n=("gap_pct", "size"))
    assert (g.n == len(nomes(PROB21[pool]["tune_conj"])) * len(SEEDS_TUNE)).all()
    traj = []
    atual_theta, atual_gap = json.loads(g.loc["gen00", "params"])["theta"], float(g.loc["gen00", "gap"])
    traj.append((0, atual_theta, atual_gap, 1))
    geracoes = sorted({int(c[3:5]) for c in g.index if c.startswith("gen") and "c" in c[3:]})
    for gen in geracoes:
        filhos = [(float(g.loc[c, "gap"]), c) for c in g.index if c.startswith(f"gen{gen:02d}c")]
        if len(filhos) < 4:
            break
        gap_f, cid = min(filhos)
        if gap_f < atual_gap:
            atual_gap, atual_theta = gap_f, json.loads(g.loc[cid, "params"])["theta"]
        traj.append((gen, atual_theta, atual_gap, 1 + 4 * gen))
    assert all(len(t[1]) == K for t in traj)
    return traj


def configs_rs(pool):
    """33 configuracoes de RP: a padrao (r00) e 32 sorteios do espaco do Estudo 18."""
    rng = random.Random(SEMENTE_RS[pool])
    cfgs = [("r00", dict(PADRAO_RP))]
    for i in range(1, r18.N_AVAL):
        cfgs.append((f"r{i:02d}", r18.espaco("RP18", rng)))
    return cfgs


def modo_ajuste_RP(pool, workers):
    out = csv_ajuste(pool, "RP")
    nome = f"tune21_{pool}_RP"
    tar = []
    for cid, params in configs_rs(pool):
        tar += tarefas_ajuste(pool, spec21(pool, nome, "RP", params=params, config_id=cid), SEEDS_TUNE)
    print(f"ajuste {pool}/RP: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as mpool:
        rodar21(tar, out, mpool)
    print(f"ajuste {pool}/RP concluido", flush=True)


def modo_selecionar(pool):
    P = PROB21[pool]
    kd = len(P["destruir"])
    traj = trajetoria_es(pool)
    por_ger = {t[0]: t for t in traj}
    esc = {"trajetoria": [{"geracao": g, "n_avaliacoes": n, "gap_ajuste": gp, "theta": th} for g, th, gp, n in traj]}
    t33, t65 = por_ger[GER_S33], por_ger[GER_S65]
    g8 = json.loads(estado_es(pool, "_gen08").read_text())
    g16 = json.loads(estado_es(pool).read_text())
    assert g8["theta"] == t33[1] and abs(g8["melhor"] - t33[2]) < 1e-9 and g16["theta"] == t65[1] and abs(g16["melhor"] - t65[2]) < 1e-9, \
        "a trajetoria reconstruida do CSV nao coincide com os estados do ES"
    pd33, pr33 = pesos_eval(t33[1], kd)
    pd65, pr65 = pesos_eval(t65[1], kd)
    esc["S33"] = {"theta": t33[1], "pesos_d": pd33, "pesos_r": pr33, "gap_ajuste": t33[2], "n_avaliacoes": t33[3]}
    esc["S65"] = {"mudou": bool(t65[1] != t33[1]), "theta": t65[1], "pesos_d": pd65, "pesos_r": pr65, "gap_ajuste": t65[2], "n_avaliacoes": t65[3]}
    lst = r18.melhor_por_config(csv_ajuste(pool, "RP"), f"tune21_{pool}_RP")
    padrao = [x for x in lst if x[0] == "r00"][0]
    t = lst[0] if lst[0][2] < padrao[2] - 1e-12 else padrao                  # empate ou pior que o padrao -> padrao
    esc["RPt"] = {"mudou": t[0] != "r00", "params": json.loads(t[1]), "gap_ajuste": t[2], "gap_padrao": padrao[2], "config_id": t[0], "n_configs": len(lst)}
    (RES / f"etapa21_ajuste_escolhidas_{pool}.json").write_text(json.dumps(esc, indent=1))
    print(f"{pool}: S33 gap de ajuste {t33[2]:.4f} ({t33[3]} aval.)  S65 {t65[2]:.4f} ({t65[3]} aval.; {'MUDOU' if esc['S65']['mudou'] else 'sem melhora'})")
    print(f"{pool}: RP padrao {padrao[2]:.4f}  RPt {t[2]:.4f} (config {t[0]}, {esc['RPt']['params']})")
    print(f"{pool}: curva do ES (gap de ajuste do incumbente): " + ", ".join(f"g{g}:{gp:.3f}" for g, _th, gp, _n in traj))


def specs_eval(pool):
    esc = json.loads((RES / f"etapa21_ajuste_escolhidas_{pool}.json").read_text())
    p = lambda a: f"s21_{pool}_{a}"  # noqa: E731
    sp = {p("U"): spec21(pool, p("U"), "U"), p("RP"): spec21(pool, p("RP"), "RP", params=dict(PADRAO_RP)),
          p("S33"): spec21(pool, p("S33"), "static", pesos_d=esc["S33"]["pesos_d"], pesos_r=esc["S33"]["pesos_r"])}
    if esc["RPt"]["mudou"]:
        sp[p("RPt")] = spec21(pool, p("RPt"), "RP", params=esc["RPt"]["params"])
    if pool == "P21":
        if esc["S65"]["mudou"]:
            sp[p("S65")] = spec21(pool, p("S65"), "static", pesos_d=esc["S65"]["pesos_d"], pesos_r=esc["S65"]["pesos_r"])
        sp[p("HG")] = spec21(pool, p("HG"), "hgucb", params=dict(PARAMS_HG))
    return sp


def modo_validar(pool, workers):
    """Diagnostico (seeds 3003-3005, instancias de ajuste): os bracos da avaliacao e os incumbentes do ES por geracao (curva de aprendizado sem o vies
    de selecao do proprio ajuste)."""
    P = PROB21[pool]
    kd = len(P["destruir"])
    esc = json.loads((RES / f"etapa21_ajuste_escolhidas_{pool}.json").read_text())
    sp = dict(specs_eval(pool))
    sp[f"s21_{pool}_RPt"] = sp.get(f"s21_{pool}_RPt") or spec21(pool, f"s21_{pool}_RPt", "RP", params=esc["RPt"]["params"])
    if pool == "P21":
        sp[f"s21_{pool}_S65"] = sp.get(f"s21_{pool}_S65") or spec21(pool, f"s21_{pool}_S65", "static", pesos_d=esc["S65"]["pesos_d"],
                                                                      pesos_r=esc["S65"]["pesos_r"])
    for t in esc["trajetoria"]:
        if t["geracao"] in GER_CURVA:
            pd_, pr = pesos_de_theta(t["theta"], kd)
            nome = f"val21_{pool}_gen{t['geracao']:02d}"
            sp[nome] = spec21(pool, nome, "static", pesos_d=pd_, pesos_r=pr)
    tar = [t for s in sp.values() for t in tarefas_ajuste(pool, s, SEEDS_VALID)]
    print(f"validacao {pool} (instancias de ajuste, seeds novas): {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as mpool:
        rodar21(tar, RES / f"etapa21_validacao_{pool}.csv", mpool)
    import pandas as pd
    d = pd.read_csv(RES / f"etapa21_validacao_{pool}.csv")
    print(d[d.viavel].groupby("braco").gap_pct.mean().sort_values().round(4).to_string())


def instancias_aval():
    out = [(c, n, ns, r9._instancia(c, n)[0].n_customers) for c, ns in PROB21["P21"]["aval"] for n in nomes(c)]
    return sorted(out, key=lambda t: -t[3])                                  # as maiores primeiro (balanceia a carga)


def modo_eval(pool, workers):
    sp = specs_eval(pool)
    tar = [(s, 1, c, n, SEMENTES_AVAL + sd) for c, n, ns, _nc in instancias_aval() for s in sp.values() for sd in range(ns)]
    print(f"eval Estudo 21 {pool}: {len(tar)} execucoes ({', '.join(sp)})", flush=True)
    with mp.Pool(workers) as mpool:
        rodar21(tar, RES / f"etapa21_eval_{pool}.csv", mpool)
    print(f"eval Estudo 21 {pool} concluido", flush=True)


# ------------------------------------------------------------------------------------------------ regressao exata e identidades
CAMPOS_REG = ("iteracoes", "calls_usadas", "inviaveis", "n_rotas", "uso_d", "uso_r", "score_d", "score_r", "calls_d", "calls_r")


def modo_regressao(workers):
    """(1) O executor reproduz EXATAMENTE (custo, iteracoes, avaliacoes, usos, escores) execucoes dos Estudos 9 (U, RP, HG-UCB), 12 (mistura S33) e 20
    (RP ajustado, mistura de 65 avaliacoes) com a tabela P7. (2) Identidade de imersao: a mistura S33 do Estudo 12, estendida com peso 0 nos operadores
    novos, reproduz a execucao P7 dentro de P12 e de P21 (os operadores novos nunca sao sorteados e nenhum numero aleatorio muda). (3) Os conjuntos
    de instancias e de seeds de avaliacao coincidem com os dos CSVs de referencia."""
    import pandas as pd
    rs = random.Random(21)
    e9 = pd.read_csv(RES / "etapa9_eval.csv")
    e12 = pd.read_csv(RES / "etapa12_eval.csv")
    e20 = pd.read_csv(RES / "etapa20_eval_CVRP.csv")
    esc20 = json.loads((RES / "etapa20_tuning_escolhidas_CVRP.json").read_text())
    esc9 = json.loads((RES / "etapa9_tuning_escolhidas.json").read_text())
    assert esc9["hgucb"]["params"] == PARAMS_HG and esc9["RP"]["params"] == PADRAO_RP, "parametros de referencia mudaram"
    pd33, pr33 = r20.static33("CVRP")
    for c, ns in PROB21["P21"]["aval"]:                                         # (3)
        assert sorted(e9[e9.conjunto == c].instancia.unique()) == nomes(c), c
        assert int(e9[(e9.conjunto == c) & (e9.braco == "s9_U")].groupby("instancia").seed.nunique().min()) == ns, (c, ns)
    fontes = [("U", spec21("P7", "reg_U", "U"), e9, "s9_U"), ("RP", spec21("P7", "reg_RP", "RP", params=dict(PADRAO_RP)), e9, "s9_RP"),
              ("HG", spec21("P7", "reg_HG", "hgucb", params=dict(PARAMS_HG)), e9, "s9_hgucb"),
              ("S33", spec21("P7", "reg_S33", "static", pesos_d=pd33, pesos_r=pr33), e12, "s12_static"),
              ("RPt20", spec21("P7", "reg_RPt", "RP", params=esc20["RP"]["params"]), e20, "s20_CVRP_RP"),
              ("S65", spec21("P7", "reg_S65", "static", pesos_d=esc20["static65"]["pesos_d"], pesos_r=esc20["static65"]["pesos_r"]), e20, "s20_CVRP_static65")]
    casos = []
    todas = sorted(e9.instancia.unique())
    for arm, s, fonte, nome_braco in fontes:
        for i in rs.sample(todas, 3):
            ref = fonte[(fonte.braco == nome_braco) & (fonte.instancia == i)]
            casos.append((arm, s, ref.iloc[rs.randrange(len(ref))]))
    tar = [(s, 1, lin.conjunto, lin.instancia, int(lin.seed)) for (_a, s, lin) in casos]
    with mp.Pool(min(workers, len(tar))) as mpool:
        novas = mpool.map(_worker21, tar)
    for (arm, s, lin), nv in zip(casos, novas):
        assert nv["instancia"] == lin.instancia, (arm, nv["instancia"], lin.instancia)
        for campo in CAMPOS_REG:
            assert str(nv[campo]) == str(lin[campo]), (arm, lin.instancia, lin.seed, campo, nv[campo], lin[campo])
        assert abs(nv["custo"] - lin["custo"]) < 1e-9 and bool(nv["viavel"]) == bool(lin["viavel"]), (arm, lin.instancia, lin.seed)
    print(f"REGRESSAO EXATA: OK ({len(casos)} execucoes dos Estudos 9, 12 e 20 reproduzidas com a tabela P7: custo, iteracoes, avaliacoes, usos e escores)", flush=True)
    for pool in POOLS_NOVOS:                                                   # (2)
        P = PROB21[pool]
        kd, kr = len(P["destruir"]), len(P["reparar"])
        n0 = sorted(nomes("SYN_TUNE"), key=lambda n: r9._instancia("SYN_TUNE", n)[0].n_customers)[0]
        a = executar_21(spec21("P7", "i_p7", "static", pesos_d=pd33, pesos_r=pr33), 0.15, "SYN_TUNE", n0, 5)
        b = executar_21(spec21(pool, "i_pool", "static", pesos_d=list(pd33) + [0.0] * (kd - 4), pesos_r=list(pr33) + [0.0] * (kr - 3)), 0.15,
                        "SYN_TUNE", n0, 5)
        for campo in ("iteracoes", "calls_usadas", "inviaveis", "n_rotas", "custo"):
            assert str(a[campo]) == str(b[campo]), (pool, campo, a[campo], b[campo])
        corta = lambda s, n: "|".join(s.split("|")[:n])  # noqa: E731
        for campo, n in (("uso_d", 4), ("score_d", 4), ("calls_d", 4), ("uso_r", 3), ("score_r", 3), ("calls_r", 3)):
            assert corta(b[campo], n) == a[campo], (pool, campo, a[campo], b[campo])      # os quatro (tres) primeiros operadores coincidem
            assert all(float(x) == 0 for x in b[campo].split("|")[n:]), (pool, campo, b[campo])    # os novos nunca foram sorteados
        print(f"IDENTIDADE DE IMERSAO {pool}: mistura S33 com peso 0 nos operadores novos == execucao P7 (custo {a['custo']:.4f}, {a['iteracoes']} iteracoes)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("regressao", "selecionar", "validar", "eval"):
        g.add_argument(f"--{m}", action="store_true")
    g.add_argument("--ajuste", choices=("static", "RP"))
    ap.add_argument("--pool", choices=POOLS_NOVOS)
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.regressao:
        return modo_regressao(a.workers)
    assert a.pool, "--pool P12|P21 e obrigatorio neste modo"
    if a.selecionar:
        return modo_selecionar(a.pool)
    if a.ajuste == "static":
        return modo_ajuste_static(a.pool, a.workers)
    if a.ajuste == "RP":
        return modo_ajuste_RP(a.pool, a.workers)
    if a.validar:
        return modo_validar(a.pool, a.workers)
    return modo_eval(a.pool, a.workers)


if __name__ == "__main__":
    main()
