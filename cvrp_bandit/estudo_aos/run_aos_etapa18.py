#!/usr/bin/env python3
"""
run_aos_etapa18.py
==================
Estudo 18 da Direcao A (diario, Entrada 127 -- pre-registro): POLITICAS COM ESTADO e ESFORCO DE AJUSTE IGUAL no regime ALNS do CVRPTW
(o do Estudo 16: mesmas 4 destruicoes x 3 reparos, w0 = 0,2, kappa = 1.500, as 56 instancias de Solomon e as seeds 1000-1009; os bracos do
Estudo 16 sao REUTILIZADOS como referencias, sem nova execucao -- o controle `--regressao` prova que o codigo novo os reproduz exatamente).

Bracos novos (8): static18 (mistura estatica, ES), RP18 (Ropke e Pisinger com ajuste igual), tilt (RP sobre o prior static18), sched
(mistura por fase, ES a partir de static18), loglin (mistura log-linear no estado, ES a partir de static18), ctxucb, linucb, qlearn.
ESFORCO DE AJUSTE IGUAL: 33 avaliacoes por metodo em SYN_TUNE_W + SYN_TUNE_W2 (48 instancias, seeds 4000-4001; cada avaliacao = 96 execucoes):
ES (1+4) de 8 geracoes (o do Estudo 16) para as misturas, busca aleatoria (configuracao padrao + 32 sorteios, semente por metodo) para as demais;
sched e loglin recebem MAIS 33 avaliacoes a partir de static18 (vantagem declarada aos modelos mais ricos).

Modos: --regressao | --tuning METODO | --selecionar | --validar | --eval        (todos retomaveis: pulam o que ja esta no CSV)
Uso (WSL, venv de direcao1_local): python3 -u run_aos_etapa18.py --tuning static18 --workers 11
"""
import argparse
import csv
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa16 as r16  # noqa: E402
import run_aos_etapa14 as r14  # noqa: E402
from tw_alns import EstadoTW, DESTRUIR_TW, REPARAR_TW  # noqa: E402
from aos_vnd import EstaticaPolitica  # noqa: E402
from alns_loop import rodar_alns, construir_politica_alns  # noqa: E402
from politicas_contexto import (CtxUCBPolitica, LinUCBPolitica, QLearnPolitica, RPTiltPolitica, FasesMistPolitica,  # noqa: E402
                                LogLinMistPolitica, softmax_np)

RES = ROOT / "results" / "aos_study"
CAMPOS_18 = r16.CAMPOS_A + ["uso_d_fase", "uso_r_fase"]
CONJ_TUNE = ("SYN_TUNE_W", "SYN_TUNE_W2")
SEEDS_TUNE_18 = [4000, 4001]
SEEDS_VALID_18 = [4002, 4003]
N_AVAL = 33                       # avaliacoes de ajuste por metodo (a do Estudo 16: 1 + 8 geracoes x 4 filhos)
ES_FILHOS_18, ES_GERACOES_18 = 4, 8
ES_SIGMA0_STATIC, ES_SIGMA0_RICO = 0.5, 0.3
SEMENTE_ES = {"static18": 20261120, "sched": 20261121, "loglin": 20261122}
SEMENTE_RS = {"RP18": 20261131, "tilt": 20261132, "ctxucb": 20261133, "linucb": 20261134, "qlearn": 20261135}
METODOS_ES = ("static18", "sched", "loglin")
METODOS_RS = ("RP18", "tilt", "ctxucb", "linucb", "qlearn")
BRACOS_18 = ("static18", "RP18", "tilt", "sched", "loglin", "ctxucb", "linucb", "qlearn")
SEGMENTOS = (25, 50, 100, 200, 400)
TAUS = (25, 50, 100, 200, 400)
NFASES = (2, 3, 4)


# ------------------------------------------------------------------------------------------------ instancias
def conjunto_tuning():
    return [(c, n) for c in CONJ_TUNE for n in sorted(r14.nomes(c), reverse=True)]


# ------------------------------------------------------------------------------------------------ politicas
def fabricar(sp, comp, k, rng):
    """(politica, modo) do componente `comp` ('d' destruicao, 'r' reparo). Bracos do Estudo 16: os mesmos caminhos de codigo."""
    pol, p = sp["pol"], sp["params"]
    if pol == "static":
        return EstaticaPolitica(sp["pesos_" + comp], rng=rng), "score"
    if pol == "ctxucb":
        return CtxUCBPolitica(k, rng, C=p["C"], tau=p["tau"], n_fases=p["n_fases"]), "score"
    if pol == "linucb":
        return LinUCBPolitica(k, rng, alpha=p["alpha"], lam=p["lam"]), "score"
    if pol == "qlearn":
        return QLearnPolitica(k, rng, alpha=p["alpha"], gamma=p["gamma"], eps0=p["eps0"], tau=p["tau"], n_fases=p["n_fases"]), "score"
    if pol == "tilt":
        return RPTiltPolitica(k, rng, sp["pesos_" + comp], segmento=p["segmento"], rho=p["rho"]), "raw"
    if pol == "sched":
        return FasesMistPolitica(k, rng, p["logits_" + comp]), "score"
    if pol == "loglin":
        return LogLinMistPolitica(k, rng, p["theta_" + comp], p["a_" + comp], p["b_" + comp]), "score"
    return construir_politica_alns(pol, k, rng, p)


def executar_18(sp, nivel, conj, nome_inst, seed):
    """Copia de run_aos_etapa16.executar_a com a fabrica ampliada e o uso por fase (tercos do orcamento); os bracos do Estudo 16 passam
    pelos mesmos caminhos (mesmas sementes de politica: seed * 31 + 1 e + 2)."""
    inst, x0 = r16.carregar(conj, nome_inst)
    inst.chamadas = 0
    rng = random.Random(seed)
    pol_d, md = fabricar(sp, "d", len(DESTRUIR_TW), random.Random(seed * 31 + 1))
    pol_r, mr = fabricar(sp, "r", len(REPARAR_TW), random.Random(seed * 31 + 2))
    orcamento = int(nivel * sp["kappa"] * inst.n_customers ** 2)
    kd, kr = len(DESTRUIR_TW), len(REPARAR_TW)
    fase_d = [[0] * kd for _ in range(3)]
    fase_r = [[0] * kr for _ in range(3)]

    def on_it(it, jd, jr, sigma):
        f = min(2, int(3 * inst.chamadas / orcamento))
        fase_d[f][jd] += 1
        fase_r[f][jr] += 1

    t0 = time.perf_counter()
    best, info = rodar_alns(inst, [list(r) for r in x0], rng, orcamento, pol_d, md, pol_r, mr, lambda: inst.chamadas,
                            w0=sp["w0"], tf_fator=sp["tf"], max_rotas=inst.nveic, destruir=DESTRUIR_TW, reparar=REPARAR_TW,
                            estado_cls=EstadoTW, on_iteracao=on_it)
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
    return {"braco": sp["nome"], "config_id": sp["config_id"], "params": json.dumps(params, sort_keys=True),
            "nivel": nivel, "conjunto": conj, "instancia": inst.nome, "n_customers": inst.n_customers,
            "seed": seed, "orcamento": orcamento, "custo": custo, "ref": inst.ref,
            "gap_pct": 100 * (custo - inst.ref) / inst.ref, "viavel": viavel,
            "n_rotas": len([r for r in best if r]), "calls_usadas": inst.chamadas, "iteracoes": info["iteracoes"],
            "inviaveis": info["inviaveis"], "uso_d": j(info["uso_d"]), "uso_r": j(info["uso_r"]),
            "score_d": j(info["score_d"]), "score_r": j(info["score_r"]), "calls_d": j(info["calls_d"]),
            "calls_r": j(info["calls_r"]), "tempo_s": dt, "uso_d_fase": jf(fase_d), "uso_r_fase": jf(fase_r)}


def _worker18(args):
    sp, nivel, conj, nome, seed = args
    return executar_18(sp, nivel, conj, nome, seed)


def rodar18(tarefas, out_csv, pool, cada=100):
    feitos = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            feitos = {r16.chave(r) for r in csv.DictReader(f)}
    pend = [t for t in tarefas if (t[0]["nome"], t[0]["config_id"], float(t[1]), t[3], t[4]) not in feitos]
    if not pend:
        return 0
    novo = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "a", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=CAMPOS_18)
        if novo:
            w.writeheader()
        t0, feitas = time.perf_counter(), 0
        for linha in pool.imap_unordered(_worker18, pend, chunksize=1):
            w.writerow(linha)
            feitas += 1
            if feitas % cada == 0:
                f_out.flush()
                print(f"    {feitas}/{len(pend)}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return len(pend)


def spec18(nome, pol, params=None, pesos_d=None, pesos_r=None, config_id="e18"):
    return r16.spec_a(nome, pol, params=params, pesos_d=pesos_d, pesos_r=pesos_r, config_id=config_id)


# ------------------------------------------------------------------------------------------------ parametrizacoes das buscas
def pesos_de_theta(theta):
    th = np.asarray(theta, float)
    return softmax_np(th[:4]), softmax_np(th[4:7])


def params_sched(theta, nf=3):
    th = list(map(float, theta))
    return {"theta": th, "logits_d": [th[7 * f:7 * f + 4] for f in range(nf)], "logits_r": [th[7 * f + 4:7 * f + 7] for f in range(nf)]}


def params_loglin(theta):
    th = list(map(float, theta))
    return {"theta": th, "theta_d": th[0:4], "theta_r": th[4:7], "a_d": th[7:11], "a_r": th[11:14], "b_d": th[14:18], "b_r": th[18:21]}


def spec_es(metodo, theta, cid):
    nome = f"tune18_{metodo}"
    if metodo == "static18":
        pd_, pr = pesos_de_theta(theta)
        sp = spec18(nome, "static", pesos_d=pd_, pesos_r=pr, config_id=cid)
        sp["params"] = {"theta": list(map(float, theta))}
        return sp
    if metodo == "sched":
        return spec18(nome, "sched", params=params_sched(theta), config_id=cid)
    if metodo == "loglin":
        return spec18(nome, "loglin", params=params_loglin(theta), config_id=cid)
    raise ValueError(metodo)


def espaco(metodo, rng):
    """Um sorteio do espaco de hiperparametros do metodo (busca aleatoria); valores arredondados a 4 algarismos significativos."""
    r4 = lambda x: float(f"{x:.4g}")  # noqa: E731
    lu = lambda a, b: r4(math.exp(rng.uniform(math.log(a), math.log(b))))  # noqa: E731
    if metodo in ("RP18", "tilt"):
        return {"rho": lu(0.02, 0.5), "segmento": rng.choice(SEGMENTOS)}
    if metodo == "ctxucb":
        return {"C": lu(0.05, 4.0), "tau": rng.choice(TAUS), "n_fases": rng.choice(NFASES)}
    if metodo == "linucb":
        return {"alpha": lu(0.02, 2.0), "lam": lu(0.1, 10.0)}
    if metodo == "qlearn":
        return {"alpha": lu(0.02, 0.5), "gamma": r4(rng.uniform(0.0, 0.95)), "eps0": r4(rng.uniform(0.05, 0.5)),
                "tau": rng.choice(TAUS), "n_fases": rng.choice(NFASES)}
    raise ValueError(metodo)


PADRAO_RS = {"RP18": {"rho": 0.1, "segmento": 100}, "tilt": {"rho": 0.1, "segmento": 100},
             "ctxucb": {"C": 1.0, "tau": 100, "n_fases": 3}, "linucb": {"alpha": 0.5, "lam": 1.0},
             "qlearn": {"alpha": 0.1, "gamma": 0.5, "eps0": 0.2, "tau": 100, "n_fases": 3}}
POL_RS = {"RP18": "RP", "tilt": "tilt", "ctxucb": "ctxucb", "linucb": "linucb", "qlearn": "qlearn"}


def configs_rs(metodo):
    """33 configuracoes: a padrao (r00) e 32 sorteios (semente por metodo): lista de (config_id, params)."""
    rng = random.Random(SEMENTE_RS[metodo])
    cfgs = [("r00", dict(PADRAO_RS[metodo]))]
    for i in range(1, N_AVAL):
        cfgs.append((f"r{i:02d}", espaco(metodo, rng)))
    return cfgs


def melhor_por_config(csv_path, braco):
    """(config_id, params(json), gap medio, n) em ordem crescente de gap medio (so execucoes viaveis)."""
    import pandas as pd
    d = pd.read_csv(csv_path)
    d = d[(d.braco == braco) & d.viavel]
    g = d.groupby(["config_id", "params"]).gap_pct.agg(["mean", "count"]).reset_index().sort_values("mean")
    return [(r.config_id, r.params, float(r["mean"]), int(r["count"])) for _, r in g.iterrows()]


def melhor_static18():
    top = melhor_por_config(RES / "etapa18_tuning_static18.csv", "tune18_static18")[0]
    th = json.loads(top[1])["theta"]
    return th, top


# ------------------------------------------------------------------------------------------------ ajuste
def es18(estado_p, theta0, sigma0, base_seed, avaliar, rotulo):
    """(1+4)-ES de 8 geracoes (a do Estudo 16; regra do sigma: x0,9 se nenhum filho melhora). Retomavel e determinista."""
    k = len(theta0)
    estado = (json.loads(estado_p.read_text()) if estado_p.exists()
              else {"geracao": 0, "theta": list(map(float, theta0)), "sigma": sigma0, "melhor": None})
    if estado["melhor"] is None:
        estado["melhor"] = avaliar(estado["theta"], "gen00")
        estado_p.write_text(json.dumps(estado))
        print(f"{rotulo} gen 0: gap {estado['melhor']:.4f}", flush=True)
    while estado["geracao"] < ES_GERACOES_18:
        g = estado["geracao"] + 1
        rng = np.random.default_rng(base_seed + g)
        filhos = [np.array(estado["theta"]) + estado["sigma"] * rng.normal(size=k) for _ in range(ES_FILHOS_18)]
        gaps = [avaliar(th, f"gen{g:02d}c{i}") for i, th in enumerate(filhos)]
        i = int(np.argmin(gaps))
        if gaps[i] < estado["melhor"]:
            estado["theta"], estado["melhor"] = filhos[i].tolist(), gaps[i]
        else:
            estado["sigma"] *= 0.9
        estado["geracao"] = g
        estado_p.write_text(json.dumps(estado))
        print(f"{rotulo} gen {g}: melhor filho {min(gaps):.4f}  atual {estado['melhor']:.4f}  sigma {estado['sigma']:.3f}", flush=True)
    print(f"busca ES ({rotulo}) concluida", flush=True)


def modo_tuning(metodo, workers):
    insts = conjunto_tuning()
    out = RES / f"etapa18_tuning_{metodo}.csv"
    n_ref = len(insts) * len(SEEDS_TUNE_18)

    def tarefas(sp):
        return [(sp, 1, c, n, s) for c, n in insts for s in SEEDS_TUNE_18]

    with mp.Pool(workers) as pool:
        if metodo in METODOS_ES:
            if metodo == "static18":
                theta0, sigma0 = [0.0] * 7, ES_SIGMA0_STATIC
            else:
                th_s, _ = melhor_static18()
                theta0 = th_s * 3 if metodo == "sched" else th_s + [0.0] * 14
                sigma0 = ES_SIGMA0_RICO

            def avaliar(theta, cid):
                sp = spec_es(metodo, theta, cid)
                rodar18(tarefas(sp), out, pool)
                m, n = r16.media_gap(out, sp["nome"], cid)
                assert n == n_ref, (metodo, cid, n, n_ref)
                return m

            es18(RES / f"etapa18_tuning_{metodo}_estado.json", theta0, sigma0, SEMENTE_ES[metodo], avaliar, metodo)
        else:
            cfgs = configs_rs(metodo)
            extra = {}
            if metodo == "tilt":
                th_s, _ = melhor_static18()
                extra["pesos_d"], extra["pesos_r"] = pesos_de_theta(th_s)
            tar = []
            for cid, params in cfgs:
                sp = spec18(f"tune18_{metodo}", POL_RS[metodo], params=params, config_id=cid, **extra)
                tar += tarefas(sp)
            print(f"ajuste {metodo}: {len(cfgs)} configuracoes, {len(tar)} execucoes", flush=True)
            rodar18(tar, out, pool)
    print(f"ajuste {metodo} concluido", flush=True)


def modo_selecionar():
    esc = {}
    th_s, top = melhor_static18()
    pd_, pr = pesos_de_theta(th_s)
    ini = [x for x in melhor_por_config(RES / "etapa18_tuning_static18.csv", "tune18_static18") if x[0] == "gen00"][0]
    esc["static18"] = {"theta": th_s, "pesos_d": pd_, "pesos_r": pr, "gap_tune": top[2], "gap_inicio": ini[2], "config_id": top[0],
                       "n_configs": len(melhor_por_config(RES / "etapa18_tuning_static18.csv", "tune18_static18"))}
    for m, fp in (("sched", params_sched), ("loglin", params_loglin)):
        p = RES / f"etapa18_tuning_{m}.csv"
        if not p.exists():
            continue
        lst = melhor_por_config(p, f"tune18_{m}")
        th = json.loads(lst[0][1])["theta"]
        ini = [x for x in lst if x[0] == "gen00"][0]
        esc[m] = {**fp(th), "gap_tune": lst[0][2], "gap_inicio": ini[2], "config_id": lst[0][0], "n_configs": len(lst)}
    for m in METODOS_RS:
        p = RES / f"etapa18_tuning_{m}.csv"
        if not p.exists():
            continue
        lst = melhor_por_config(p, f"tune18_{m}")
        padrao = [x for x in lst if x[0] == "r00"][0]
        top = lst[0] if lst[0][2] < padrao[2] - 1e-12 else padrao                  # empate ou pior que o padrao -> padrao
        esc[m] = {"params": {k: v for k, v in json.loads(top[1]).items() if k not in ("pesos_d", "pesos_r")}, "gap_tune": top[2],
                  "gap_padrao": padrao[2], "config_id": top[0], "n_configs": len(lst)}
        if m == "tilt":
            esc[m]["pesos_d"], esc[m]["pesos_r"] = pd_, pr
    (RES / "etapa18_tuning_escolhidas.json").write_text(json.dumps(esc, indent=1))
    for m, v in esc.items():
        print(f"{m}: config {v['config_id']}  gap_tune {v['gap_tune']:.4f}  ({'inicio' if 'gap_inicio' in v else 'padrao'} "
              f"{v.get('gap_inicio', v.get('gap_padrao')):.4f})  n_configs {v['n_configs']}")


def specs_eval_18():
    esc = json.loads((RES / "etapa18_tuning_escolhidas.json").read_text())
    sp = {"s18_static18": spec18("s18_static18", "static", pesos_d=esc["static18"]["pesos_d"], pesos_r=esc["static18"]["pesos_r"]),
          "s18_RP18": spec18("s18_RP18", "RP", params=esc["RP18"]["params"]),
          "s18_tilt": spec18("s18_tilt", "tilt", params=esc["tilt"]["params"], pesos_d=esc["tilt"]["pesos_d"], pesos_r=esc["tilt"]["pesos_r"]),
          "s18_sched": spec18("s18_sched", "sched", params=params_sched(esc["sched"]["theta"])),
          "s18_loglin": spec18("s18_loglin", "loglin", params=params_loglin(esc["loglin"]["theta"]))}
    for m in ("ctxucb", "linucb", "qlearn"):
        sp[f"s18_{m}"] = spec18(f"s18_{m}", m, params=esc[m]["params"])
    return sp


def modo_validar(workers):
    sp = specs_eval_18()
    tar = [(s, 1, c, n, sd) for c, n in conjunto_tuning() for s in sp.values() for sd in SEEDS_VALID_18]
    print(f"validacao (instancias de ajuste, seeds novas): {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar18(tar, RES / "etapa18_validacao.csv", pool)
    import pandas as pd
    d = pd.read_csv(RES / "etapa18_validacao.csv")
    d = d[d.viavel]
    print(d.groupby("braco").gap_pct.mean().sort_values().round(4).to_string())


def instancias_aval():
    return [("SOLOMON", n, 10) for n in r14.nomes("SOLOMON")]


def modo_eval(workers):
    sp = specs_eval_18()
    tar = [(s, 1, c, n, r16.SEMENTES_AVAL + sd) for c, n, ns in instancias_aval() for s in sp.values() for sd in range(ns)]
    tar.sort(key=lambda t: t[3])
    print(f"eval Estudo 18: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar18(tar, RES / "etapa18_eval.csv", pool)
    print("eval Estudo 18 concluido", flush=True)


# ------------------------------------------------------------------------------------------------ regressao exata
def modo_regressao(workers):
    """O codigo do Estudo 18 (fabrica ampliada, gancho de contexto, uso por fase) reproduz EXATAMENTE as execucoes do Estudo 16
    (custo, iteracoes, avaliacoes de distancia, usos e escores de cada operador) para os bracos U, RP, hgucb, static e fixrep."""
    import pandas as pd
    e = pd.read_csv(RES / "etapa16_eval_a.csv")
    sp16 = r16.specs_eval_a()
    rs = random.Random(5)
    insts = sorted(e.instancia.unique())
    alvo = [(a, i, 1000 + rs.randrange(10)) for a in ("s16a_U", "s16a_RP", "s16a_hgucb", "s16a_static", "s16a_fixrep", "s16a_ts")
            for i in rs.sample(insts, 3)]
    tar = [(sp16[a], 1, "SOLOMON", i, s) for a, i, s in alvo]
    with mp.Pool(min(workers, len(tar))) as pool:
        novas = pool.map(_worker18, tar)
    n_ok = 0
    for (a, i, s), nv in zip(alvo, novas):
        ref = e[(e.braco == a) & (e.instancia == i) & (e.seed == s)]
        assert len(ref) == 1, (a, i, s)
        ref = ref.iloc[0]
        for campo in ("iteracoes", "calls_usadas", "inviaveis", "n_rotas", "uso_d", "uso_r", "score_d", "score_r", "calls_d", "calls_r"):
            assert str(nv[campo]) == str(ref[campo]), (a, i, s, campo, nv[campo], ref[campo])
        assert abs(nv["custo"] - ref["custo"]) < 1e-9 and bool(nv["viavel"]) == bool(ref["viavel"]), (a, i, s, nv["custo"], ref["custo"])
        n_ok += 1
    print(f"REGRESSAO EXATA: OK ({n_ok} execucoes do Estudo 16 reproduzidas: custo, iteracoes, avaliacoes, usos e escores)")


MODOS_SIMPLES = {"regressao": modo_regressao, "validar": modo_validar, "eval": modo_eval}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in MODOS_SIMPLES:
        g.add_argument(f"--{m}", action="store_true")
    g.add_argument("--selecionar", action="store_true")
    g.add_argument("--tuning", choices=list(METODOS_ES) + list(METODOS_RS))
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.selecionar:
        modo_selecionar()
    elif a.tuning:
        modo_tuning(a.tuning, a.workers)
    else:
        for m, f in MODOS_SIMPLES.items():
            if getattr(a, m):
                f(a.workers)


if __name__ == "__main__":
    main()
