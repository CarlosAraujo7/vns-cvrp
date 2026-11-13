#!/usr/bin/env python3
"""
run_aos_etapa20.py
==================
Estudo 20 da Direcao A (diario, Entrada 132 -- pre-registro): GENERALIDADE do indicio dos Estudos 18 e 19 (os pesos adaptativos de Ropke e Pisinger
MODULADOS pela mistura estatica ajustada, `tilt`, contra a propria mistura) em DOIS problemas que nao entraram nesses estudos: o ALNS do CVRP (o regime
dos Estudos 9 e 12; instancias X e SYN_VAL) e o ALNS do TSP (o do Estudo 13; TSPLIB e SYN_VAL_T). Regime, operadores, orcamento (1.500 n^2 avaliacoes de
distancia), temperatura, instancias e seeds de avaliacao (1000+) sao os dos Estudos 9, 12 e 13; os bracos U e `static` (a mistura estatica ajustada em
33 avaliacoes pelos Estudos 12 e 13) sao REUTILIZADOS desses estudos (a regressao exata confirma que este executor os reproduz execucao a execucao).

Bracos NOVOS (por problema):
  RP      pesos de Ropke-Pisinger com hiperparametros ajustados (rho, segmento) por busca aleatoria: 33 avaliacoes (a padrao + 32 sorteios)
  tilt    pesos de Ropke-Pisinger modulados pela mistura `static` (a dos Estudos 12/13, congelada); (rho, segmento) por busca aleatoria, 33 avaliacoes
          proprias (esforco total 33 + 33 = 66 avaliacoes)
  tilt0   o mesmo hibrido com os hiperparametros PADRAO (rho 0,1; segmento 100), sem ajuste proprio (esforco total 33)
  static65 a mistura estatica com o DOBRO do esforco de ajuste: a busca (1+4)-ES do Estudo 12/13 continuada de 8 para 16 geracoes (65 avaliacoes), mesmas
          instancias de ajuste, mesmas seeds (3000-3002) e mesma sequencia de sementes do ES (controle de esforco total; se a continuacao nao achar
          melhora, static65 = static e o braco nao e executado)
  tilt65  (EMENDA 1, Entrada 133, antes de qualquer dado de avaliacao) o hibrido com os hiperparametros PADRAO sobre a mistura static65 (esforco total 65);
          so existe se static65 mudou (senao e o tilt0)
Ajuste: SYN_TUNE (CVRP) / SYN_TUNE_T (TSP), 24 instancias, seeds 3000-3002 (72 execucoes por avaliacao); validacao (diagnostico) com seeds 3003-3005.
Modos: --regressao | --tuning {RP,tilt,static65} | --selecionar | --validar | --eval, sempre com --problema CVRP|TSP (retomaveis por chave).
Uso (WSL): python3 -u run_aos_etapa20.py --problema TSP --tuning RP --workers 11
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa18 as r18  # noqa: E402
import run_aos_etapa12 as r12  # noqa: E402
import run_aos_etapa9 as r9  # noqa: E402
import run_aos_etapa13 as r13  # noqa: E402
import run_aos_etapa4 as r4  # noqa: E402
from alns import DESTRUIR, REPARAR  # noqa: E402
from alns_loop import rodar_alns  # noqa: E402
from tsp_alns import DESTRUIR_TSP, REPARAR_TSP, MAX_ROTAS_TSP  # noqa: E402

RES = r18.RES
CAMPOS = r18.CAMPOS_18
SEEDS_TUNE = [3000, 3001, 3002]
SEEDS_VALID = [3003, 3004, 3005]
SEMENTES_AVAL = 1000
assert r12.SEEDS_TUNE12 == r13.SEEDS_TUNE_A == SEEDS_TUNE
ES_GERACOES_TOTAL = 16                         # static65: 1 + 16 x 4 = 65 avaliacoes (a busca dos Estudos 12 e 13 tinha 8 geracoes = 33)
PROB = {
    "CVRP": dict(destruir=DESTRUIR, reparar=REPARAR, kappa=r9.KAPPA9, w0=r9.W09, tf=r9.TF9, tune_conj="SYN_TUNE",
                 aval=[("X", 10), ("SYN_VAL", 5)], es_estado=RES / "etapa12_tuning_static_estado.json", es_seed=r12.ES_BASE_SEED,
                 arredonda=True),     # o braco static do Estudo 12 usou pesos arredondados a 4 casas (os do CSV de ajuste)
    "TSP": dict(destruir=DESTRUIR_TSP, reparar=REPARAR_TSP, kappa=r13.KAPPA_A, w0=r13.W0_A, tf=r13.TF_A, tune_conj="SYN_TUNE_T",
                aval=[("TSPLIB", 10), ("SYN_VAL_T", 5)], es_estado=RES / "etapa13_tuning_static_a_estado.json", es_seed=r13.ES_SEED_A,
                arredonda=False),     # o do Estudo 13 usou a precisao total
}
METODOS = ("RP", "tilt", "static65")
M18 = {"RP": "RP18", "tilt": "tilt"}           # nomes dos metodos de busca aleatoria no executor do Estudo 18
SEMENTE_RS = {("CVRP", "RP"): 20261201, ("CVRP", "tilt"): 20261202, ("TSP", "RP"): 20261211, ("TSP", "tilt"): 20261212}


# ------------------------------------------------------------------------------------------------ instancias e execucao
def nomes(prob, conj):
    return sorted(r4.nomes(conj)) if prob == "CVRP" else sorted(r13.nomes(conj))


def tamanho(prob, conj, nome):
    return r9._instancia(conj, nome)[0].n_customers if prob == "CVRP" else r13.carregar(conj, nome)[0].n_customers


def static33(prob):
    """(pesos_d, pesos_r) do braco `static` dos Estudos 12 (CVRP) e 13 (TSP), exatamente como foram avaliados."""
    if prob == "CVRP":
        e = json.loads((RES / "etapa12_tuning_escolhida.json").read_text())
    else:
        e = json.loads((RES / "etapa13_tuning_escolhidas_a.json").read_text())["static"]
    return e["pesos_d"], e["pesos_r"]


def pesos_eval(prob, theta):
    """Pesos de avaliacao de um theta, com o mesmo arredondamento que o Estudo 12 (4 casas) ou 13 (precisao total) usou para o braco static."""
    pd_, pr = r18.pesos_de_theta(theta)
    if PROB[prob]["arredonda"]:
        return [round(x, 4) for x in pd_], [round(x, 4) for x in pr]
    return pd_, pr


def spec20(prob, nome, pol, params=None, pesos_d=None, pesos_r=None, config_id="e20"):
    P = PROB[prob]
    return {"nome": nome, "pol": pol, "params": params or {}, "pesos_d": pesos_d, "pesos_r": pesos_r, "kappa": P["kappa"], "w0": P["w0"],
            "tf": P["tf"], "config_id": config_id, "regime": "a", "problema": prob}


def executar_20(sp, nivel, conj, nome_inst, seed):
    """Executor generico (CVRP e TSP). Os bracos U, RP e static passam pelos mesmos caminhos de codigo dos Estudos 9, 12 e 13 (mesmas sementes de
    politica: seed * 31 + 1 e + 2); acrescenta o uso por terco do orcamento (como o Estudo 18)."""
    prob = sp["problema"]
    P = PROB[prob]
    if prob == "CVRP":
        inst, cws = r9._instancia(conj, nome_inst)
        obj = r9._proxy(conj, nome_inst)
        x0, ref, nome = cws, inst.optimum, inst.name
        max_rotas = inst.n_vehicles_ref if inst.vehicles_exact else None
    else:
        inst, nn = r13.carregar(conj, nome_inst)
        obj, x0, ref, nome = inst, [list(nn)], inst.ref, inst.nome
        max_rotas = MAX_ROTAS_TSP
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
    best, info = rodar_alns(obj, [list(r) for r in x0], rng, orcamento, pol_d, md, pol_r, mr, lambda: obj.chamadas, w0=sp["w0"],
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
            "instancia": nome, "n_customers": inst.n_customers, "seed": seed, "orcamento": orcamento, "custo": custo, "ref": ref,
            "gap_pct": 100 * (custo - ref) / ref, "viavel": viavel, "n_rotas": len([r for r in best if r]), "calls_usadas": obj.chamadas,
            "iteracoes": info["iteracoes"], "inviaveis": info["inviaveis"], "uso_d": j(info["uso_d"]), "uso_r": j(info["uso_r"]),
            "score_d": j(info["score_d"]), "score_r": j(info["score_r"]), "calls_d": j(info["calls_d"]), "calls_r": j(info["calls_r"]),
            "tempo_s": dt, "uso_d_fase": jf(fase_d), "uso_r_fase": jf(fase_r)}


def _worker20(args):
    return executar_20(*args)


def rodar20(tarefas, out_csv, pool, cada=100):
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
        for linha in pool.imap_unordered(_worker20, pend, chunksize=1):
            w.writerow(linha)
            feitas += 1
            if feitas % cada == 0:
                f_out.flush()
                print(f"    {feitas}/{len(pend)}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return len(pend)


# ------------------------------------------------------------------------------------------------ ajuste
def tarefas_ajuste(prob, sp, seeds):
    conj = PROB[prob]["tune_conj"]
    return [(sp, 1, conj, n, s) for n in sorted(nomes(prob, conj), reverse=True) for s in seeds]


def configs_rs(prob, metodo):
    """33 configuracoes: a padrao (r00) e 32 sorteios do espaco do Estudo 18 (semente por problema e metodo)."""
    rng = random.Random(SEMENTE_RS[(prob, metodo)])
    cfgs = [("r00", dict(r18.PADRAO_RS[M18[metodo]]))]
    for i in range(1, r18.N_AVAL):
        cfgs.append((f"r{i:02d}", r18.espaco(M18[metodo], rng)))
    return cfgs


def csv_tuning(prob, metodo):
    return RES / f"etapa20_tuning_{metodo}_{prob}.csv"


def modo_static65(prob, workers):
    """Continua o (1+4)-ES do Estudo 12/13 de 8 para 16 geracoes. Antes, reavalia o theta guardado (controle de continuidade: o gap tem de
    coincidir com o do Estudo 12/13 em 1e-9)."""
    P = PROB[prob]
    out = csv_tuning(prob, "static65")
    estado_p = RES / f"etapa20_tuning_static65_{prob}_estado.json"
    if not estado_p.exists():
        shutil.copy(P["es_estado"], estado_p)
    nome = f"tune20_{prob}_static65"
    n_ref = len(nomes(prob, P["tune_conj"])) * len(SEEDS_TUNE)
    with mp.Pool(workers) as pool:
        def avaliar(theta, cid):
            pd_, pr = r18.pesos_de_theta(theta)                       # precisao total, como na busca original
            sp = spec20(prob, nome, "static", pesos_d=pd_, pesos_r=pr, config_id=cid)
            sp["params"] = {"theta": list(map(float, theta))}
            rodar20(tarefas_ajuste(prob, sp, SEEDS_TUNE), out, pool)
            m, n = r18.r16.media_gap(out, nome, cid)
            assert n == n_ref, (prob, cid, n, n_ref)
            return m

        orig = json.loads(P["es_estado"].read_text())
        m0 = avaliar(orig["theta"], "gen08best")
        assert abs(m0 - orig["melhor"]) < 1e-9, ("continuidade do ES quebrada", prob, m0, orig["melhor"])
        print(f"{prob}/static65: controle de continuidade OK (gap {m0:.6f} = {orig['melhor']:.6f} do Estudo 12/13)", flush=True)
        r18.ES_GERACOES_18 = ES_GERACOES_TOTAL
        r18.es18(estado_p, [0.0] * 7, 0.0, P["es_seed"], avaliar, f"{prob}/static65")


def modo_tuning(prob, metodo, workers):
    if metodo == "static65":
        return modo_static65(prob, workers)
    out = csv_tuning(prob, metodo)
    nome = f"tune20_{prob}_{metodo}"
    extra = {}
    if metodo == "tilt":
        extra["pesos_d"], extra["pesos_r"] = static33(prob)
    tar = []
    for cid, params in configs_rs(prob, metodo):
        sp = spec20(prob, nome, "RP" if metodo == "RP" else "tilt", params=params, config_id=cid, **extra)
        tar += tarefas_ajuste(prob, sp, SEEDS_TUNE)
    print(f"ajuste {prob}/{metodo}: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar20(tar, out, pool)
    print(f"ajuste {prob}/{metodo} concluido", flush=True)


def modo_selecionar(prob):
    P = PROB[prob]
    pd33, pr33 = static33(prob)
    esc = {"static": {"pesos_d": pd33, "pesos_r": pr33}}
    est = json.loads((RES / f"etapa20_tuning_static65_{prob}_estado.json").read_text())
    orig = json.loads(P["es_estado"].read_text())
    mudou = est["theta"] != orig["theta"]
    pd65, pr65 = pesos_eval(prob, est["theta"])
    esc["static65"] = {"mudou": bool(mudou), "theta": est["theta"], "pesos_d": pd65, "pesos_r": pr65, "gap_tune_65": est["melhor"],
                       "gap_tune_33": orig["melhor"], "geracao": est["geracao"], "n_avaliacoes": 1 + 4 * est["geracao"]}
    for m in ("RP", "tilt"):
        lst = r18.melhor_por_config(csv_tuning(prob, m), f"tune20_{prob}_{m}")
        padrao = [x for x in lst if x[0] == "r00"][0]
        t = lst[0] if lst[0][2] < padrao[2] - 1e-12 else padrao                  # empate ou pior que o padrao -> padrao
        esc[m] = {"params": {k: v for k, v in json.loads(t[1]).items() if k not in ("pesos_d", "pesos_r")}, "gap_tune": t[2],
                  "gap_padrao": padrao[2], "config_id": t[0], "n_configs": len(lst)}
    (RES / f"etapa20_tuning_escolhidas_{prob}.json").write_text(json.dumps(esc, indent=1))
    print(f"{prob}/static65: {'MUDOU' if mudou else 'sem melhora (= static)'}  gap_tune {orig['melhor']:.4f} -> {est['melhor']:.4f}  "
          f"({1 + 4 * est['geracao']} avaliacoes)")
    for m in ("RP", "tilt"):
        v = esc[m]
        print(f"{prob}/{m}: config {v['config_id']}  gap_tune {v['gap_tune']:.4f}  (padrao {v['gap_padrao']:.4f})  n_configs {v['n_configs']}  "
              f"params {v['params']}")


def specs_eval(prob, com_static=False):
    esc = json.loads((RES / f"etapa20_tuning_escolhidas_{prob}.json").read_text())
    p = lambda a: f"s20_{prob}_{a}"  # noqa: E731
    pd33, pr33 = esc["static"]["pesos_d"], esc["static"]["pesos_r"]
    sp = {p("RP"): spec20(prob, p("RP"), "RP", params=esc["RP"]["params"]),
          p("tilt"): spec20(prob, p("tilt"), "tilt", params=esc["tilt"]["params"], pesos_d=pd33, pesos_r=pr33),
          p("tilt0"): spec20(prob, p("tilt0"), "tilt", params=dict(r18.PADRAO_RS["tilt"]), pesos_d=pd33, pesos_r=pr33)}
    if esc["static65"]["mudou"]:
        sp[p("static65")] = spec20(prob, p("static65"), "static", pesos_d=esc["static65"]["pesos_d"], pesos_r=esc["static65"]["pesos_r"])
        # EMENDA 1 (Entrada 133, antes de qualquer dado de avaliacao): o hibrido sobre a mistura de 65 avaliacoes, hiperparametros padrao (esforco 65)
        sp[p("tilt65")] = spec20(prob, p("tilt65"), "tilt", params=dict(r18.PADRAO_RS["tilt"]), pesos_d=esc["static65"]["pesos_d"],
                                 pesos_r=esc["static65"]["pesos_r"])
    if com_static:                                                            # so na validacao (a avaliacao reutiliza o dos Estudos 12/13)
        sp[p("static")] = spec20(prob, p("static"), "static", pesos_d=pd33, pesos_r=pr33)
    return sp


def modo_validar(prob, workers):
    sp = specs_eval(prob, com_static=True)
    tar = [t for s in sp.values() for t in tarefas_ajuste(prob, s, SEEDS_VALID)]
    print(f"validacao {prob} (instancias de ajuste, seeds novas): {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar20(tar, RES / f"etapa20_validacao_{prob}.csv", pool)
    import pandas as pd
    d = pd.read_csv(RES / f"etapa20_validacao_{prob}.csv")
    print(d[d.viavel].groupby("braco").gap_pct.mean().sort_values().round(4).to_string())


def instancias_aval(prob):
    out = [(c, n, ns, tamanho(prob, c, n)) for c, ns in PROB[prob]["aval"] for n in nomes(prob, c)]
    return sorted(out, key=lambda t: -t[3])                                  # as maiores primeiro (balanceia a carga)


def modo_eval(prob, workers):
    sp = specs_eval(prob)
    tar = [(s, 1, c, n, SEMENTES_AVAL + sd) for c, n, ns, _nc in instancias_aval(prob) for s in sp.values() for sd in range(ns)]
    print(f"eval Estudo 20 {prob}: {len(tar)} execucoes ({', '.join(sp)})", flush=True)
    with mp.Pool(workers) as pool:
        rodar20(tar, RES / f"etapa20_eval_{prob}.csv", pool)
    print(f"eval Estudo 20 {prob} concluido", flush=True)


# ------------------------------------------------------------------------------------------------ regressao exata e identidades
CAMPOS_REG = ("iteracoes", "calls_usadas", "inviaveis", "n_rotas", "uso_d", "uso_r", "score_d", "score_r", "calls_d", "calls_r")


def modo_regressao(workers):
    """(1) O executor generico reproduz EXATAMENTE as execucoes dos Estudos 9 e 12 (CVRP: U, RP, static) e 13 (TSP: U, RP, static): custo,
    iteracoes, avaliacoes, usos e escores. (2) Os conjuntos de instancias do executor coincidem com os dos CSVs. (3) Identidades de ponta a ponta
    em orcamento reduzido: tilt com rho = 0 == static; tilt com p0 uniforme == RP."""
    import pandas as pd
    rs = random.Random(11)
    esc9 = json.loads((RES / "etapa9_tuning_escolhidas.json").read_text())
    esc13 = json.loads((RES / "etapa13_tuning_escolhidas_a.json").read_text())
    fontes = {"CVRP": (pd.read_csv(RES / "etapa9_eval.csv"), pd.read_csv(RES / "etapa12_eval.csv"), "s9_", "s12_", esc9["RP"]["params"]),
              "TSP": (pd.read_csv(RES / "etapa13_eval_a.csv"), None, "s13a_", "s13a_", esc13["RP"]["params"])}
    casos = []                                    # (prob, braco, spec, linha de referencia)
    for prob, (e, e12, pref, pref_st, p_rp) in fontes.items():
        for c, ns in PROB[prob]["aval"]:                                          # (2) mesmos conjuntos de instancias
            assert sorted(e[e.conjunto == c].instancia.unique()) == nomes(prob, c), (prob, c)
            assert int(e[(e.conjunto == c) & (e.braco == pref + "U")].groupby("instancia").seed.nunique().min()) == ns, (prob, c, ns)
        pd33, pr33 = static33(prob)
        sp = {"U": (spec20(prob, "reg_U", "U"), e, pref + "U"), "RP": (spec20(prob, "reg_RP", "RP", params=p_rp), e, pref + "RP"),
              "static": (spec20(prob, "reg_static", "static", pesos_d=pd33, pesos_r=pr33), e12 if e12 is not None else e, pref_st + "static")}
        todas = sorted(e.instancia.unique())
        for arm, (s, fonte, nome_braco) in sp.items():
            for i in rs.sample(todas, 3):
                ref = fonte[(fonte.braco == nome_braco) & (fonte.instancia == i)]
                casos.append((prob, arm, s, ref.iloc[rs.randrange(len(ref))]))
    tar = [(s, 1, lin.conjunto, lin.instancia, int(lin.seed)) for (_p, _a, s, lin) in casos]
    with mp.Pool(min(workers, len(tar))) as pool:
        novas = pool.map(_worker20, tar)
    for (prob, arm, s, lin), nv in zip(casos, novas):
        assert nv["instancia"] == lin.instancia, (prob, arm, nv["instancia"], lin.instancia)         # chave de retomada = nome da instancia
        for campo in CAMPOS_REG:
            assert str(nv[campo]) == str(lin[campo]), (prob, arm, lin.instancia, lin.seed, campo, nv[campo], lin[campo])
        assert abs(nv["custo"] - lin["custo"]) < 1e-9 and bool(nv["viavel"]) == bool(lin["viavel"]), (prob, arm, lin.instancia, lin.seed)
    print(f"REGRESSAO EXATA: OK ({len(casos)} execucoes dos Estudos 9, 12 e 13 reproduzidas: custo, iteracoes, avaliacoes, usos e escores)")
    for prob in PROB:                                                              # (3) identidades de ponta a ponta
        P = PROB[prob]
        conj = P["tune_conj"]
        n0 = sorted(nomes(prob, conj), key=lambda n: tamanho(prob, conj, n))[0]
        pd33, pr33 = static33(prob)
        uni_d, uni_r = [0.25] * len(P["destruir"]), [1 / 3] * len(P["reparar"])
        pares = [("tilt(rho=0) == static", spec20(prob, "i_st", "static", pesos_d=pd33, pesos_r=pr33),
                  spec20(prob, "i_t0", "tilt", params={"rho": 0.0, "segmento": 100}, pesos_d=pd33, pesos_r=pr33)),
                 ("tilt(p0 uniforme) == RP", spec20(prob, "i_rp", "RP", params={"rho": 0.1, "segmento": 100}),
                  spec20(prob, "i_tu", "tilt", params={"rho": 0.1, "segmento": 100}, pesos_d=uni_d, pesos_r=uni_r))]
        for rotulo, a, b in pares:
            ra, rb = executar_20(a, 0.1, conj, n0, 1), executar_20(b, 0.1, conj, n0, 1)
            for campo in CAMPOS_REG + ("custo",):
                assert str(ra[campo]) == str(rb[campo]), (prob, rotulo, campo, ra[campo], rb[campo])
            print(f"IDENTIDADE {prob}: {rotulo} OK em {n0} (custo {ra['custo']:.4f}, {ra['iteracoes']} iteracoes)")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("regressao", "selecionar", "validar", "eval"):
        g.add_argument(f"--{m}", action="store_true")
    g.add_argument("--tuning", choices=METODOS)
    ap.add_argument("--problema", choices=list(PROB))
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.regressao:
        return modo_regressao(a.workers)
    assert a.problema, "--problema CVRP|TSP e obrigatorio neste modo"
    if a.selecionar:
        return modo_selecionar(a.problema)
    if a.tuning:
        return modo_tuning(a.problema, a.tuning, a.workers)
    if a.validar:
        return modo_validar(a.problema, a.workers)
    return modo_eval(a.problema, a.workers)


if __name__ == "__main__":
    main()
