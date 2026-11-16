#!/usr/bin/env python3
"""
run_aos_etapa28.py
==================
Estudo 28 da Direcao A (diario, Entrada 148 -- pre-registro): O CONJUNTO GRANDE DE OPERADORES NO CVRPTW. No ALNS do CVRP com 21 operadores (Estudo 21) os pesos adaptativos de Ropke e Pisinger (RP, sem ajuste)
superam a seleção uniforme e as misturas estaticas ajustadas; no ALNS do CVRPTW com 7 operadores (Estudo 16) a mistura estatica ajustada e MELHOR que o RP (-0,52 pp contra -0,17 pp da uniforme) e nenhuma politica a
supera. Aqui o ALNS do CVRPTW (`src/tw_alns.py`) recebe o conjunto de 21 operadores (`src/tw_alns_ext.py`: 12 destruicoes x 9 reparos com insercoes viaveis em janelas de tempo) e se repete o desenho do Estudo 21:
mistura estatica (1+4)-ES de theta em R^21 em SYN_TUNE_W (seeds 3000-3002, 72 execucoes por avaliacao; S33 = 33 avaliacoes, S65 = 65), RP e HG-UCB congelados (os do Estudo 16), uniforme; avaliacao nas 56
instancias de Solomon (seeds 1000-1004). Referencia do conjunto de 7 operadores: os bracos do Estudo 16 nas mesmas instancias e seeds. Recozimento (w0 0,2, tf 1e-3), 1.500 n^2 avaliacoes, como no Estudo 16.
Modos: --regressao | --instancias | --ajuste | --selecionar | --validar | --eval (retomaveis por chave). Uso (WSL): python3 -u run_aos_etapa28.py --ajuste --workers 11
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
import run_aos_etapa14 as r14  # noqa: E402
import run_aos_etapa16 as r16  # noqa: E402
import run_aos_etapa18 as r18  # noqa: E402
import tw_alns_ext as tx  # noqa: E402
from alns_loop import rodar_alns  # noqa: E402
from politicas_contexto import softmax_np  # noqa: E402
from tw_alns import EstadoTW  # noqa: E402

RES = r16.RES
CAMPOS = r16.CAMPOS_A
SEEDS_TUNE = r16.SEEDS_TUNE_A                      # 3000-3002
SEEDS_VALID = [3003, 3004, 3005]
SEMENTES_AVAL = 1000
SEEDS_AVAL = 5
GER_S33, GER_S65 = 8, 16
SIGMA0 = r18.ES_SIGMA0_STATIC
SEMENTE_ES = 20261802
KD, KR = len(tx.DESTRUIR_P21), len(tx.REPARAR_P21)
TUNE_CONJ = "SYN_TUNE_W"
AVAL_CONJ = "SOLOMON"
PADRAO_RP = {"rho": 0.1, "segmento": 100}


def esc16():
    return json.loads((RES / "etapa16_tuning_escolhidas_a.json").read_text())


def spec28(pool, nome, pol, params=None, pesos_d=None, pesos_r=None, config_id="e28"):
    sp = r16.spec_a(nome, pol, params=params, pesos_d=pesos_d, pesos_r=pesos_r, config_id=config_id)
    sp["pool"] = pool
    return sp


def executar_28(sp, nivel, conj, nome_inst, seed):
    """Como `run_aos_etapa16.executar_a` (sementes de politica seed * 31 + 1 e + 2), com as tabelas de operadores do conjunto `pool`; com pool = P7 reproduz exatamente as execucoes do Estudo 16 (`--regressao`)."""
    D, R = tx.POOLS_TW[sp["pool"]]
    inst, x0 = r16.carregar(conj, nome_inst)
    inst.chamadas = 0
    rng = random.Random(seed)
    pol_d, md = r18.fabricar(sp, "d", len(D), random.Random(seed * 31 + 1))
    pol_r, mr = r18.fabricar(sp, "r", len(R), random.Random(seed * 31 + 2))
    orcamento = int(nivel * sp["kappa"] * inst.n_customers ** 2)
    t0 = time.perf_counter()
    best, info = rodar_alns(inst, [list(r) for r in x0], rng, orcamento, pol_d, md, pol_r, mr, lambda: inst.chamadas, w0=sp["w0"], tf_fator=sp["tf"], max_rotas=inst.nveic, destruir=D, reparar=R,
                            estado_cls=EstadoTW)
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(best)
    viavel, _ = inst.is_feasible(best)
    j = lambda v: "|".join(f"{x:g}" for x in v)  # noqa: E731
    params = dict(sp["params"])
    if sp["pol"] == "static":
        params = {"pesos_d": [round(x, 4) for x in sp["pesos_d"]], "pesos_r": [round(x, 4) for x in sp["pesos_r"]]}
        if "theta" in sp["params"]:
            params["theta"] = sp["params"]["theta"]
    params["pool"] = sp["pool"]
    return {"braco": sp["nome"], "config_id": sp["config_id"], "params": json.dumps(params, sort_keys=True), "nivel": nivel, "conjunto": conj, "instancia": inst.nome, "n_customers": inst.n_customers,
            "seed": seed, "orcamento": orcamento, "custo": custo, "ref": inst.ref, "gap_pct": 100 * (custo - inst.ref) / inst.ref, "viavel": viavel, "n_rotas": len([r for r in best if r]),
            "calls_usadas": inst.chamadas, "iteracoes": info["iteracoes"], "inviaveis": info["inviaveis"], "uso_d": j(info["uso_d"]), "uso_r": j(info["uso_r"]), "score_d": j(info["score_d"]),
            "score_r": j(info["score_r"]), "calls_d": j(info["calls_d"]), "calls_r": j(info["calls_r"]), "tempo_s": dt}


def _worker28(args):
    return executar_28(*args)


def rodar28(tarefas, out_csv, pool, cada=100):
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
        for linha in pool.imap_unordered(_worker28, pend, chunksize=1):
            w.writerow(linha)
            feitas += 1
            if feitas % cada == 0:
                f_out.flush()
                print(f"    {feitas}/{len(pend)}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return len(pend)


def nomes(conj):
    return sorted(r14.nomes(conj))


def tarefas_ajuste(sp, seeds):
    return [(sp, 1, TUNE_CONJ, n, s) for n in sorted(nomes(TUNE_CONJ), reverse=True) for s in seeds]


def pesos_de_theta(theta):
    th = np.asarray(theta, float)
    return list(map(float, softmax_np(th[:KD]))), list(map(float, softmax_np(th[KD:])))


def pesos_eval(theta):
    pd_, pr = pesos_de_theta(theta)
    return [round(x, 4) for x in pd_], [round(x, 4) for x in pr]


def csv_ajuste():
    return RES / "etapa28_ajuste_static.csv"


def estado_es(sufixo=""):
    return RES / f"etapa28_ajuste_static_estado{sufixo}.json"


# ------------------------------------------------------------------------------------------------ ajuste da mistura (S33, S65)
def modo_ajuste(workers):
    """(1+4)-ES de theta em R^21 a partir de theta = 0 (mistura uniforme), 16 geracoes; o estado da geracao 8 (33 avaliacoes) e guardado a parte (S33); o final (65 avaliacoes) e S65."""
    out = csv_ajuste()
    nome = "tune28_static"
    n_ref = len(nomes(TUNE_CONJ)) * len(SEEDS_TUNE)
    with mp.Pool(workers) as mpool:
        def avaliar(theta, cid):
            pd_, pr = pesos_de_theta(theta)
            sp = spec28("P21", nome, "static", pesos_d=pd_, pesos_r=pr, config_id=cid)
            sp["params"] = {"theta": list(map(float, theta))}
            rodar28(tarefas_ajuste(sp, SEEDS_TUNE), out, mpool)
            m, n = r16.media_gap(out, nome, cid)
            assert n == n_ref, (cid, n, n_ref)
            return m

        g8 = estado_es("_gen08")
        if not g8.exists():
            r18.ES_GERACOES_18 = GER_S33
            r18.es18(estado_es(), [0.0] * (KD + KR), SIGMA0, SEMENTE_ES, avaliar, "P21-TW/static")
            assert json.loads(estado_es().read_text())["geracao"] == GER_S33
            shutil.copy(estado_es(), g8)
        r18.ES_GERACOES_18 = GER_S65
        r18.es18(estado_es(), [0.0] * (KD + KR), SIGMA0, SEMENTE_ES, avaliar, "P21-TW/static")


def trajetoria_es():
    """Incumbentes do ES por geracao, reconstruidos do CSV de ajuste: [(geracao, theta, gap de ajuste, n avaliacoes)]."""
    import pandas as pd
    d = pd.read_csv(csv_ajuste())
    d = d[(d.braco == "tune28_static") & d.viavel]
    g = d.groupby("config_id").agg(gap=("gap_pct", "mean"), params=("params", "first"), n=("gap_pct", "size"))
    assert (g.n == len(nomes(TUNE_CONJ)) * len(SEEDS_TUNE)).all()
    traj = []
    atual_theta, atual_gap = json.loads(g.loc["gen00", "params"])["theta"], float(g.loc["gen00", "gap"])
    traj.append((0, atual_theta, atual_gap, 1))
    for gen in sorted({int(c[3:5]) for c in g.index if c.startswith("gen") and "c" in c[3:]}):
        filhos = [(float(g.loc[c, "gap"]), c) for c in g.index if c.startswith(f"gen{gen:02d}c")]
        if len(filhos) < 4:
            break
        gap_f, cid = min(filhos)
        if gap_f < atual_gap:
            atual_gap, atual_theta = gap_f, json.loads(g.loc[cid, "params"])["theta"]
        traj.append((gen, atual_theta, atual_gap, 1 + 4 * gen))
    assert all(len(t[1]) == KD + KR for t in traj)
    return traj


def modo_selecionar():
    traj = trajetoria_es()
    por_ger = {t[0]: t for t in traj}
    esc = {"trajetoria": [{"geracao": g, "n_avaliacoes": n, "gap_ajuste": gp, "theta": th} for g, th, gp, n in traj]}
    t33, t65 = por_ger[GER_S33], por_ger[GER_S65]
    g8, g16 = json.loads(estado_es("_gen08").read_text()), json.loads(estado_es().read_text())
    assert g8["theta"] == t33[1] and abs(g8["melhor"] - t33[2]) < 1e-9 and g16["theta"] == t65[1] and abs(g16["melhor"] - t65[2]) < 1e-9, "a trajetoria reconstruida do CSV nao coincide com os estados do ES"
    pd33, pr33 = pesos_eval(t33[1])
    pd65, pr65 = pesos_eval(t65[1])
    esc["S33"] = {"theta": t33[1], "pesos_d": pd33, "pesos_r": pr33, "gap_ajuste": t33[2], "n_avaliacoes": t33[3]}
    esc["S65"] = {"mudou": bool(t65[1] != t33[1]), "theta": t65[1], "pesos_d": pd65, "pesos_r": pr65, "gap_ajuste": t65[2], "n_avaliacoes": t65[3]}
    (RES / "etapa28_ajuste_escolhidas.json").write_text(json.dumps(esc, indent=1))
    print(f"S33 gap de ajuste {t33[2]:.4f} ({t33[3]} aval.)  S65 {t65[2]:.4f} ({t65[3]} aval.; {'MUDOU' if esc['S65']['mudou'] else 'sem melhora'})")
    print("curva do ES (gap de ajuste do incumbente): " + ", ".join(f"g{g}:{gp:.3f}" for g, _th, gp, _n in traj))


# ------------------------------------------------------------------------------------------------ validacao e avaliacao
def specs_eval():
    esc = json.loads((RES / "etapa28_ajuste_escolhidas.json").read_text())
    p = lambda a: f"s28_{a}"  # noqa: E731
    sp = {p("U"): spec28("P21", p("U"), "U"), p("RP"): spec28("P21", p("RP"), "RP", params=dict(PADRAO_RP)),
          p("S33"): spec28("P21", p("S33"), "static", pesos_d=esc["S33"]["pesos_d"], pesos_r=esc["S33"]["pesos_r"]),
          p("HG"): spec28("P21", p("HG"), "hgucb", params=dict(esc16()["hgucb"]["params"]))}
    if esc["S65"]["mudou"]:
        sp[p("S65")] = spec28("P21", p("S65"), "static", pesos_d=esc["S65"]["pesos_d"], pesos_r=esc["S65"]["pesos_r"])
    return sp


def modo_validar(workers):
    """Diagnostico (seeds 3003-3005, instancias de ajuste): os bracos da avaliacao, incluindo S65 mesmo que igual a S33 (a curva de aprendizado do ajuste sem o vies de selecao)."""
    esc = json.loads((RES / "etapa28_ajuste_escolhidas.json").read_text())
    sp = dict(specs_eval())
    if "s28_S65" not in sp:
        sp["s28_S65"] = spec28("P21", "s28_S65", "static", pesos_d=esc["S65"]["pesos_d"], pesos_r=esc["S65"]["pesos_r"])
    tar = [t for s in sp.values() for t in tarefas_ajuste(s, SEEDS_VALID)]
    print(f"validacao (instancias de ajuste, seeds novas): {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as mpool:
        rodar28(tar, RES / "etapa28_validacao.csv", mpool)
    import pandas as pd
    d = pd.read_csv(RES / "etapa28_validacao.csv")
    print(d[d.viavel].groupby("braco").gap_pct.mean().sort_values().round(4).to_string())


def instancias_aval():
    out = [(AVAL_CONJ, n, r16.carregar(AVAL_CONJ, n)[0].n_customers) for n in nomes(AVAL_CONJ)]
    return sorted(out, key=lambda t: t[1])


def modo_eval(workers):
    sp = specs_eval()
    tar = [(s, 1, c, n, SEMENTES_AVAL + sd) for c, n, _nc in instancias_aval() for s in sp.values() for sd in range(SEEDS_AVAL)]
    print(f"eval Estudo 28 (P21 do CVRPTW): {len(tar)} execucoes ({', '.join(sp)})", flush=True)
    with mp.Pool(workers) as mpool:
        rodar28(tar, RES / "etapa28_eval.csv", mpool)
    print("eval Estudo 28 concluido", flush=True)


def modo_instancias():
    for conj, n_esp in ((TUNE_CONJ, 24), (AVAL_CONJ, 56)):
        nm = nomes(conj)
        assert len(nm) == n_esp, (conj, len(nm))
        for n in nm:
            inst, x0 = r16.carregar(conj, n)
            assert inst.n_customers == 100 and sorted(c for r in x0 for c in r) == list(range(1, 101)), (conj, n)
            ok, motivo = inst.is_feasible(x0)
            assert ok, (conj, n, motivo)
        print(f"{conj}: {len(nm)} instancias de 100 clientes com solucao inicial viavel", flush=True)
    for n in nomes(AVAL_CONJ):
        assert (ROOT / "instances" / AVAL_CONJ / "bks" / f"{n}.sol").exists(), n
    print("INSTANCIAS: OK", flush=True)


# ------------------------------------------------------------------------------------------------ regressao exata
CAMPOS_REG = ("iteracoes", "calls_usadas", "inviaveis", "n_rotas", "uso_d", "uso_r", "score_d", "score_r", "calls_d", "calls_r")


def modo_regressao(workers):
    """Com pool = P7 e as configuracoes do Estudo 16 (U, RP padrao, HG-UCB ajustado, mistura estatica ajustada) as execucoes reproduzem EXATAMENTE (custo, iteracoes, avaliacoes, usos e escores) 12 execucoes de
    etapa16_eval_a.csv (3 por braco)."""
    import pandas as pd
    rs = random.Random(28)
    e16 = pd.read_csv(RES / "etapa16_eval_a.csv")
    esc = esc16()
    sp7 = {"s16a_U": spec28("P7", "reg_U", "U"), "s16a_RP": spec28("P7", "reg_RP", "RP", params=esc["RP"]["params"]), "s16a_hgucb": spec28("P7", "reg_HG", "hgucb", params=esc["hgucb"]["params"]),
           "s16a_static": spec28("P7", "reg_static", "static", pesos_d=esc["static"]["pesos_d"], pesos_r=esc["static"]["pesos_r"])}
    casos = []
    for b, sp in sp7.items():
        d = e16[e16.braco == b]
        for i in rs.sample(sorted(d.instancia.unique()), 3):
            ref = d[d.instancia == i]
            casos.append((b, sp, ref.iloc[rs.randrange(len(ref))]))
    tar = [(s, 1, lin.conjunto, lin.instancia, int(lin.seed)) for (_b, s, lin) in casos]
    with mp.Pool(min(workers, len(tar))) as mpool:
        novas = mpool.map(_worker28, tar)
    for (b, s, lin), nv in zip(casos, novas):
        assert nv["instancia"] == lin.instancia, (b, nv["instancia"], lin.instancia)
        for campo in CAMPOS_REG:
            assert str(nv[campo]) == str(lin[campo]), (b, lin.instancia, lin.seed, campo, nv[campo], lin[campo])
        assert abs(nv["custo"] - lin["custo"]) < 1e-9 and bool(nv["viavel"]) == bool(lin["viavel"]), (b, lin.instancia, lin.seed)
    print(f"REGRESSAO EXATA: OK ({len(casos)} execucoes do Estudo 16 reproduzidas com o executor do Estudo 28 em P7)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("regressao", "instancias", "ajuste", "selecionar", "validar", "eval"):
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.regressao:
        return modo_regressao(a.workers)
    if a.instancias:
        return modo_instancias()
    if a.ajuste:
        return modo_ajuste(a.workers)
    if a.selecionar:
        return modo_selecionar()
    if a.validar:
        return modo_validar(a.workers)
    return modo_eval(a.workers)


if __name__ == "__main__":
    main()
