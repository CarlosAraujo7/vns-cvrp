#!/usr/bin/env python3
"""
run_aos_etapa22.py
==================
Estudo 22 da Direcao A (diario, Entradas 138 e 140 -- pre-registro e emenda 1): DESLOCAMENTOS DE IMPLANTACAO do conjunto de 21 operadores (P21). A Entrada 138 pre-registrou o estudo
para o conjunto de 7 operadores (P7); a EMENDA 1 (Entrada 140, antes de qualquer dado do Estudo 22) o troca por P21, onde o Estudo 21 achou que os pesos adaptativos de Ropke e Pisinger
(RP, sem ajuste) superam as misturas estaticas ajustadas em 33 e 65 avaliacoes. Aqui as configuracoes CONGELADAS do Estudo 21 (e da emenda do Estudo 21) sao implantadas em situacoes
que o ajuste nao viu:
  cls      outra CLASSE de instancias: Augerat A e B e Christofides-Mingozzi-Toth (53 instancias, 30 a 120 clientes; as do estudo original, nunca usadas nos ALNS), mesmo orcamento
  short    orcamento 4 vezes MENOR (375 n^2) nas mesmas 53 instancias
  longabc  orcamento 20 vezes MAIOR (30.000 n^2) nas mesmas 53 instancias
  longx    orcamento 4 vezes maior (6.000 n^2) nas 21 instancias X (com a referencia de 1.500 n^2 do Estudo 21)
  acc      outra REGRA DE ACEITACAO: record-to-record travel (desvio inicial delta0 escolhido em SYN_TUNE com a selecao uniforme de P21)
Bracos (congelados do Estudo 21): U, RP (padrao), RPt (RP ajustado, 33 avaliacoes), S33, S65 (misturas de 33 e 65 avaliacoes), HG (HG-UCB do Estudo 9) e S129 (a mistura de 129 avaliacoes da
emenda do Estudo 21; so se a continuacao do ES melhorou S65). S* = S129 se existe, senao S65 (o comparador estatico de maior esforco).
Modos: --regressao | --calibrar | --eval {cls,short,longabc,longx,acc}   (retomaveis por chave)
Uso (WSL): python3 -u run_aos_etapa22.py --eval cls --workers 11
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
import run_aos_etapa18 as r18  # noqa: E402
import run_aos_etapa9 as r9  # noqa: E402
import run_aos_etapa4 as r4  # noqa: E402
import run_aos_etapa20 as r20  # noqa: E402
import run_aos_etapa21 as r21  # noqa: E402
import alns_ext as ax  # noqa: E402
from alns_loop_rrt import rodar_alns_aceitacao  # noqa: E402

RES = r18.RES
CAMPOS = r18.CAMPOS_18
SEEDS_TUNE, SEEDS_VALID, SEMENTES_AVAL = r20.SEEDS_TUNE, r20.SEEDS_VALID, 1000
POOL = "P21"
DESTRUIR_P, REPARAR_P = ax.POOLS[POOL]
DELTAS = (0.0025, 0.005, 0.01, 0.02, 0.04)           # desvios iniciais candidatos do record-to-record travel
REG = {
    "cls": dict(conj=("A", "B", "CMT"), nivel=1.0, seeds=10, aceit="sa", bracos=("U", "RP", "RPt", "S33", "S65", "S129", "HG")),
    "short": dict(conj=("A", "B", "CMT"), nivel=0.25, seeds=10, aceit="sa", bracos=("U", "RP", "S65", "S129")),
    "longabc": dict(conj=("A", "B", "CMT"), nivel=20.0, seeds=5, aceit="sa", bracos=("U", "RP", "S65", "S129")),
    "longx": dict(conj=("X",), nivel=4.0, seeds=3, aceit="sa", bracos=("U", "RP", "S65", "S129")),
    "acc": dict(conj=("A", "B", "CMT"), nivel=1.0, seeds=10, aceit="rrt", bracos=("U", "RP", "S65", "S129")),
}


# ------------------------------------------------------------------------------------------------ especificacoes e execucao
def spec22(nome, pol, params=None, pesos_d=None, pesos_r=None, aceit="sa", delta0=0.0, config_id="e22"):
    return {"nome": nome, "pol": pol, "params": params or {}, "pesos_d": pesos_d, "pesos_r": pesos_r, "kappa": r9.KAPPA9, "w0": r9.W09, "tf": r9.TF9,
            "config_id": config_id, "regime": "a", "problema": "CVRP", "pool": POOL, "aceit": aceit, "delta0": delta0}


def congeladas():
    """(pol, params, pesos_d, pesos_r) dos bracos de P21, exatamente como o Estudo 21 os avaliou; S129 so existe se a continuacao do ES melhorou S65."""
    esc = json.loads((RES / f"etapa21_ajuste_escolhidas_{POOL}.json").read_text())
    out = {"U": ("U", {}, None, None), "RP": ("RP", dict(r21.PADRAO_RP), None, None),
           "S33": ("static", {}, esc["S33"]["pesos_d"], esc["S33"]["pesos_r"]), "HG": ("hgucb", dict(r21.PARAMS_HG), None, None)}
    if esc["RPt"]["mudou"]:
        out["RPt"] = ("RP", esc["RPt"]["params"], None, None)
    if esc["S65"]["mudou"]:
        out["S65"] = ("static", {}, esc["S65"]["pesos_d"], esc["S65"]["pesos_r"])
    f = RES / f"etapa21b_ajuste_escolhidas_{POOL}.json"
    if f.exists():
        e129 = json.loads(f.read_text())["S129"]
        if e129["mudou"]:
            out["S129"] = ("static", {}, e129["pesos_d"], e129["pesos_r"])
    return out


def delta0_escolhido():
    return json.loads((RES / "etapa22_rrt_delta0.json").read_text())["delta0"]


def specs_regime(reg):
    cfg = REG[reg]
    fr = congeladas()
    delta0 = delta0_escolhido() if cfg["aceit"] == "rrt" else 0.0
    out = {}
    for a in cfg["bracos"]:
        if a not in fr:                                   # RPt = RP, S65 = S33 ou S129 = S65: o braco nao existe (a analise copia a coluna)
            continue
        pol, params, pd_, pr = fr[a]
        out[a] = spec22(f"s22_{reg}_{a}", pol, params=params, pesos_d=pd_, pesos_r=pr, aceit=cfg["aceit"], delta0=delta0)
    return out


def executar_22(sp, nivel, conj, nome_inst, seed):
    """Como executar_21 (mesmas sementes de politica: seed * 31 + 1 e + 2), com as tabelas de P21 e a regra de aceitacao selecionavel (`sa` reproduz o laco dos Estudos 9 a 21)."""
    inst, cws = r9._instancia(conj, nome_inst)
    obj = r9._proxy(conj, nome_inst)
    max_rotas = inst.n_vehicles_ref if inst.vehicles_exact else None
    obj.chamadas = 0
    rng = random.Random(seed)
    kd, kr = len(DESTRUIR_P), len(REPARAR_P)
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
    best, info = rodar_alns_aceitacao(obj, [list(r) for r in cws], rng, orcamento, pol_d, md, pol_r, mr, lambda: obj.chamadas, w0=sp["w0"], tf_fator=sp["tf"],
                                      max_rotas=max_rotas, on_iteracao=on_it, destruir=DESTRUIR_P, reparar=REPARAR_P, aceitacao=sp["aceit"], rrt_delta0=sp["delta0"])
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(best)
    viavel, _ = inst.is_feasible(best)
    j = lambda v: "|".join(f"{x:g}" for x in v)  # noqa: E731
    jf = lambda m: ";".join(j(v) for v in m)      # noqa: E731
    params = dict(sp["params"])
    if sp["pol"] == "static":
        params["pesos_d"] = [round(x, 4) for x in sp["pesos_d"]]
        params["pesos_r"] = [round(x, 4) for x in sp["pesos_r"]]
    if sp["aceit"] == "rrt":
        params["rrt_delta0"] = sp["delta0"]
    return {"braco": sp["nome"], "config_id": sp["config_id"], "params": json.dumps(params, sort_keys=True), "nivel": nivel, "conjunto": conj,
            "instancia": inst.name, "n_customers": inst.n_customers, "seed": seed, "orcamento": orcamento, "custo": custo, "ref": inst.optimum,
            "gap_pct": 100 * (custo - inst.optimum) / inst.optimum, "viavel": viavel, "n_rotas": len([r for r in best if r]),
            "calls_usadas": obj.chamadas, "iteracoes": info["iteracoes"], "inviaveis": info["inviaveis"], "uso_d": j(info["uso_d"]),
            "uso_r": j(info["uso_r"]), "score_d": j(info["score_d"]), "score_r": j(info["score_r"]), "calls_d": j(info["calls_d"]),
            "calls_r": j(info["calls_r"]), "tempo_s": dt, "uso_d_fase": jf(fase_d), "uso_r_fase": jf(fase_r)}


def _worker22(args):
    return executar_22(*args)


def rodar22(tarefas, out_csv, pool, cada=100):
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
        for linha in pool.imap_unordered(_worker22, pend, chunksize=1):
            w.writerow(linha)
            feitas += 1
            if feitas % cada == 0:
                f_out.flush()
                print(f"    {feitas}/{len(pend)}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return len(pend)


# ------------------------------------------------------------------------------------------------ calibracao do desvio do RRT
def tarefas_ajuste(sp, seeds):
    return [(sp, 1, "SYN_TUNE", n, s) for n in sorted(r4.nomes("SYN_TUNE"), reverse=True) for s in seeds]


def modo_calibrar(workers):
    """Escolhe delta0 do record-to-record travel nas instancias de AJUSTE (selecao uniforme de P21, seeds 3000-3002): o de menor gap medio (empate: o menor delta0)."""
    out = RES / "etapa22_calibracao.csv"
    nome = "cal22_rrt_U"
    tar = []
    for d0 in DELTAS:
        tar += tarefas_ajuste(spec22(nome, "U", aceit="rrt", delta0=d0, config_id=f"d{d0}"), SEEDS_TUNE)
    print(f"calibracao do desvio do RRT (P21, selecao uniforme): {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar22(tar, out, pool)
    import pandas as pd
    d = pd.read_csv(out)
    g = d[d.viavel].groupby("config_id").gap_pct.mean()
    print(g.round(4).to_string())
    melhor = min(DELTAS, key=lambda x: (round(float(g[f"d{x}"]), 12), x))
    (RES / "etapa22_rrt_delta0.json").write_text(json.dumps({"delta0": melhor, "gap_ajuste_U": {str(x): float(g[f"d{x}"]) for x in DELTAS}}, indent=1))
    print(f"delta0 escolhido: {melhor}", flush=True)


# ------------------------------------------------------------------------------------------------ avaliacao
def instancias(conjs):
    out = [(c, n, r9._instancia(c, n)[0].n_customers) for c in conjs for n in r4.nomes(c)]
    return sorted(out, key=lambda t: -t[2])                                  # as maiores primeiro (balanceia a carga)


def modo_eval(reg, workers):
    cfg = REG[reg]
    sp = specs_regime(reg)
    tar = [(s, cfg["nivel"], c, n, SEMENTES_AVAL + sd) for c, n, _nc in instancias(cfg["conj"]) for s in sp.values() for sd in range(cfg["seeds"])]
    print(f"eval Estudo 22 {reg} ({POOL}): {len(tar)} execucoes ({', '.join(sp)})", flush=True)
    with mp.Pool(workers) as pool:
        rodar22(tar, RES / f"etapa22_eval_{reg}.csv", pool)
    print(f"eval Estudo 22 {reg} concluido", flush=True)


# ------------------------------------------------------------------------------------------------ regressao exata
def modo_regressao(workers):
    """(1) executar_22 com aceitacao 'sa' e as configuracoes congeladas de P21 reproduz EXATAMENTE (custo, iteracoes, avaliacoes, usos, escores) execucoes do Estudo 21 (U, RP, RPt, S33, S65,
    HG em X e SYN_VAL, instancias de ate 150 clientes); (2) os conjuntos A, B e CMT: 53 instancias com solucao inicial em disco e otimo conhecido."""
    import pandas as pd
    rs = random.Random(22)
    e21 = pd.read_csv(RES / f"etapa21_eval_{POOL}.csv")
    fr = congeladas()
    todas = sorted(e21[e21.n_customers <= 150].instancia.unique())              # instancias de ate 150 clientes: a regressao fica barata
    casos = []
    for a in ("U", "RP", "RPt", "S33", "S65", "HG"):
        if a not in fr:
            continue
        pol, params, pd_, pr = fr[a]
        for i in rs.sample(todas, 3):
            ref = e21[(e21.braco == f"s21_{POOL}_{a}") & (e21.instancia == i)]
            casos.append((a, spec22(f"reg_{a}", pol, params=params, pesos_d=pd_, pesos_r=pr), ref.iloc[rs.randrange(len(ref))]))
    tar = [(s, 1, lin.conjunto, lin.instancia, int(lin.seed)) for (_a, s, lin) in casos]
    with mp.Pool(min(workers, len(tar))) as pool:
        novas = pool.map(_worker22, tar)
    for (a, s, lin), nv in zip(casos, novas):
        assert nv["instancia"] == lin.instancia, (a, nv["instancia"], lin.instancia)
        for campo in r21.CAMPOS_REG:
            assert str(nv[campo]) == str(lin[campo]), (a, lin.instancia, lin.seed, campo, nv[campo], lin[campo])
        assert abs(nv["custo"] - lin["custo"]) < 1e-9 and bool(nv["viavel"]) == bool(lin["viavel"]), (a, lin.instancia, lin.seed)
    print(f"REGRESSAO EXATA: OK ({len(casos)} execucoes de P21 do Estudo 21 reproduzidas com o laco de aceitacao selecionavel em modo 'sa')", flush=True)
    tot = 0
    for c in ("A", "B", "CMT"):
        for n in r4.nomes(c):
            inst, cws = r9._instancia(c, n)
            assert inst.optimum and inst.optimum > 0 and sum(len(r) for r in cws) == inst.n_customers, (c, n)
            tot += 1
    assert tot == 53, tot
    print("CONJUNTOS: 53 instancias A, B e CMT com solucao inicial em disco e otimo conhecido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("regressao", "calibrar"):
        g.add_argument(f"--{m}", action="store_true")
    g.add_argument("--eval", choices=list(REG))
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.regressao:
        return modo_regressao(a.workers)
    if a.calibrar:
        return modo_calibrar(a.workers)
    return modo_eval(a.eval, a.workers)


if __name__ == "__main__":
    main()
