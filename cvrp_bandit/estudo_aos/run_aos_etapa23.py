#!/usr/bin/env python3
"""
run_aos_etapa23.py
==================
Estudo 23 da Direcao A (diario, Entrada 141 -- pre-registro): FIDELIDADE E MECANISMO da vantagem dos pesos adaptativos de Ropke e Pisinger (RP) no conjunto de 21 operadores (P21; Estudo 21).
Duas descobertas pos hoc do Estudo 21 motivam o estudo: (1) o RP concentra 75% das destruicoes em operadores leves (rota_pequena 30%, aleatoria_leve 18%, rota 16%, pior_leve 11%) e faz 2,3 vezes
mais iteracoes no mesmo orcamento; (2) esses operadores devolvem a MESMA solucao em 86%, 65%, 18% e 53% das aplicacoes (diag_noop_p21.py, instancias de ajuste), e o laco dos Estudos 9 a 22
pontua como 'aceito pior' (sigma3 = 13) todo candidato aceito, inclusive os identicos ao corrente; Ropke e Pisinger (2006) so pontuam solucoes NAO aceitas antes (tabela de hashes).
Bracos novos (avaliacao: as 51 instancias e seeds do Estudo 21; as referencias U, RP, S33, S65 e S129 vem dele e do P7 dos Estudos 9, 12 e 20):
  RPh     P21: RP (rho 0,1; segmento 100) com a pontuacao FIEL (novidade = True)
  RPh7    P7 : o mesmo no conjunto de 7 operadores dos Estudos 9 a 20
  RPsr    P21 sem rota_pequena (20 operadores): RP padrao com a pontuacao dos Estudos 9 a 22
  Uq      P21: selecao uniforme com a faixa de remocao q multiplicada por c* (c* em {0,25; 0,4; 0,6; 1}, escolhido em SYN_TUNE, seeds 3000-3002, pelo menor gap medio; se c* = 1 o braco = U)
  Upr     P21: selecao uniforme sobre o conjunto PODADO: destruicoes com participacao do RP >= 1/12 e reparos com participacao >= 1/9 nas 72 execucoes de VALIDACAO do RP (instancias de ajuste)
  SRP     P21: a mistura estatica com o uso medio do RP nessas mesmas execucoes (rodado por run_aos_etapa21c.py; reutilizado)
Modos: --calibrar | --selecionar | --eval {p21,p7}   (retomaveis por chave). Uso (WSL): python3 -u run_aos_etapa23.py --eval p21 --workers 11
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
SEEDS_TUNE, SEMENTES_AVAL = r20.SEEDS_TUNE, 1000
ESCALAS = (0.25, 0.4, 0.6, 1.0)
LIM_D, LIM_R = 1 / 12, 1 / 9                      # participacao minima do RP para manter uma destruicao / um reparo no conjunto podado
D21, R21 = ax.POOLS["P21"]
D7, R7 = ax.POOLS["P7"]
ND, NR = [n for n, _ in D21], [n for n, _ in R21]


# ------------------------------------------------------------------------------------------------ conjuntos e especificacoes
def uso_rp_validacao():
    """Participacao media (fracao) dos operadores de P21 no RP padrao, nas 72 execucoes de validacao do Estudo 21 (instancias de ajuste, seeds 3003-3005)."""
    import pandas as pd
    v = pd.read_csv(RES / "etapa21_validacao_P21.csv")
    v = v[(v.braco == "s21_P21_RP") & v.viavel]
    assert len(v) == 72, len(v)
    out = []
    for col in ("uso_d", "uso_r"):
        m = pd.DataFrame(v[col].str.split("|").apply(lambda s: [float(x) for x in s]).tolist()).mean()
        out.append([float(x) for x in m / m.sum()])
    return out


def podado():
    """(indices de destruicao mantidos, indices de reparo mantidos) pela regra pre-registrada; ordem original."""
    ud, ur = uso_rp_validacao()
    return [j for j, u in enumerate(ud) if u >= LIM_D], [j for j, u in enumerate(ur) if u >= LIM_R]


def tabelas(braco):
    """(destruir, reparar) do braco: P21 inteiro, P21 sem rota_pequena, P21 podado ou P7."""
    if braco == "RPh7":
        return D7, R7
    if braco == "RPsr":
        return [x for x in D21 if x[0] != "rota_pequena"], list(R21)
    if braco == "Upr":
        jd, jr = podado()
        return [D21[j] for j in jd], [R21[j] for j in jr]
    return D21, R21


def spec23(nome, pol, params=None, q_escala=1.0, novidade=False, config_id="e23"):
    return {"nome": nome, "pol": pol, "params": params or {}, "pesos_d": None, "pesos_r": None, "kappa": r9.KAPPA9, "w0": r9.W09, "tf": r9.TF9, "config_id": config_id,
            "regime": "a", "problema": "CVRP", "q_escala": q_escala, "novidade": novidade}


def executar_23(sp, nivel, conj, nome_inst, seed):
    """Como executar_21 (sementes de politica seed * 31 + 1 e + 2), com tabelas de operadores por braco, pontuacao com novidade e escala de q opcionais."""
    D, R = tabelas(sp["bracoid"])
    inst, cws = r9._instancia(conj, nome_inst)
    obj = r9._proxy(conj, nome_inst)
    max_rotas = inst.n_vehicles_ref if inst.vehicles_exact else None
    obj.chamadas = 0
    rng = random.Random(seed)
    kd, kr = len(D), len(R)
    pol_d, md = r18.fabricar(sp, "d", kd, random.Random(seed * 31 + 1))
    pol_r, mr = r18.fabricar(sp, "r", kr, random.Random(seed * 31 + 2))
    orcamento = int(nivel * sp["kappa"] * inst.n_customers ** 2)
    t0 = time.perf_counter()
    best, info = rodar_alns_aceitacao(obj, [list(r) for r in cws], rng, orcamento, pol_d, md, pol_r, mr, lambda: obj.chamadas, w0=sp["w0"], tf_fator=sp["tf"],
                                      max_rotas=max_rotas, destruir=D, reparar=R, novidade=sp["novidade"], q_escala=sp["q_escala"])
    dt = time.perf_counter() - t0
    custo = inst.solution_cost(best)
    viavel, _ = inst.is_feasible(best)
    j = lambda v: "|".join(f"{x:g}" for x in v)  # noqa: E731
    params = dict(sp["params"])
    params.update({"bracoid": sp["bracoid"], "q_escala": sp["q_escala"], "novidade": sp["novidade"]})
    return {"braco": sp["nome"], "config_id": sp["config_id"], "params": json.dumps(params, sort_keys=True), "nivel": nivel, "conjunto": conj,
            "instancia": inst.name, "n_customers": inst.n_customers, "seed": seed, "orcamento": orcamento, "custo": custo, "ref": inst.optimum,
            "gap_pct": 100 * (custo - inst.optimum) / inst.optimum, "viavel": viavel, "n_rotas": len([r for r in best if r]), "calls_usadas": obj.chamadas,
            "iteracoes": info["iteracoes"], "inviaveis": info["inviaveis"], "uso_d": j(info["uso_d"]), "uso_r": j(info["uso_r"]), "score_d": j(info["score_d"]),
            "score_r": j(info["score_r"]), "calls_d": j(info["calls_d"]), "calls_r": j(info["calls_r"]), "tempo_s": dt, "uso_d_fase": "", "uso_r_fase": ""}


def _worker23(args):
    return executar_23(*args)


def rodar23(tarefas, out_csv, pool, cada=100):
    feitos = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            feitos = {(r["braco"], r["config_id"], float(r["nivel"]), r["instancia"], int(r["seed"])) for r in csv.DictReader(f)}
    pend = [t for t in tarefas if (t[0]["nome"], t[0]["config_id"], float(t[1]), t[3], t[4]) not in feitos]
    if not pend:
        return 0
    novo = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=CAMPOS)
        if novo:
            w.writeheader()
        t0, feitas = time.perf_counter(), 0
        for linha in pool.imap_unordered(_worker23, pend, chunksize=1):
            w.writerow(linha)
            feitas += 1
            if feitas % cada == 0:
                f_out.flush()
                print(f"    {feitas}/{len(pend)}  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return len(pend)


def com_id(sp, bracoid):
    sp = dict(sp)
    sp["bracoid"] = bracoid
    return sp


# ------------------------------------------------------------------------------------------------ calibracao da escala de q, selecao e avaliacao
def modo_calibrar(workers):
    """c* da faixa de remocao: a selecao uniforme de P21 em SYN_TUNE (24 instancias x seeds 3000-3002) para cada escala; o de menor gap medio (empate: o maior c, isto e, o mais proximo de U)."""
    out = RES / "etapa23_calibracao_q.csv"
    nome = "cal23_Uq"
    tar = []
    for c in ESCALAS:
        sp = com_id(spec23(nome, "U", q_escala=c, config_id=f"c{c}"), "Uq")
        tar += [(sp, 1, "SYN_TUNE", n, s) for n in sorted(r4.nomes("SYN_TUNE"), reverse=True) for s in SEEDS_TUNE]
    print(f"calibracao da escala de q: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        rodar23(tar, out, pool)
    import pandas as pd
    d = pd.read_csv(out)
    g = d[d.viavel].groupby("config_id").gap_pct.mean()
    print(g.round(4).to_string())
    melhor = min(ESCALAS, key=lambda c: (round(float(g[f"c{c}"]), 12), -c))
    jd, jr = podado()
    esc = {"c_estrela": melhor, "gap_ajuste_Uq": {str(c): float(g[f"c{c}"]) for c in ESCALAS}, "podado_destruir": [ND[j] for j in jd], "podado_reparar": [NR[j] for j in jr],
           "uso_rp_destruir": dict(zip(ND, uso_rp_validacao()[0])), "uso_rp_reparar": dict(zip(NR, uso_rp_validacao()[1]))}
    (RES / "etapa23_selecao.json").write_text(json.dumps(esc, indent=1))
    print(f"c* = {melhor}; conjunto podado: destruir {esc['podado_destruir']} reparar {esc['podado_reparar']}", flush=True)


def specs(reg):
    esc = json.loads((RES / "etapa23_selecao.json").read_text())
    p = lambda a: f"s23_{reg}_{a}"  # noqa: E731
    if reg == "p7":
        return {"RPh7": com_id(spec23(p("RPh7"), "RP", params=dict(r21.PADRAO_RP), novidade=True), "RPh7")}
    out = {"RPh": com_id(spec23(p("RPh"), "RP", params=dict(r21.PADRAO_RP), novidade=True), "RPh"),
           "RPsr": com_id(spec23(p("RPsr"), "RP", params=dict(r21.PADRAO_RP)), "RPsr"),
           "Upr": com_id(spec23(p("Upr"), "U"), "Upr")}
    if esc["c_estrela"] != 1.0:
        out["Uq"] = com_id(spec23(p("Uq"), "U", q_escala=esc["c_estrela"]), "Uq")
    return out


def modo_eval(reg, workers):
    sp = specs(reg)
    tar = [(s, 1, c, n, SEMENTES_AVAL + sd) for c, n, ns, _nc in r21.instancias_aval() for s in sp.values() for sd in range(ns)]
    print(f"eval Estudo 23 {reg}: {len(tar)} execucoes ({', '.join(sp)})", flush=True)
    with mp.Pool(workers) as pool:
        rodar23(tar, RES / f"etapa23_eval_{reg}.csv", pool)
    print(f"eval Estudo 23 {reg} concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--calibrar", action="store_true")
    g.add_argument("--eval", choices=("p21", "p7"))
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.calibrar:
        return modo_calibrar(a.workers)
    return modo_eval(a.eval, a.workers)


if __name__ == "__main__":
    main()
