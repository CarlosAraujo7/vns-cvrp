#!/usr/bin/env python3
"""
run_aos_etapa31.py
==================
Estudo 31 da Direcao A (diario, Entrada 160 -- pre-registro): CLONAGEM DE OPERADORES. No ALNS do CVRP a mistura estatica ajustada com 33 avaliacoes (S33) iguala os pesos adaptativos de Ropke e Pisinger (RP) com 7
operadores (Estudo 12) e com 12 (Estudo 21) e perde para eles com 21 (Estudo 21); o conjunto de 21 operadores muda ao mesmo tempo o NUMERO de parametros da mistura (7, 12, 21) e a COMPOSICAO (operadores
diferentes), e a ameaca 1 do Estudo 28 e do suplemento diz que as comparacoes entre conjuntos sao de desenho. Aqui o conjunto de 7 operadores (4 destruicoes + 3 reparos, o do Estudo 9) tem cada operador
TRIPLICADO (copias identicas; 12 destruicoes + 9 reparos, a mesma forma de P21): o numero de parametros da mistura vai a 21 e a qualidade disponivel continua a de P7. Bracos novos (avaliacao: as 51 instancias e
seeds do Estudo 21; referencias do conjunto de 7 operadores: os Estudos 9 e 12, exatamente como no Estudo 21): U (uniforme sobre as 21 copias), RP (padrao, sobre as 21 copias), S33 (mistura estatica de 21
logits pelo (1+4)-ES de 8 geracoes = 33 avaliacoes de 72 execucoes em SYN_TUNE, seeds 3000-3002). Modos: --regressao | --ajuste | --selecionar | --eval (retomaveis por chave).
Uso (WSL): python3 -u run_aos_etapa31.py --ajuste --workers 11
"""
import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa18 as r18  # noqa: E402
import run_aos_etapa21 as r21  # noqa: E402
import run_aos_etapa23 as r23  # noqa: E402
import run_aos_etapa4 as r4  # noqa: E402
from politicas_contexto import softmax_np  # noqa: E402

RES = r23.RES
SEEDS_TUNE = r23.SEEDS_TUNE                       # 3000-3002
SEMENTES_AVAL = 1000
KD, KR = 12, 9                                    # 4 x 3 destruicoes e 3 x 3 reparos
SIGMA0 = r18.ES_SIGMA0_STATIC                     # 0,5
SEMENTE_ES = 20263101
GER_S33 = 8                                       # 1 + 4 x 8 = 33 avaliacoes
TUNE_CONJ = "SYN_TUNE"
NOME_TUNE = "tune31_static"
PADRAO_RP = dict(r21.PADRAO_RP)
ESTADO = RES / "etapa31_ajuste_static_estado.json"
CSV_AJUSTE = RES / "etapa31_ajuste_static.csv"


def spec31(nome, pol, tipo, params=None, pesos_d=None, pesos_r=None, config_id="e31"):
    sp = r23.spec23(nome, pol, params=params, config_id=config_id)
    sp["pesos_d"], sp["pesos_r"] = pesos_d, pesos_r
    return r23.com_id(sp, f"P7C21_{tipo}")


def tarefas_tune(sp):
    return [(sp, 1, TUNE_CONJ, n, s) for n in sorted(r4.nomes(TUNE_CONJ), reverse=True) for s in SEEDS_TUNE]


def pesos_de_theta(theta):
    th = np.asarray(theta, float)
    return list(map(float, softmax_np(th[:KD]))), list(map(float, softmax_np(th[KD:])))


def pesos_eval(theta):
    pd_, pr = pesos_de_theta(theta)
    return [round(x, 4) for x in pd_], [round(x, 4) for x in pr]


# ------------------------------------------------------------------------------------------------ regressao
def modo_regressao(workers):
    """(i) a tabela clonada tem 12 destruicoes e 9 reparos, cada operador de P7 tres vezes, com as MESMAS funcoes; (ii) o executor modificado reproduz EXATAMENTE 6 execucoes do Estudo 21 em P21 (3 de U e 3 de RP
    em instancias de ate 150 clientes), isto e, a mudanca em `tabelas` nao altera os bracos antigos."""
    import pandas as pd
    D, R = r23.tabelas("P7C21_U")
    assert len(D) == KD and len(R) == KR
    assert all(D[i] is r23.D7[i % 4] for i in range(KD)) and all(R[i] is r23.R7[i % 3] for i in range(KR))
    rs = random.Random(31)
    e21 = pd.read_csv(RES / "etapa21_eval_P21.csv")
    casos = []
    for a, pol, params in (("U", "U", {}), ("RP", "RP", dict(PADRAO_RP))):
        d = e21[(e21.braco == f"s21_P21_{a}") & (e21.n_customers <= 150)]
        for i in rs.sample(sorted(d.instancia.unique()), 3):
            ref = d[d.instancia == i]
            casos.append((r23.com_id(r23.spec23(f"reg_{a}", pol, params=params), a), ref.iloc[rs.randrange(len(ref))]))
    tar = [(s, 1, lin.conjunto, lin.instancia, int(lin.seed)) for (s, lin) in casos]
    with mp.Pool(min(workers, len(tar))) as pool:
        novas = pool.starmap(r23.executar_23, tar)
    for (s, lin), nv in zip(casos, novas):
        assert nv["instancia"] == lin.instancia
        for campo in r21.CAMPOS_REG:
            assert str(nv[campo]) == str(lin[campo]), (s["nome"], lin.instancia, lin.seed, campo, nv[campo], lin[campo])
        assert abs(nv["custo"] - lin["custo"]) < 1e-9 and bool(nv["viavel"]) == bool(lin["viavel"])
    print(f"REGRESSAO EXATA: OK (tabela clonada conferida; {len(casos)} execucoes do Estudo 21 reproduzidas)", flush=True)


# ------------------------------------------------------------------------------------------------ ajuste da mistura (ES de 21 logits)
def modo_ajuste(workers):
    n_ref = len(r4.nomes(TUNE_CONJ)) * len(SEEDS_TUNE)
    with mp.Pool(workers) as pool:
        def avaliar(theta, cid):
            pd_, pr = pesos_de_theta(theta)                                      # precisao total, como na busca original
            sp = spec31(NOME_TUNE, "static", "tune", pesos_d=pd_, pesos_r=pr, config_id=cid)
            sp["params"] = {"theta": list(map(float, theta))}
            r23.rodar23(tarefas_tune(sp), CSV_AJUSTE, pool)
            m, n = r18.r16.media_gap(CSV_AJUSTE, NOME_TUNE, cid)
            assert n == n_ref, (cid, n, n_ref)
            return m

        r18.ES_GERACOES_18 = GER_S33
        r18.es18(ESTADO, [0.0] * (KD + KR), SIGMA0, SEMENTE_ES, avaliar, "P7C21/static")
        assert json.loads(ESTADO.read_text())["geracao"] == GER_S33


def trajetoria_es():
    """Incumbentes do ES por geracao, reconstruidos do CSV de ajuste: [(geracao, theta, gap de ajuste, n avaliacoes)]."""
    import pandas as pd
    d = pd.read_csv(CSV_AJUSTE)
    d = d[(d.braco == NOME_TUNE) & d.viavel]
    g = d.groupby("config_id").agg(gap=("gap_pct", "mean"), params=("params", "first"), n=("gap_pct", "size"))
    assert (g.n == len(r4.nomes(TUNE_CONJ)) * len(SEEDS_TUNE)).all()
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
    return traj


def modo_selecionar():
    traj = trajetoria_es()
    est = json.loads(ESTADO.read_text())
    t = traj[-1]
    assert t[0] == GER_S33 and est["theta"] == t[1] and abs(est["melhor"] - t[2]) < 1e-9, "a trajetoria reconstruida do CSV nao coincide com o estado do ES"
    pd_, pr = pesos_eval(t[1])
    esc = {"trajetoria": [{"geracao": g, "n_avaliacoes": n, "gap_ajuste": gp, "theta": th} for g, th, gp, n in traj], "S33": {"theta": t[1], "pesos_d": pd_, "pesos_r": pr, "gap_ajuste": t[2], "n_avaliacoes": t[3]}}
    (RES / "etapa31_selecao.json").write_text(json.dumps(esc, indent=1))
    print(f"S33 (clones) gap de ajuste {t[2]:.4f} ({t[3]} aval.); curva do ES: " + ", ".join(f"g{g}:{gp:.3f}" for g, _th, gp, _n in traj))
    print(f"pesos de destruicao {pd_}; de reparo {pr}", flush=True)


# ------------------------------------------------------------------------------------------------ avaliacao
def specs_eval():
    esc = json.loads((RES / "etapa31_selecao.json").read_text())
    return {"U": spec31("s31_U", "U", "U"), "RP": spec31("s31_RP", "RP", "RP", params=dict(PADRAO_RP)),
            "S33": spec31("s31_S33", "static", "S33", pesos_d=esc["S33"]["pesos_d"], pesos_r=esc["S33"]["pesos_r"])}


def modo_eval(workers):
    sp = specs_eval()
    tar = [(s, 1, c, n, SEMENTES_AVAL + sd) for c, n, ns, _nc in r21.instancias_aval() for s in sp.values() for sd in range(ns)]
    print(f"eval Estudo 31 (7 operadores triplicados): {len(tar)} execucoes ({', '.join(sp)})", flush=True)
    with mp.Pool(workers) as pool:
        r23.rodar23(tar, RES / "etapa31_eval.csv", pool)
    print("eval Estudo 31 concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("regressao", "ajuste", "selecionar", "eval"):
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.regressao:
        return modo_regressao(a.workers)
    if a.ajuste:
        return modo_ajuste(a.workers)
    if a.selecionar:
        return modo_selecionar()
    return modo_eval(a.workers)


if __name__ == "__main__":
    main()
