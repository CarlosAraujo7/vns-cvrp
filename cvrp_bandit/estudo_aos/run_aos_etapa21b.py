#!/usr/bin/env python3
"""
run_aos_etapa21b.py
===================
EMENDA 1 ao pre-registro do Estudo 21 (diario, Entrada 139), antes de qualquer analise dos dados de avaliacao: o braco S129, a mistura estatica de P21 com QUATRO vezes o esforco
de ajuste de S33 (a mesma busca (1+4)-ES de 8 geracoes = 33 avaliacoes, continuada ate 32 geracoes = 129 avaliacoes, mesmas instancias de ajuste, seeds 3000-3002 e sequencia
aleatoria; o estado da geracao 16 e COPIADO, o arquivo original nao e tocado). Modulo separado de run_aos_etapa21.py para nao tocar o codigo em uso pela fila do Estudo 21.
Modos: --ajuste | --selecionar | --validar | --eval   (retomaveis por chave). Uso (WSL): python3 -u run_aos_etapa21b.py --ajuste --workers 11
"""
import argparse
import json
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa21 as r21  # noqa: E402
import run_aos_etapa18 as r18  # noqa: E402

RES = r21.RES
POOL = "P21"
GER_FINAL = 32                                   # 1 + 4 x 32 = 129 avaliacoes
GER_CURVA_B = (20, 24, 28, 32)
CSV_B = RES / "etapa21b_ajuste_static_P21.csv"
EST_B = RES / "etapa21b_ajuste_static_P21_estado.json"
ESC_B = RES / "etapa21b_ajuste_escolhidas_P21.json"
NOME = f"tune21_{POOL}_static"


def modo_ajuste(workers):
    P = r21.PROB21[POOL]
    kd, kr = len(P["destruir"]), len(P["reparar"])
    n_ref = len(r21.nomes(P["tune_conj"])) * len(r21.SEEDS_TUNE)
    if not EST_B.exists():
        est = json.loads(r21.estado_es(POOL).read_text())
        assert est["geracao"] == r21.GER_S65, est["geracao"]                  # o estado final do ajuste original (65 avaliacoes)
        shutil.copy(r21.estado_es(POOL), EST_B)
    with mp.Pool(workers) as mpool:
        def avaliar(theta, cid):
            pd_, pr = r21.pesos_de_theta(theta, kd)
            sp = r21.spec21(POOL, NOME, "static", pesos_d=pd_, pesos_r=pr, config_id=cid)
            sp["params"] = {"theta": list(map(float, theta))}
            r21.rodar21(r21.tarefas_ajuste(POOL, sp, r21.SEEDS_TUNE), CSV_B, mpool)
            m, n = r18.r16.media_gap(CSV_B, NOME, cid)
            assert n == n_ref, (cid, n, n_ref)
            return m

        r18.ES_GERACOES_18 = GER_FINAL
        r18.es18(EST_B, [0.0] * (kd + kr), r21.SIGMA0, r21.SEMENTE_ES[POOL], avaliar, f"{POOL}/static129")


def trajetoria_ext():
    """Incumbentes do ES de 0 a 32 geracoes (CSV original + continuacao): [(geracao, theta, gap de ajuste, n avaliacoes)]."""
    import pandas as pd
    d = pd.concat([pd.read_csv(r21.csv_ajuste(POOL, "static")), pd.read_csv(CSV_B)], ignore_index=True)
    d = d[(d.braco == NOME) & d.viavel]
    g = d.groupby("config_id").agg(gap=("gap_pct", "mean"), params=("params", "first"), n=("gap_pct", "size"))
    assert (g.n == len(r21.nomes(r21.PROB21[POOL]["tune_conj"])) * len(r21.SEEDS_TUNE)).all()
    atual_theta, atual_gap = json.loads(g.loc["gen00", "params"])["theta"], float(g.loc["gen00", "gap"])
    traj = [(0, atual_theta, atual_gap, 1)]
    for gen in range(1, GER_FINAL + 1):
        filhos = [(float(g.loc[c, "gap"]), c) for c in g.index if c.startswith(f"gen{gen:02d}c")]
        assert len(filhos) == 4, (gen, len(filhos))
        gap_f, cid = min(filhos)
        if gap_f < atual_gap:
            atual_gap, atual_theta = gap_f, json.loads(g.loc[cid, "params"])["theta"]
        traj.append((gen, atual_theta, atual_gap, 1 + 4 * gen))
    return traj


def modo_selecionar():
    kd = len(r21.PROB21[POOL]["destruir"])
    traj = trajetoria_ext()
    por = {t[0]: t for t in traj}
    est = json.loads(EST_B.read_text())
    assert est["geracao"] == GER_FINAL and est["theta"] == por[GER_FINAL][1] and abs(est["melhor"] - por[GER_FINAL][2]) < 1e-9
    ant = json.loads((RES / f"etapa21_ajuste_escolhidas_{POOL}.json").read_text())
    assert ant["S65"]["theta"] == por[r21.GER_S65][1], "o incumbente da geracao 16 nao coincide com S65 da selecao original"
    pd129, pr129 = r21.pesos_eval(por[GER_FINAL][1], kd)
    esc = {"trajetoria": [{"geracao": g, "n_avaliacoes": n, "gap_ajuste": gp, "theta": th} for g, th, gp, n in traj],
           "S129": {"mudou": bool(por[GER_FINAL][1] != por[r21.GER_S65][1]), "theta": por[GER_FINAL][1], "pesos_d": pd129, "pesos_r": pr129,
                    "gap_ajuste": por[GER_FINAL][2], "n_avaliacoes": por[GER_FINAL][3]}}
    ESC_B.write_text(json.dumps(esc, indent=1))
    print(f"{POOL}: S129 gap de ajuste {por[GER_FINAL][2]:.4f} (129 aval.; {'MUDOU' if esc['S129']['mudou'] else 'sem melhora: S129 = S65'}); S65 {por[16][2]:.4f}; S33 {por[8][2]:.4f}")
    print("curva do ES (gap de ajuste do incumbente): " + ", ".join(f"g{g}:{gp:.3f}" for g, _th, gp, _n in traj))


def modo_validar(workers):
    kd = len(r21.PROB21[POOL]["destruir"])
    esc = json.loads(ESC_B.read_text())
    sp = {}
    nome = f"s21_{POOL}_S129"
    sp[nome] = r21.spec21(POOL, nome, "static", pesos_d=esc["S129"]["pesos_d"], pesos_r=esc["S129"]["pesos_r"])
    for t in esc["trajetoria"]:
        if t["geracao"] in GER_CURVA_B:
            pd_, pr = r21.pesos_de_theta(t["theta"], kd)
            n_ = f"val21_{POOL}_gen{t['geracao']:02d}"
            sp[n_] = r21.spec21(POOL, n_, "static", pesos_d=pd_, pesos_r=pr)
    tar = [t for s in sp.values() for t in r21.tarefas_ajuste(POOL, s, r21.SEEDS_VALID)]
    print(f"validacao (emenda 1, {POOL}): {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as mpool:
        r21.rodar21(tar, RES / f"etapa21b_validacao_{POOL}.csv", mpool)
    import pandas as pd
    d = pd.read_csv(RES / f"etapa21b_validacao_{POOL}.csv")
    print(d[d.viavel].groupby("braco").gap_pct.mean().sort_values().round(4).to_string())


def modo_eval(workers):
    esc = json.loads(ESC_B.read_text())
    if not esc["S129"]["mudou"]:
        print("S129 = S65 (a continuacao nao achou melhora): nada a executar")
        return
    nome = f"s21_{POOL}_S129"
    s = r21.spec21(POOL, nome, "static", pesos_d=esc["S129"]["pesos_d"], pesos_r=esc["S129"]["pesos_r"])
    tar = [(s, 1, c, n, r21.SEMENTES_AVAL + sd) for c, n, ns, _nc in r21.instancias_aval() for sd in range(ns)]
    print(f"eval Estudo 21 emenda 1 ({POOL}): {len(tar)} execucoes ({nome})", flush=True)
    with mp.Pool(workers) as mpool:
        r21.rodar21(tar, RES / f"etapa21b_eval_{POOL}.csv", mpool)
    print("eval emenda 1 concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("ajuste", "selecionar", "validar", "eval"):
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.ajuste:
        return modo_ajuste(a.workers)
    if a.selecionar:
        return modo_selecionar()
    if a.validar:
        return modo_validar(a.workers)
    return modo_eval(a.workers)


if __name__ == "__main__":
    main()
