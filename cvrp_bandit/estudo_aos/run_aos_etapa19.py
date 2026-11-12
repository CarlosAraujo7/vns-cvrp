#!/usr/bin/env python3
"""
run_aos_etapa19.py
==================
Estudo 19 da Direcao A (diario, Entrada 129 -- pre-registro): REPLICACAO CONFIRMATORIA do unico indicio do Estudo 18, que os pesos adaptativos de
Ropke e Pisinger MODULADOS pela mistura estatica ajustada (`tilt`) sao melhores que a propria mistura (`static18`): -0,14 pp em Solomon, p = 0,016
sem correcao e 0,17 com Holm sobre 15 contrastes (o Estudo 18 pre-registrou H104 e a refutou). Configuracoes CONGELADAS do Estudo 18 (nenhum reajuste),
mesmo regime ALNS-CVRPTW (kappa 1.500, w0 0,2), quatro bracos (U, static18, RP18, tilt) em dois conjuntos que o Estudo 18 nao usou:
  SYN_VAL_W : 48 instancias sinteticas NOVAS (k = 48..95 do mesmo gerador; referencias do PyVRP), seeds 5000-5009;
  SOLOMON   : as 56 do Estudo 18 com seeds NOVAS 2000-2009 (as do Estudo 18 foram 1000-1009).
Modos: --regressao (reproduz exatamente execucoes do Estudo 18 com as configuracoes congeladas) | --eval (retomavel).
Uso (WSL): python3 -u run_aos_etapa19.py --regressao ; python3 -u run_aos_etapa19.py --eval --workers 11
"""
import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_aos_etapa18 as r18  # noqa: E402
import run_aos_etapa14 as r14  # noqa: E402

RES = r18.RES
CONJ_NOVO = "SYN_VAL_W"
SEEDS_SYN = list(range(5000, 5010))
SEEDS_SOL = list(range(2000, 2010))


def specs19():
    """Configuracoes CONGELADAS do Estudo 18 (etapa18_tuning_escolhidas.json); nomes s19_*."""
    esc = json.loads((RES / "etapa18_tuning_escolhidas.json").read_text())
    return {"s19_U": r18.spec18("s19_U", "U"),
            "s19_static18": r18.spec18("s19_static18", "static", pesos_d=esc["static18"]["pesos_d"], pesos_r=esc["static18"]["pesos_r"]),
            "s19_RP18": r18.spec18("s19_RP18", "RP", params=esc["RP18"]["params"]),
            "s19_tilt": r18.spec18("s19_tilt", "tilt", params=esc["tilt"]["params"], pesos_d=esc["tilt"]["pesos_d"], pesos_r=esc["tilt"]["pesos_r"])}


def modo_regressao(workers):
    """As configuracoes congeladas reproduzem EXATAMENTE as execucoes do Estudo 18 (mesmas instancias e seeds 1000-1009): custo, iteracoes,
    avaliacoes, usos e escores; o braco U, as do Estudo 16."""
    import pandas as pd
    e18 = pd.read_csv(RES / "etapa18_eval.csv")
    e16 = pd.read_csv(RES / "etapa16_eval_a.csv")
    sp = specs19()
    rs = random.Random(7)
    insts = sorted(e18.instancia.unique())
    alvo = []
    for arm in ("tilt", "static18", "RP18", "U"):
        for i in rs.sample(insts, 3):
            alvo.append((arm, i, 1000 + rs.randrange(10)))
    tar = [(sp[f"s19_{a}"], 1, "SOLOMON", i, s) for a, i, s in alvo]
    with mp.Pool(min(workers, len(tar))) as pool:
        novas = pool.map(r18._worker18, tar)
    n_ok = 0
    for (a, i, s), nv in zip(alvo, novas):
        ref = (e16[(e16.braco == "s16a_U") & (e16.instancia == i) & (e16.seed == s)] if a == "U"
               else e18[(e18.braco == f"s18_{a}") & (e18.instancia == i) & (e18.seed == s)])
        assert len(ref) == 1, (a, i, s)
        ref = ref.iloc[0]
        for campo in ("iteracoes", "calls_usadas", "inviaveis", "n_rotas", "uso_d", "uso_r", "score_d", "score_r", "calls_d", "calls_r"):
            assert str(nv[campo]) == str(ref[campo]), (a, i, s, campo, nv[campo], ref[campo])
        assert abs(nv["custo"] - ref["custo"]) < 1e-9, (a, i, s, nv["custo"], ref["custo"])
        n_ok += 1
    print(f"REGRESSAO EXATA: OK ({n_ok} execucoes do Estudo 18 / 16 reproduzidas com as configuracoes congeladas)")


def modo_eval(workers):
    sp = specs19()
    tar = [(s, 1, "SOLOMON", n, sd) for n in r14.nomes("SOLOMON") for s in sp.values() for sd in SEEDS_SOL]
    tar += [(s, 1, CONJ_NOVO, n, sd) for n in r14.nomes(CONJ_NOVO) for s in sp.values() for sd in SEEDS_SYN]
    tar.sort(key=lambda t: t[3])
    print(f"eval Estudo 19: {len(tar)} execucoes", flush=True)
    with mp.Pool(workers) as pool:
        r18.rodar18(tar, RES / "etapa19_eval.csv", pool)
    print("eval Estudo 19 concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--regressao", action="store_true")
    g.add_argument("--eval", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    (modo_regressao if a.regressao else modo_eval)(a.workers)


if __name__ == "__main__":
    main()
