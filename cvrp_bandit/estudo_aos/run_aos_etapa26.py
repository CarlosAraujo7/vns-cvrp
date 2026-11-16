#!/usr/bin/env python3
"""
run_aos_etapa26.py
==================
Estudo 26 da Direcao A (diario, Entradas 145 e 146 -- pre-registro): REPLICA DAS CONCLUSOES DE P21 EM INSTANCIAS NOVAS. Os Estudos 21, 23, 24 e 25 usam as MESMAS 51 instancias de avaliacao (X de 100 a 195
clientes e SYN_VAL), e a analise exploratoria da folga e a escolha dos regimes do Estudo 25 foram feitas depois de ver essas instancias. Aqui as configuracoes CONGELADAS do Estudo 21 (e do Estudo 23)
sao rodadas em 98 instancias que nenhum braco dos Estudos 1 a 25 usou:
  X_FRESH  38 instancias X de Uchoa et al. (2017) com 200 a 401 clientes (do repositorio publico vidalt/HGS-CVRP; referencia = melhor de 2 execucoes do HGS-CVRP, 0,4 s por cliente)
  SYN_REP  60 instancias sinteticas (20 de 100, 20 de 200 e 20 de 300 clientes; o gerador validado, semente 20261601; referencia do HGS como em SYN_VAL)
Bracos (todos congelados, P21, recozimento, 1.500 n^2, a pontuacao dos Estudos 9 a 22): U; Uq (uniforme com a faixa de remocao q escalada por c* do Estudo 23); RP (pesos adaptativos, rho 0,1, segmento 100);
S65 (a mistura estatica de maior esforco do Estudo 21; S129 = S65); HG (HG-UCB do Estudo 9). Seeds 1000 e 1001. Modos: --regressao | --instancias | --eval (retomavel por chave).
Uso (WSL): python3 -u run_aos_etapa26.py --eval --workers 11
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
import run_aos_etapa9 as r9  # noqa: E402
import run_aos_etapa4 as r4  # noqa: E402
import run_aos_etapa21 as r21  # noqa: E402
import run_aos_etapa22 as r22  # noqa: E402
import run_aos_etapa23 as r23  # noqa: E402

RES = r23.RES
CONJ26 = ("X_FRESH", "SYN_REP")
N_INST = {"X_FRESH": 38, "SYN_REP": 60}
SEEDS26 = 2
SEMENTES_AVAL = 1000
BRACOS26 = ("U", "Uq", "RP", "S65", "HG")


def specs26():
    fr = r22.congeladas()
    esc23 = json.loads((RES / "etapa23_selecao.json").read_text())
    out = {}
    for a in ("U", "RP", "S65", "HG"):
        pol, params, pd_, pr = fr[a]
        sp = r23.spec23(f"s26_{a}", pol, params=params, config_id="e26")
        sp["pesos_d"], sp["pesos_r"] = pd_, pr
        out[a] = r23.com_id(sp, a)
    out["Uq"] = r23.com_id(r23.spec23("s26_Uq", "U", q_escala=esc23["c_estrela"], config_id="e26"), "Uq")
    return {a: out[a] for a in BRACOS26}


def instancias26():
    out = [(c, n, r9._instancia(c, n)[0].n_customers) for c in CONJ26 for n in r4.nomes(c)]
    return sorted(out, key=lambda t: -t[2])                                  # as maiores primeiro (balanceia a carga)


def modo_instancias():
    """Confere os dois conjuntos: numero de instancias, solucao inicial em disco, referencia positiva, cobertura dos clientes e tamanhos."""
    tot = {}
    for c in CONJ26:
        tam = []
        for n in r4.nomes(c):
            inst, cws = r9._instancia(c, n)
            assert inst.optimum and inst.optimum > 0 and sum(len(r) for r in cws) == inst.n_customers, (c, n)
            ok, motivo = inst.is_feasible(cws)
            assert ok, (c, n, motivo)
            tam.append(inst.n_customers)
        tot[c] = len(tam)
        print(f"{c}: {len(tam)} instancias; clientes de {min(tam)} a {max(tam)}", flush=True)
    assert tot == N_INST, tot
    print("INSTANCIAS: OK", flush=True)


def modo_regressao(workers):
    """As execucoes do Estudo 26 com as configuracoes congeladas reproduzem EXATAMENTE (custo, iteracoes, avaliacoes, usos) execucoes do Estudo 21 (U, RP, S65, HG) e do Estudo 23 (Uq) nas mesmas
    instancias e seeds (instancias de ate 150 clientes)."""
    import pandas as pd
    rs = random.Random(26)
    e21 = pd.read_csv(RES / "etapa21_eval_P21.csv")
    e23 = pd.read_csv(RES / "etapa23_eval_p21.csv")
    sp = specs26()
    casos = []
    for a in BRACOS26:
        e, nome = (e23, "s23_p21_Uq") if a == "Uq" else (e21, f"s21_P21_{a}")
        todas = sorted(e[(e.braco == nome) & (e.n_customers <= 150)].instancia.unique())
        for i in rs.sample(todas, 3):
            ref = e[(e.braco == nome) & (e.instancia == i)]
            casos.append((a, ref.iloc[rs.randrange(len(ref))]))
    tar = [(sp[a], 1, lin.conjunto, lin.instancia, int(lin.seed)) for a, lin in casos]
    with mp.Pool(min(workers, len(tar))) as pool:
        novas = pool.starmap(r23.executar_23, tar)
    for (a, lin), nv in zip(casos, novas):
        assert nv["instancia"] == lin.instancia, (a, nv["instancia"], lin.instancia)
        for campo in r21.CAMPOS_REG:
            assert str(nv[campo]) == str(lin[campo]), (a, lin.instancia, lin.seed, campo, nv[campo], lin[campo])
        assert abs(nv["custo"] - lin["custo"]) < 1e-9 and bool(nv["viavel"]) == bool(lin["viavel"]), (a, lin.instancia, lin.seed)
    print(f"REGRESSAO EXATA: OK ({len(casos)} execucoes dos Estudos 21 e 23 reproduzidas com as especificacoes do Estudo 26)", flush=True)


def modo_eval(workers):
    sp = specs26()
    tar = [(s, 1, c, n, SEMENTES_AVAL + sd) for c, n, _nc in instancias26() for s in sp.values() for sd in range(SEEDS26)]
    print(f"eval Estudo 26 (P21): {len(tar)} execucoes ({', '.join(sp)}) em {len(set((t[2], t[3]) for t in tar))} instancias", flush=True)
    with mp.Pool(workers) as pool:
        r23.rodar23(tar, RES / "etapa26_eval.csv", pool)
    print("eval Estudo 26 concluido", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--regressao", action="store_true")
    g.add_argument("--instancias", action="store_true")
    g.add_argument("--eval", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.regressao:
        return modo_regressao(a.workers)
    if a.instancias:
        return modo_instancias()
    return modo_eval(a.workers)


if __name__ == "__main__":
    main()
