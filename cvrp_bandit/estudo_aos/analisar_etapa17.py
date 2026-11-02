#!/usr/bin/env python3
"""
analisar_etapa17.py
===================
Analise do Estudo 17 (selecao adaptativa do conjunto de operadores da busca local no ILS do PyVRP; CVRPLIB 74 + SYN_VAL 30, dois
orcamentos de tempo) conforme o PRE-REGISTRO (diario, Entrada 124). dif = gap(a) - gap(b); negativo = a melhor. Unidade = instancia
(media das seeds viaveis; as 104 instancias juntas), Wilcoxon bilateral pareado, IC95% bootstrap (10.000, semente 0), empate
|dif| < 1e-9. Familias (por NIVEL de orcamento, exceto a escada):
  Primario (sem correcao)  : RP - U no nivel 1
  Secundarios (Holm m=15)  : {ucb1, pm, hgucb, ts, exp3, eff, static} - U nos dois niveis e RP - U no nivel 2
  Escada de conjuntos      : P(j+1) - P(j), j = 3..6, nos dois niveis (Holm m=8)
  Terciarios (Holm m=4)    : static - P6 e static - P7 em cada nivel
  Padrao                   : U - P6 em cada nivel (Holm m=2)
  Melhor desenho nao adaptativo: {RP, ucb1, pm, hgucb, ts, exp3, eff} - {P6, P7, static} em cada nivel (Holm m=21 por nivel)
  Exploratorios (rotular)  : por conjunto de instancias (CVRPLIB, SYN_VAL) e por familia (A, B, X, CMT); equivalencia (IC90% em +-0,10 pp);
                             MDE; uso dos conjuntos por politica.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
N_BOOT = 10000
ADAPT = ["RP", "ucb1", "pm", "hgucb", "ts", "exp3", "eff"]
SEC = ["ucb1", "pm", "hgucb", "ts", "exp3", "eff", "static"]
DESENHOS = ["P6", "P7", "static"]


def holm(pvals, m):
    ordem = np.argsort(pvals)
    aj = np.empty(len(pvals))
    corrente = 0.0
    for rank, idx in enumerate(ordem):
        corrente = max(corrente, min(1.0, (m - rank) * pvals[idx]))
        aj[idx] = corrente
    return aj


def contraste(piv, cid, a, b, rng, grupo, **extra):
    dif = piv[a].values - piv[b].values
    dif = np.where(np.abs(dif) < 1e-9, 0.0, dif)
    idx = rng.integers(0, len(dif), size=(N_BOOT, len(dif)))
    med = dif[idx].mean(axis=1)
    lo, hi = np.percentile(med, [2.5, 97.5])
    lo90, hi90 = np.percentile(med, [5, 95])
    p = stats.wilcoxon(dif).pvalue if np.any(dif != 0) else 1.0
    return dict(grupo=grupo, id=cid, a=a, b=b, n_inst=len(dif), dif_pp=dif.mean(), ic_lo=lo, ic_hi=hi,
                vit_a=int((dif < 0).sum()), der_a=int((dif > 0).sum()), emp=int((dif == 0).sum()), p_bruto=p,
                equiv_010=bool(lo90 > -0.1 and hi90 < 0.1), mde_pp=2.8 * dif.std(ddof=1) / np.sqrt(len(dif)), **extra)


def pivo(d):
    return d[d.viavel].groupby(["braco", "instancia"]).gap_pct.mean().unstack(0).dropna()


def com_holm(linhas, m):
    r = pd.DataFrame(linhas)
    r["p_holm"] = holm(r.p_bruto.values, m)
    r["signif_005"] = r.p_holm < 0.05
    return r


def familia(nome):
    """A, B, X, CMT ou SYN_VAL (as sinteticas se chamam X-synv##-n###)"""
    return "SYN_VAL" if nome.startswith("X-syn") else ("CMT" if nome.startswith("CMT") else nome[0])


def main():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    e = pd.read_csv(RES / "etapa17_eval.csv")
    e["braco"] = e.braco.str.replace("s17_", "", regex=False)
    niveis = sorted(e.nivel.unique())
    assert len(niveis) == 2, niveis
    print(f"avaliacao: {len(e)} execucoes, inviaveis {int((~e.viavel).sum())}; niveis {niveis}")
    linhas, expl, pivs = [], [], {}
    for k, nv in enumerate(niveis, 1):
        t = pivo(e[e.nivel == nv])
        pivs[k] = t
        print(f"\n=== nivel {k} ({nv} s/cliente): {len(t)} instancias; gap medio (%) ===")
        print(t.mean().round(3).sort_values().to_string())
    prim = pd.DataFrame([contraste(pivs[1], "RP-U@1", "RP", "U", rng, "primario", nivel=1)])
    prim["p_holm"], prim["signif_005"] = prim.p_bruto, prim.p_bruto < 0.05
    sec = [contraste(pivs[k], f"{a}-U@{k}", a, "U", rng, "secundario", nivel=k) for k in (1, 2) for a in SEC]
    sec.append(contraste(pivs[2], "RP-U@2", "RP", "U", rng, "secundario", nivel=2))
    sec = com_holm(sec, 15)
    esc = com_holm([contraste(pivs[k], f"P{j + 1}-P{j}@{k}", f"P{j + 1}", f"P{j}", rng, "escada", nivel=k)
                    for k in (1, 2) for j in range(3, 7)], 8)
    ter = com_holm([contraste(pivs[k], f"static-{b}@{k}", "static", b, rng, "terciario", nivel=k)
                    for k in (1, 2) for b in ("P6", "P7")], 4)
    pad = com_holm([contraste(pivs[k], f"U-P6@{k}", "U", "P6", rng, "padrao", nivel=k) for k in (1, 2)], 2)
    mel = pd.concat([com_holm([contraste(pivs[k], f"{a}-{b}@{k}", a, b, rng, "melhor_nao_adaptativo", nivel=k)
                               for a in ADAPT for b in DESENHOS], 21) for k in (1, 2)], ignore_index=True)
    r = pd.concat([prim, sec, esc, ter, pad, mel], ignore_index=True)
    print("\n=== primario, secundarios (m=15), escada (m=8), terciarios (m=4), padrao (m=2), melhor nao adaptativo (m=21 por nivel) ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa17_a.csv", index=False)

    # exploratorio: por conjunto de instancias (CVRPLIB x SYN_VAL) e por familia, sem correcao
    for k in (1, 2):
        t = pivs[k]
        fam = pd.Series({n: familia(n) for n in t.index})
        for rot, mask in (("CVRPLIB", fam != "SYN_VAL"), ("SYN_VAL", fam == "SYN_VAL"), ("A", fam == "A"), ("B", fam == "B"),
                          ("X", fam == "X"), ("CMT", fam == "CMT")):
            s = t[mask.values]
            if len(s) < 3:
                continue
            for a, b in (("RP", "U"), ("hgucb", "U"), ("static", "U"), ("U", "P6"), ("hgucb", "P6"), ("hgucb", "P7"), ("P7", "P6")):
                expl.append(contraste(s, f"{a}-{b}@{k}:{rot}", a, b, rng, "explor_conjunto", nivel=k, conjunto=rot))
    ex = pd.DataFrame(expl)
    print("\n=== exploratorio: por conjunto de instancias (sem correcao; rotular) ===")
    print(ex[["id", "n_inst", "dif_pp", "ic_lo", "ic_hi", "vit_a", "der_a", "p_bruto"]].round(4).to_string(index=False))
    ex.to_csv(RES / "analise_etapa17_conjunto.csv", index=False)

    print("\nuso medio dos conjuntos P1..P7 por politica (fracao das iteracoes) e fracao do TEMPO:")
    for a in ["U", "RP", "ucb1", "pm", "hgucb", "ts", "exp3", "eff", "static"]:
        g = e[(e.braco == a) & e.viavel]
        u = pd.DataFrame(g.uso.str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean()
        tp = pd.DataFrame(g.tempo_pool.str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean()
        print(f"  {a:7s} iteracoes {(u / u.sum()).round(3).tolist()}   tempo {(tp / tp.sum()).round(3).tolist()}")
    print("iteracoes medias por braco (nivel 1):",
          e[e.nivel == niveis[0]].groupby("braco").iteracoes.mean().round(0).to_dict())


if __name__ == "__main__":
    main()
