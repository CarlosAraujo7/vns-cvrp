#!/usr/bin/env python3
"""
analisar_folga_tamanho.py
=========================
Analise EXPLORATORIA (declarada depois de ver a correlacao de folga inflada pelo ruido compartilhado; diario, Entrada 145): a vantagem dos pesos adaptativos (RP - U) por TAMANHO e por classe de
instancia, em tres conjuntos de instancias: (i) avaliacao do Estudo 21 (X e SYN_VAL, 51 instancias, 1.500 n^2, pools P7 e P21); (ii) VALIDACAO do Estudo 21 (SYN_TUNE, 24 instancias de n = 50, 100
e 150, seeds 3003-3005, pool P21; as instancias foram usadas para ajustar as misturas, mas o braco RP nao e ajustado nelas); (iii) A, B e CMT do Estudo 22 (regime cls, 1.500 n^2, P21; 53 instancias de
30 a 120 clientes). Para cada classe: n de instancias, gap medio da uniforme, RP - U em pp, a reducao relativa (RP - U)/gap da uniforme e o p de Wilcoxon (descritivo, sem correcao).
Uso: python estudo_aos/analisar_folga_tamanho.py     (grava results/aos_study/analise_folga_tamanho.csv)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
sys.path.insert(0, str(AQUI))
import analisar_etapa21 as a21  # noqa: E402


def medias(d, a, b):
    h = d[d.viavel]
    p = h.groupby(["braco", "instancia"]).agg(gap=("gap_pct", "mean"), n=("n_customers", "first")).reset_index()
    pu = p[p.braco == a].set_index("instancia")
    pr = p[p.braco == b].set_index("instancia")
    j = pu.join(pr, lsuffix="_u", rsuffix="_rp", how="inner")
    return pd.DataFrame(dict(n=j.n_u, gap_u=j.gap_u, gap_rp=j.gap_rp))


def resumo(rotulo, df, saida, classe_de):
    df = df.assign(classe=[classe_de(i, n) for i, n in zip(df.index, df.n)], dif=df.gap_rp - df.gap_u)
    for cl, g in df.groupby("classe", sort=False):
        d = g.dif.values
        p = stats.wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else float("nan")
        saida.append(dict(conjunto=rotulo, classe=cl, n_inst=len(g), n_min=int(g.n.min()), n_max=int(g.n.max()), gap_u=float(g.gap_u.mean()), dif_pp=float(d.mean()),
                          rel=float(d.mean() / g.gap_u.mean()), p_wilcoxon=float(p)))
    d = df.dif.values
    saida.append(dict(conjunto=rotulo, classe="TODAS", n_inst=len(df), n_min=int(df.n.min()), n_max=int(df.n.max()), gap_u=float(df.gap_u.mean()), dif_pp=float(d.mean()),
                      rel=float(d.mean() / df.gap_u.mean()), p_wilcoxon=float(stats.wilcoxon(d).pvalue)))
    rho, p = stats.spearmanr(df.n, df.dif)
    saida.append(dict(conjunto=rotulo, classe="Spearman(n, RP-U)", n_inst=len(df), n_min=int(df.n.min()), n_max=int(df.n.max()), gap_u=float("nan"), dif_pp=float(rho), rel=float("nan"), p_wilcoxon=float(p)))


def main():
    pd.set_option("display.width", 220)
    saida = []
    ref, novo = a21.referencias(), a21.ler_novos("P21")
    tam_x = lambda i, n: "X (100-194)" if i.startswith("X-n") else f"SYN_VAL n={n}"  # noqa: E731
    resumo("avaliacao P7 (Estudo 9)", medias(ref, "U@P7", "RP@P7"), saida, tam_x)
    resumo("avaliacao P21 (Estudo 21)", medias(novo, "U@P21", "RP@P21"), saida, tam_x)
    v = pd.read_csv(RES / "etapa21_validacao_P21.csv")
    print("bracos da validacao de P21:", sorted(v.braco.unique()))
    bu = [b for b in v.braco.unique() if b.endswith("_U")][0]
    br = [b for b in v.braco.unique() if b.endswith("_RP")][0]
    resumo("validacao P21 (SYN_TUNE, seeds 3003-3005)", medias(v, bu, br), saida, lambda i, n: f"SYN_TUNE n={n}")
    c = pd.read_csv(RES / "etapa22_eval_cls.csv")
    cu = [b for b in c.braco.unique() if b.endswith("_U")][0]
    cr = [b for b in c.braco.unique() if b.endswith("_RP")][0]
    conj = c.groupby("instancia").conjunto.first()
    dfc = medias(c, cu, cr)
    resumo("A, B e CMT (Estudo 22, cls, P21)", dfc, saida, lambda i, n: f"{conj[i]}")
    resumo("A, B e CMT por tamanho (Estudo 22, cls, P21)", dfc, saida, lambda i, n: "n<=50" if n <= 50 else ("51-80" if n <= 80 else ">80"))
    r = pd.DataFrame(saida)
    print(r.round(3).to_string(index=False))
    r.to_csv(RES / "analise_folga_tamanho.csv", index=False)


if __name__ == "__main__":
    main()
