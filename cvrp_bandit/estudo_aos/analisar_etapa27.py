#!/usr/bin/env python3
"""
analisar_etapa27.py
===================
Analise do Estudo 27 (o numero de rotas modera a vantagem dos pesos adaptativos? manipulacao da capacidade a leiaute e demandas fixos; SYN_CAP, 40 leiautes de SYN_REP x 3 tamanhos medios de rota alvo de
4, 10 e 25 clientes) conforme o PRE-REGISTRO (diario, Entrada 147). dif = gap(a) - gap(b); negativo = a melhor. Unidade = LEIAUTE (a media das seeds viaveis de cada variante; as tres variantes de um
leiaute sao pareadas); Wilcoxon bilateral pareado; IC95% bootstrap (10.000, semente 0); empate |dif| < 1e-9; MDE = 2,8 DP / raiz(n); equivalencia = IC90% dentro de +-0,10 pp. Bracos congelados: U e RP.
  Primarios (Holm m=2)   : H189 DiD de (RP - U) entre r = 4 (muitas rotas) e r = 25 (poucas), negativa ; H190 RP - U em r = 4, negativa
  Secundarios (Holm m=2) : H191 DiD de (RP - U) entre r = 4 e r = 10, negativa ; H192 DiD de (RP - U) entre r = 10 e r = 25, negativa
  Exploratorios          : RP - U em r = 10 e r = 25 (com equivalencia e MDE); o efeito por tamanho (100 e 200 clientes); a reducao relativa (RP - U)/gap de U por r; o gap da uniforme, o numero de rotas de
                           referencia, a razao de iteracoes RP/U e o uso de rota_pequena por r; a tendencia dentro do leiaute (Spearman de K com RP - U em cada leiaute).
Uso: python estudo_aos/analisar_etapa27.py [--ensaio]   (--ensaio fabrica dados para exercitar o codigo; nada e gravado em results/)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, com_holm  # noqa: E402
import analisar_etapa21 as a21  # noqa: E402

RES = a21.RES
COLS = a21.COLS + ["iteracoes"]
RS = (4, 10, 25)


def ler27():
    d = pd.read_csv(RES / "etapa27_eval.csv")
    d["braco"] = d.braco.str.replace("s27_", "", regex=False)
    return d[COLS]


def ler_k():
    m = pd.read_csv(RES.parent.parent / "instances" / "SYN_CAP" / "metadata.csv")
    return pd.Series(dict(zip(m.nome, m.rotas_ref)), dtype=float)


def chave(nome):
    m = re.fullmatch(r"X-synq(\d+)-n(\d+)-r(\d+)", nome)
    return f"{m.group(1)}-n{int(m.group(2)) - 1}", int(m.group(3))


def fabricar(rng):
    """ENSAIO: dados fabricados (a vantagem cresce com o numero de rotas) para exercitar o codigo; devolve (dados, K por instancia)."""
    out, kk = [], {}
    for n in (100, 200):
        for bb in range(20):
            base = 1.0 + 0.004 * n + rng.normal(0, 0.4)
            for r in RS:
                nome = f"X-synq{bb + 1 + (0 if n == 100 else 20):02d}-n{n + 1}-r{r:02d}"
                K = n / r
                kk[nome] = float(round(K))
                for a in ("U", "RP"):
                    for sd in (1000, 1001):
                        ef = -0.004 * K if a == "RP" else 0.0
                        out.append(dict(braco=a, conjunto="SYN_CAP", instancia=nome, n_customers=n, seed=sd, viavel=True, gap_pct=base * (1 + ef) + rng.normal(0, 0.25),
                                        uso_d="|".join(["1"] * 12), uso_r="|".join(["1"] * 9), iteracoes=1000 + 20 * K * (a == "RP")))
    return pd.DataFrame(out), pd.Series(kk)


def main(ensaio=False):
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    dados, kser = fabricar(np.random.default_rng(7)) if ensaio else (ler27(), ler_k())
    h = dados[dados.viavel].groupby(["instancia", "braco"]).agg(gap=("gap_pct", "mean"), it=("iteracoes", "mean")).reset_index()
    g = h.pivot(index="instancia", columns="braco", values="gap").dropna()
    it = h.pivot(index="instancia", columns="braco", values="it").reindex(g.index)
    meta = pd.DataFrame([dict(instancia=i, layout=chave(i)[0], r=chave(i)[1], n=int(i.split("-n")[1].split("-")[0]) - 1) for i in g.index]).set_index("instancia")
    meta["K"] = kser.reindex(meta.index)
    L = pd.DataFrame(index=sorted(meta.layout.unique()))
    L["n"] = meta.groupby("layout").n.first()
    for r in RS:
        s = meta[meta.r == r]
        idx = pd.Index(s.layout)
        L[f"dif{r}"] = pd.Series((g.loc[s.index, "RP"] - g.loc[s.index, "U"]).values, index=idx)
        L[f"gapU{r}"] = pd.Series(g.loc[s.index, "U"].values, index=idx)
        L[f"K{r}"] = pd.Series(s.K.values, index=idx)
        L[f"lr{r}"] = pd.Series(np.log(it.loc[s.index, "RP"] / it.loc[s.index, "U"]).values, index=idx)
    L = L.dropna()
    nL = len(L)
    print(f"=== {nL} leiautes completos (3 variantes cada); gap medio da uniforme e RP - U por tamanho medio de rota alvo ===")
    for r in RS:
        print(f"  r = {r:2d}: K medio {L[f'K{r}'].mean():5.1f}; gap U {L[f'gapU{r}'].mean():.3f}%; RP - U {L[f'dif{r}'].mean():+.3f} pp; reducao relativa media {(L[f'dif{r}'] / L[f'gapU{r}']).mean():+.3f}")
    L["zero"] = 0.0
    L["did_4_25"], L["did_4_10"], L["did_10_25"] = L.dif4 - L.dif25, L.dif4 - L.dif10, L.dif10 - L.dif25
    C = lambda cid, a, b, gr, sub=None, **kw: contraste(L if sub is None else L[sub], cid, a, b, rng, gr, **kw)  # noqa: E731
    prim = com_holm([C("H189 DiD(RP-U) r4-r25", "did_4_25", "zero", "primario"), C("H190 RP-U@r4", "dif4", "zero", "primario")], 2)
    sec = com_holm([C("H191 DiD(RP-U) r4-r10", "did_4_10", "zero", "secundario"), C("H192 DiD(RP-U) r10-r25", "did_10_25", "zero", "secundario")], 2)
    expl = [C("RP-U@r10", "dif10", "zero", "explor"), C("RP-U@r25", "dif25", "zero", "explor")]
    for nn in (100, 200):
        sub = L.n == nn
        if int(sub.sum()) >= 6:
            expl += [C(f"DiD(RP-U) r4-r25 @n{nn}", "did_4_25", "zero", "explor_n", sub, tamanho=nn), C(f"RP-U@r4 @n{nn}", "dif4", "zero", "explor_n", sub, tamanho=nn),
                     C(f"RP-U@r25 @n{nn}", "dif25", "zero", "explor_n", sub, tamanho=nn)]
    ex = pd.DataFrame(expl)
    ex["p_holm"], ex["signif_005"] = ex.p_bruto, ex.p_bruto < 0.05
    r_ = pd.concat([prim, sec, ex], ignore_index=True)
    print("\n=== primarios (Holm m=2), secundarios (Holm m=2) e exploratorios ===")
    print(r_.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    gg = lambda cid: r_[r_.id.str.startswith(cid + " ")].iloc[0]  # noqa: E731
    neg = lambda x: bool(x.p_holm < 0.05 and x.dif_pp < 0)  # noqa: E731
    H = {"H189 DiD de (RP - U) entre r = 4 e r = 25 significativamente negativa (primario, Holm m=2)": neg(gg("H189")),
         "H190 RP - U com r = 4 significativamente negativo (primario, Holm m=2)": neg(gg("H190")),
         "H191 DiD de (RP - U) entre r = 4 e r = 10 significativamente negativa (secundario, Holm m=2)": neg(gg("H191")),
         "H192 DiD de (RP - U) entre r = 10 e r = 25 significativamente negativa (secundario, Holm m=2)": neg(gg("H192"))}
    print("\n=== hipoteses (pre-registro, Entrada 147) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de 4 (esperado no pre-registro: 2,55)")
    # tendencia dentro do leiaute e resumo por r
    rhos = []
    for lay, lin in L.iterrows():
        k = [lin[f"K{r}"] for r in RS]
        d = [lin[f"dif{r}"] for r in RS]
        rhos.append(stats.spearmanr(k, d)[0] if len(set(k)) > 1 and len(set(d)) > 1 else np.nan)
    rhos = pd.Series(rhos).dropna()
    p_rho = stats.wilcoxon(rhos).pvalue if len(rhos) >= 6 and np.any(rhos != 0) else float("nan")
    print(f"\nSpearman de K com RP - U dentro de cada leiaute (3 pontos): media {rhos.mean():+.3f}, mediana {rhos.median():+.3f}, Wilcoxon contra zero p {p_rho:.4f} (n {len(rhos)})")
    resumo = []
    for r in RS:
        s = meta[meta.r == r].index
        uso = pd.DataFrame(dados[(dados.braco == "RP") & dados.instancia.isin(s)].uso_d.str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean()
        uso = 100 * uso / uso.sum()
        resumo.append(dict(r=r, K=float(L[f"K{r}"].mean()), gap_U=float(L[f"gapU{r}"].mean()), dif=float(L[f"dif{r}"].mean()), rel=float((L[f"dif{r}"] / L[f"gapU{r}"]).mean()),
                           razao_it=float(np.exp(L[f"lr{r}"].mean())), uso_rota_pequena=float(uso.iloc[11]) if len(uso) > 11 else float("nan")))
    resumo = pd.DataFrame(resumo)
    print("\n=== exploratorio: por tamanho medio de rota alvo ===\n" + resumo.round(3).to_string(index=False))
    if not ensaio:
        r_.to_csv(RES / "analise_etapa27.csv", index=False)
        resumo.to_csv(RES / "analise_etapa27_resumo.csv", index=False)
        pd.DataFrame([dict(rho_medio=float(rhos.mean()), rho_mediana=float(rhos.median()), p_wilcoxon=float(p_rho), n=len(rhos))]).to_csv(RES / "analise_etapa27_tendencia.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa27_hipoteses.csv", index=False)


if __name__ == "__main__":
    main(ensaio="--ensaio" in sys.argv)
