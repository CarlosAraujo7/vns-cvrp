#!/usr/bin/env python3
"""
analisar_etapa28.py
===================
Analise do Estudo 28 (o conjunto de 21 operadores no ALNS do CVRPTW: Solomon, 56 instancias de 100 clientes) conforme o PRE-REGISTRO (diario, Entrada 148). dif = gap(a) - gap(b); negativo = a melhor.
Unidade = instancia (media das seeds viaveis); Wilcoxon bilateral pareado; IC95% bootstrap (10.000, semente 0); empate |dif| < 1e-9; MDE = 2,8 DP / raiz(n); equivalencia = IC90% dentro de +-0,10 pp.
Bracos de P21 (seeds 1000-1004): U, RP (rho 0,1, segmento 100), S33 e S65 (misturas estaticas ajustadas em 33 e 65 avaliacoes), HG (HG-UCB do Estudo 16). Referencia de P7 (Estudo 16, mesmas instancias, seeds
1000-1004): U, RP, S33 (a mistura estatica de 33 avaliacoes do Estudo 16) e HG. Se S65 = S33 (a continuacao do ES nao achou melhora), o braco nao foi executado e a coluna e copiada.
  Primarios (Holm m=3)   : H193 RP - S33 em P21 ; H194 DiD de (RP - S33) entre P21 e P7 ; H195 RP - U em P21                                  [significativamente NEGATIVOS]
  Secundarios (Holm m=3) : H196 RP - S65 em P21 ; H197 HG - U em P21 ; H198 Spearman de RP - U com o numero de rotas K da solucao de referencia (classes R e RC)   [significativamente NEGATIVOS]
  Exploratorios          : S33 - U, S65 - U, HG - RP, S65 - S33; DiD de (RP - U); por classe de Solomon; reducao relativa; razao de iteracoes; uso de operadores; equivalencia; MDE.
Uso: python estudo_aos/analisar_etapa28.py [--ensaio]   (--ensaio fabrica dados para exercitar o codigo; nada e gravado em results/)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo, com_holm  # noqa: E402
import analisar_etapa21 as a21  # noqa: E402

RES = a21.RES
ROOT = RES.parent.parent
COLS = a21.COLS + ["iteracoes"]


def ler28():
    d = pd.read_csv(RES / "etapa28_eval.csv")
    d["braco"] = d.braco.str.replace("s28_", "", regex=False) + "@P21"
    return d[COLS]


def referencia_p7():
    e = pd.read_csv(RES / "etapa16_eval_a.csv")
    e = e[e.seed < 1005]                                                            # as mesmas 5 seeds do Estudo 28
    return pd.concat([e[e.braco == b][COLS].assign(braco=n) for b, n in (("s16a_U", "U@P7"), ("s16a_RP", "RP@P7"), ("s16a_hgucb", "HG@P7"), ("s16a_static", "S33@P7"))], ignore_index=True)


def ler_k(nomes):
    return pd.Series({n: float(sum(1 for l in (ROOT / "instances" / "SOLOMON" / "bks" / f"{n}.sol").read_text().splitlines() if l.startswith("Route"))) for n in nomes})


def fabricar(rng):
    """ENSAIO: dados fabricados (efeitos conhecidos + ruido) para exercitar o codigo; devolve (dados, K por instancia)."""
    import run_aos_etapa14 as r14
    nomes = sorted(r14.nomes("SOLOMON"))
    K = {n: {"C1": 10, "C2": 3, "R1": 13, "R2": 4, "RC1": 13, "RC2": 5}[re.match(r"([A-Z]+\d)", n).group(1)] for n in nomes}
    efeito = {"U@P21": 0.0, "RP@P21": -0.25, "S33@P21": -0.20, "S65@P21": -0.22, "HG@P21": -0.10, "U@P7": 0.0, "RP@P7": -0.15, "S33@P7": -0.45, "HG@P7": -0.30}
    out = []
    for n in nomes:
        base = 1.0 + (3.0 if "2" in re.match(r"[A-Z]+(\d)", n).group(1) else 0.0) + rng.normal(0, 0.4)
        for b, ef in efeito.items():
            for sd in range(1000, 1005):
                out.append(dict(braco=b, conjunto="SOLOMON", instancia=n, n_customers=100, seed=sd, viavel=True, gap_pct=base * (1 + ef) + rng.normal(0, 0.3), uso_d="|".join(["1"] * 12), uso_r="|".join(["1"] * 9),
                                iteracoes=1000))
    return pd.DataFrame(out), pd.Series(K, dtype=float)


def parcial_nada():
    return None


def main(ensaio=False):
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    dados, kser = fabricar(np.random.default_rng(7)) if ensaio else (pd.concat([ler28(), referencia_p7()], ignore_index=True), None)
    if "S65@P21" not in set(dados.braco):
        print("[P21] S65 ausente: a continuacao do ES nao achou melhora (S65 = S33); coluna copiada")
        dados = pd.concat([dados, dados[dados.braco == "S33@P21"].assign(braco="S65@P21")], ignore_index=True)
    t = pivo(dados).copy()
    if kser is None:
        kser = ler_k(t.index)
    K = kser.reindex(t.index)
    print(f"=== {len(t)} instancias; gap medio (%) por braco ===\n{t.mean().round(3).sort_values().to_string()}")
    t["zero"] = 0.0
    t["did_S33"] = (t["RP@P21"] - t["S33@P21"]) - (t["RP@P7"] - t["S33@P7"])
    t["did_U"] = (t["RP@P21"] - t["U@P21"]) - (t["RP@P7"] - t["U@P7"])
    classe = pd.Series(t.index.str.extract(r"^([A-Z]+\d)", expand=False).values, index=t.index)
    rc = classe.str.startswith("R")                                                  # classes R1, R2, RC1 e RC2 (as C ja estao na referencia: nada a melhorar)
    C = lambda cid, a, b, g, sub=None, **kw: contraste(t if sub is None else t[sub.values], cid, a, b, rng, g, **kw)  # noqa: E731
    dif0 = (t["RP@P21"] - t["U@P21"])[rc]
    rho, p = stats.spearmanr(K[rc], dif0)
    lc = dict(grupo="secundario", id="H198 Spearman(RP-U, K) [rho]", a="RP-U", b="K", n_inst=int(rc.sum()), dif_pp=float(rho), ic_lo=float("nan"), ic_hi=float("nan"), vit_a=0, der_a=0, emp=0, p_bruto=float(p),
              equiv_010=False, mde_pp=float("nan"))
    prim = com_holm([C("H193 RP-S33@P21", "RP@P21", "S33@P21", "primario"), C("H194 DiD(RP-S33) P21-P7", "did_S33", "zero", "primario"), C("H195 RP-U@P21", "RP@P21", "U@P21", "primario")], 3)
    sec = com_holm([C("H196 RP-S65@P21", "RP@P21", "S65@P21", "secundario"), C("H197 HG-U@P21", "HG@P21", "U@P21", "secundario"), lc], 3)
    expl = [C("S33-U@P21", "S33@P21", "U@P21", "explor"), C("S65-U@P21", "S65@P21", "U@P21", "explor"), C("HG-RP@P21", "HG@P21", "RP@P21", "explor"), C("S65-S33@P21", "S65@P21", "S33@P21", "explor"),
            C("DiD(RP-U) P21-P7", "did_U", "zero", "explor_did"), C("RP-U@P7", "RP@P7", "U@P7", "explor"), C("RP-S33@P7", "RP@P7", "S33@P7", "explor"), C("S33-U@P7", "S33@P7", "U@P7", "explor"),
            C("HG-U@P7", "HG@P7", "U@P7", "explor")]
    for cl in sorted(set(classe)):
        sub = classe == cl
        if int(sub.sum()) >= 6 and cl not in ("C1", "C2"):
            expl += [C(f"RP-U@P21 {cl}", "RP@P21", "U@P21", "explor_classe", sub, classe=cl), C(f"RP-S33@P21 {cl}", "RP@P21", "S33@P21", "explor_classe", sub, classe=cl)]
    ex = pd.DataFrame(expl)
    ex["p_holm"], ex["signif_005"] = ex.p_bruto, ex.p_bruto < 0.05
    r = pd.concat([prim, sec, ex], ignore_index=True)
    print("\n=== primarios (Holm m=3), secundarios (Holm m=3) e exploratorios ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    g = lambda cid: r[r.id.str.startswith(cid + " ")].iloc[0]  # noqa: E731
    neg = lambda x: bool(x.p_holm < 0.05 and x.dif_pp < 0)  # noqa: E731
    H = {"H193 RP - S33 em P21 do CVRPTW significativamente negativo (primario, Holm m=3)": neg(g("H193")),
         "H194 DiD de (RP - S33) entre P21 e P7 significativamente negativa (primario, Holm m=3)": neg(g("H194")),
         "H195 RP - U em P21 do CVRPTW significativamente negativo (primario, Holm m=3)": neg(g("H195")),
         "H196 RP - S65 em P21 do CVRPTW significativamente negativo (secundario, Holm m=3)": neg(g("H196")),
         "H197 HG - U em P21 do CVRPTW significativamente negativo (secundario, Holm m=3)": neg(g("H197")),
         "H198 correlacao de Spearman de RP - U com o numero de rotas K significativamente negativa (secundario, Holm m=3)": neg(g("H198"))}
    print("\n=== hipoteses (pre-registro, Entrada 148) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de 6 (esperado no pre-registro: 2,95)")
    # por classe: gap da uniforme, RP - U, razao de iteracoes
    linhas = []
    it = dados[dados.braco.isin(["U@P21", "RP@P21"])].groupby(["instancia", "braco"]).iteracoes.mean().unstack()
    for cl in sorted(set(classe)):
        sub = (classe == cl).values
        d = (t["RP@P21"] - t["U@P21"])[sub]
        linhas.append(dict(classe=cl, n_inst=int(sub.sum()), K=float(K[sub].mean()), gap_U=float(t["U@P21"][sub].mean()), RP_U=float(d.mean()), S33_U=float((t["S33@P21"] - t["U@P21"])[sub].mean()),
                           HG_U=float((t["HG@P21"] - t["U@P21"])[sub].mean()), RP_S33=float((t["RP@P21"] - t["S33@P21"])[sub].mean()), razao_it=float((it["RP@P21"] / it["U@P21"]).reindex(t.index)[sub].mean())))
    cls = pd.DataFrame(linhas)
    print("\n=== exploratorio: por classe de Solomon (pp; medias) ===\n" + cls.round(3).to_string(index=False))
    uso = []
    for b in ("U@P21", "RP@P21", "S33@P21", "S65@P21", "HG@P21"):
        h = dados[dados.braco == b]
        m = {c: pd.DataFrame(h[c].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean() for c in ("uso_d", "uso_r")}
        ud, ur = (m["uso_d"] / m["uso_d"].sum()).round(3).tolist(), (m["uso_r"] / m["uso_r"].sum()).round(3).tolist()
        uso.append(dict(braco=b.split("@")[0], iteracoes=float(h.iteracoes.mean()), uso_d="|".join(map(str, ud)), uso_r="|".join(map(str, ur))))
    print("\n=== exploratorio: iteracoes e uso ===\n" + pd.DataFrame(uso).to_string(index=False))
    if not ensaio:
        r.to_csv(RES / "analise_etapa28.csv", index=False)
        cls.to_csv(RES / "analise_etapa28_classes.csv", index=False)
        pd.DataFrame(uso).to_csv(RES / "analise_etapa28_uso.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa28_hipoteses.csv", index=False)


if __name__ == "__main__":
    main(ensaio="--ensaio" in sys.argv)
