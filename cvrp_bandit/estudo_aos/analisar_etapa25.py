#!/usr/bin/env python3
"""
analisar_etapa25.py
===================
Analise do Estudo 25 (deslocamentos de implantacao em instancias com folga: X e SYN_VAL) conforme o PRE-REGISTRO (diario, Entrada 144). dif = gap(a) - gap(b); negativo = a melhor. Unidade = instancia (media
das seeds viaveis; 51 instancias); Wilcoxon bilateral pareado; IC95% bootstrap (10.000, semente 0); empate |dif| < 1e-9; MDE = 2,8 DP / raiz(n); equivalencia = IC90% em +-0,10 pp.
Regimes: acc (record-to-record travel, 1.500 n^2) e short (375 n^2, recozimento); referencia `ref` = os bracos U, RP e S65 de P21 do Estudo 21 (mesmas instancias, 1.500 n^2, recozimento).
S* = S129 se existe, senao S65 (aqui sempre S65, que e o braco avaliado).
  Primarios (Holm m=2)   : H177 RP - U @acc ; H178 RP - U @short                                  [significativamente NEGATIVOS]
  Secundarios (Holm m=2) : H179 RP - S65 @acc ; H180 RP - S65 @short                               [significativamente NEGATIVOS]
  Exploratorios          : S65 - U em cada regime; DiD de (RP - U) e de (RP - S65) contra a referencia; correlacao de RP - U com o gap da uniforme em cada regime (folga); equivalencia; uso de operadores e iteracoes.
Uso: python estudo_aos/analisar_etapa25.py [--ensaio]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo, com_holm  # noqa: E402
import analisar_etapa21 as a21  # noqa: E402

RES = a21.RES
COLS = a21.COLS + ["iteracoes"]
REGS = ("acc", "short")


def ler25(reg):
    d = pd.read_csv(RES / f"etapa25_eval_{reg}.csv")
    d["braco"] = d.braco.str.replace(f"s25_{reg}_", "", regex=False) + f"@{reg}"
    return d[COLS]


def referencia():
    e = pd.read_csv(RES / "etapa21_eval_P21.csv")
    return pd.concat([e[e.braco == f"s21_P21_{a}"][COLS].assign(braco=f"{a}@ref") for a in ("U", "RP", "S65")], ignore_index=True)


def fabricar(rng):
    ref = a21.referencias()
    base = ref[ref.braco == "S33@P7"].assign(iteracoes=1000)
    out = []
    for reg, ef in (("ref", {"U": 0.2, "RP": 0.0, "S65": 0.22}), ("acc", {"U": 0.25, "RP": 0.02, "S65": 0.2}), ("short", {"U": 0.3, "RP": 0.05, "S65": 0.28})):
        for a, e in ef.items():
            d = base.assign(braco=f"{a}@{reg}", gap_pct=base.gap_pct + e + rng.normal(0, 0.3, len(base)))
            d["uso_d"], d["uso_r"] = "|".join(["1"] * 12), "|".join(["1"] * 9)
            out.append(d)
    return pd.concat(out, ignore_index=True)


def main(ensaio=False):
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    dados = fabricar(np.random.default_rng(7)) if ensaio else pd.concat([referencia()] + [ler25(r) for r in REGS], ignore_index=True)
    t = pivo(dados).copy()
    print(f"=== {len(t)} instancias; gap medio (%) ===\n{t.mean().round(3).sort_values().to_string()}")
    for r_ in REGS:
        t[f"DiDU@{r_}"] = (t[f"RP@{r_}"] - t[f"U@{r_}"]) - (t["RP@ref"] - t["U@ref"])
        t[f"DiDS@{r_}"] = (t[f"RP@{r_}"] - t[f"S65@{r_}"]) - (t["RP@ref"] - t["S65@ref"])
    t["zero"] = 0.0
    C = lambda cid, a, b, g: contraste(t, cid, a, b, rng, g)  # noqa: E731
    prim = com_holm([C("H177 RP-U@acc", "RP@acc", "U@acc", "primario"), C("H178 RP-U@short", "RP@short", "U@short", "primario")], 2)
    sec = com_holm([C("H179 RP-S65@acc", "RP@acc", "S65@acc", "secundario"), C("H180 RP-S65@short", "RP@short", "S65@short", "secundario")], 2)
    expl = [C(f"RP-U@ref", "RP@ref", "U@ref", "explor"), C("RP-S65@ref", "RP@ref", "S65@ref", "explor")]
    for r_ in REGS:
        expl += [C(f"S65-U@{r_}", f"S65@{r_}", f"U@{r_}", "explor"), C(f"DiD(RP-U) {r_}-ref", f"DiDU@{r_}", "zero", "explor_did"), C(f"DiD(RP-S65) {r_}-ref", f"DiDS@{r_}", "zero", "explor_did")]
    ex = pd.DataFrame(expl)
    ex["p_holm"], ex["signif_005"] = ex.p_bruto, ex.p_bruto < 0.05
    r = pd.concat([prim, sec, ex], ignore_index=True)
    print("\n=== primarios (Holm m=2), secundarios (Holm m=2) e exploratorios ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    g = lambda cid: r[r.id.str.startswith(cid + " ")].iloc[0]  # noqa: E731
    neg = lambda x: bool(x.p_holm < 0.05 and x.dif_pp < 0)  # noqa: E731
    H = {"H177 RP - U em record-to-record travel significativamente negativo (primario, Holm m=2)": neg(g("H177")),
         "H178 RP - U com orcamento 0,25x significativamente negativo (primario, Holm m=2)": neg(g("H178")),
         "H179 RP - S65 em record-to-record travel significativamente negativo (secundario, Holm m=2)": neg(g("H179")),
         "H180 RP - S65 com orcamento 0,25x significativamente negativo (secundario, Holm m=2)": neg(g("H180"))}
    print("\n=== hipoteses (pre-registro, Entrada 144) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de 4 (esperado no pre-registro: 2,25)")
    folga = []
    from analisar_folga_metades import metades
    for r_ in ("ref",) + REGS:
        dif, gu = t[f"RP@{r_}"] - t[f"U@{r_}"], t[f"U@{r_}"]
        rho_, p_ = stats.spearmanr(gu, dif)
        mu = dados[(dados.braco == f"U@{r_}") & dados.viavel].pivot_table(index="instancia", columns="seed", values="gap_pct", aggfunc="mean")
        mr = dados[(dados.braco == f"RP@{r_}") & dados.viavel].pivot_table(index="instancia", columns="seed", values="gap_pct", aggfunc="mean")
        rho_m, lo_m, hi_m, p_m, _n = metades(mu, mr, np.random.default_rng(0), reps=200)       # metades das seeds: a folga de uma metade, a vantagem da outra (Entrada 145)
        rho_i, p_i = stats.spearmanr(t[f"S65@{r_}"], dif)                                      # folga medida por um braco que nao compartilha execucoes com U nem com RP
        folga.append(dict(regime=r_, rho_gap=float(rho_), p_gap=float(p_), gap_U=float(gu.mean()), dif=float(dif.mean()), rho_metades=rho_m, rho_metades_lo=lo_m, rho_metades_hi=hi_m,
                          rho_indep=float(rho_i), p_indep=float(p_i), rel_mediana=float((dif / gu).median())))
    folga = pd.DataFrame(folga)
    print("\n=== exploratorio: correlacao de RP - U com o gap da uniforme (folga) ===")
    print(folga.round(4).to_string(index=False))
    rows = []
    for r_ in ("ref",) + REGS:
        for a in ("RP", "S65"):
            h = dados[dados.braco == f"{a}@{r_}"]
            m = {c: pd.DataFrame(h[c].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean() for c in ("uso_d", "uso_r")}
            ud, ur = (m["uso_d"] / m["uso_d"].sum()).round(3).tolist(), (m["uso_r"] / m["uso_r"].sum()).round(3).tolist()
            rows.append(dict(regime=r_, braco=a, iteracoes=float(h.iteracoes.mean()) if h.iteracoes.notna().any() else float("nan"), uso_d="|".join(map(str, ud)), uso_r="|".join(map(str, ur))))
    if not ensaio:
        r.to_csv(RES / "analise_etapa25.csv", index=False)
        folga.to_csv(RES / "analise_etapa25_folga.csv", index=False)
        pd.DataFrame(rows).to_csv(RES / "analise_etapa25_uso.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa25_hipoteses.csv", index=False)


if __name__ == "__main__":
    main(ensaio="--ensaio" in sys.argv)
