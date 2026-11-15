#!/usr/bin/env python3
"""
analisar_etapa22.py
===================
Analise do Estudo 22 (deslocamentos de implantacao: classe de instancias, orcamento, regra de aceitacao) conforme o PRE-REGISTRO (diario, Entrada 138).
dif = gap(a) - gap(b); negativo = a melhor. Unidade = instancia (media das seeds viaveis); Wilcoxon bilateral pareado; IC95% bootstrap (10.000, semente 0); empate |dif| < 1e-9;
MDE = 2,8 DP / raiz(n); equivalencia = IC90% dentro de +-0,10 pp. Instancias: ABC = Augerat A (26), B (23) e CMT (4) = 53; X = 21 (seeds 1000-1009 nos Estudos 9 e 12).
Regimes (sufixo @): cls (orcamento 1.500 n^2, recozimento), short (375 n^2), longabc (30.000 n^2, 5 seeds), longx (6.000 n^2, X, 3 seeds), acc (record-to-record travel);
refx = os bracos U, RP e S33 em X dos Estudos 9 e 12 (1.500 n^2, recozimento, 10 seeds). DiD = [(RP - S33) no regime] - [(RP - S33) na referencia], por instancia.

  Primarios (Holm m=5)    : H130 RP - S33 @cls; H131 DiD de (RP - S33): longabc contra cls; H132 DiD: longx contra refx (X); H133 DiD: short contra cls; H134 DiD: acc contra cls
  Secundarios (Holm m=4)  : H135 S33r - S33 @acc (reajustar sob a nova regra ajuda); H136 S33 - U @cls; H137 RP - U @cls; H138 HG - U @cls
  Exploratorios (rotular) : cada braco contra U em cada regime; RP - S33 em cada regime e por conjunto (A, B, CMT); RP - S33r @acc; RPt, S65 - S33 @cls; efeito do regime sobre cada
                            braco; correlacao de RP - S33 com o tamanho; uso de operadores; validacao nas instancias de ajuste.
Uso: python estudo_aos/analisar_etapa22.py [--ensaio]   (--ensaio fabrica dados para exercitar o codigo; nada e gravado em results/)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo, com_holm  # noqa: E402

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
INST = AQUI.parent / "instances"
COLS = ["braco", "conjunto", "instancia", "n_customers", "seed", "viavel", "gap_pct", "uso_d", "uso_r"]
REG = {"cls": ("U", "RP", "RPt", "S33", "S65", "HG"), "short": ("U", "RP", "S33"), "longabc": ("U", "RP", "S33"), "longx": ("U", "RP", "S33"),
       "acc": ("U", "RP", "S33", "S33r")}


def ler(reg):
    d = pd.read_csv(RES / f"etapa22_eval_{reg}.csv")
    d["braco"] = d.braco.str.replace(f"s22_{reg}_", "", regex=False) + f"@{reg}"
    return d[COLS]


def referencia_x():
    e9, e12 = pd.read_csv(RES / "etapa9_eval.csv"), pd.read_csv(RES / "etapa12_eval.csv")
    sel = lambda d, b, novo: d[(d.braco == b) & (d.conjunto == "X")][COLS].assign(braco=novo)  # noqa: E731
    return pd.concat([sel(e9, "s9_U", "U@refx"), sel(e9, "s9_RP", "RP@refx"), sel(e12, "s12_static", "S33@refx")], ignore_index=True)


def fabricar(rng):
    """ENSAIO: dados fabricados (efeitos conhecidos + ruido) para exercitar o codigo."""
    nomes = {c: sorted(p.stem for p in (INST / c).glob("*.vrp")) for c in ("A", "B", "CMT", "X")}
    efeito = {"U": 0.30, "RP": 0.02, "S33": 0.0, "RPt": 0.03, "S65": -0.03, "HG": 0.15, "S33r": -0.05}
    out = []
    for reg, arms in REG.items():
        conjs = ("X",) if reg == "longx" else ("A", "B", "CMT")
        for c in conjs:
            for a in arms:
                for n in nomes[c]:
                    for sd in range(3):
                        out.append(dict(braco=f"{a}@{reg}", conjunto=c, instancia=n, n_customers=50 + len(n), seed=1000 + sd, viavel=True,
                                        gap_pct=1.0 + efeito[a] + rng.normal(0, 0.25) + (0.1 if reg == "acc" and a == "S33" else 0.0), uso_d="1|1|1|1", uso_r="1|1|1"))
    for a, e in (("U", 0.3), ("RP", 0.02), ("S33", 0.0)):
        for n in nomes["X"]:
            for sd in range(3):
                out.append(dict(braco=f"{a}@refx", conjunto="X", instancia=n, n_customers=100, seed=1000 + sd, viavel=True, gap_pct=1.3 + e + rng.normal(0, 0.25),
                                uso_d="1|1|1|1", uso_r="1|1|1"))
    return pd.DataFrame(out)


def main(ensaio=False):
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    dados = fabricar(np.random.default_rng(7)) if ensaio else pd.concat([ler(r) for r in REG] + [referencia_x()], ignore_index=True)
    abc = dados[dados.conjunto.isin(["A", "B", "CMT"])]
    tx = pivo(dados[dados.conjunto == "X"])
    t = pivo(abc[~abc.braco.str.endswith("@longx")])
    conj = abc[abc.braco == "S33@cls"].groupby("instancia").conjunto.first().reindex(t.index)
    print(f"=== ABC: {len(t)} instancias; X: {len(tx)} instancias; gap medio (%) por braco e regime ===")
    for r in REG:
        cols = [c for c in (t.columns if r != "longx" else tx.columns) if c.endswith(f"@{r}")]
        print(f"  {r:8s} " + "  ".join(f"{c.split('@')[0]} {(t if r != 'longx' else tx)[c].mean():.3f}" for c in cols))
    t = t.copy()
    tx = tx.copy()
    for r in ("longabc", "short", "acc"):
        t[f"DiD@{r}"] = (t[f"RP@{r}"] - t[f"S33@{r}"]) - (t["RP@cls"] - t["S33@cls"])
    tx["DiD@longx"] = (tx["RP@longx"] - tx["S33@longx"]) - (tx["RP@refx"] - tx["S33@refx"])
    t["zero"], tx["zero"] = 0.0, 0.0
    C = lambda cid, a, b, g, **kw: contraste(t, cid, a, b, rng, g, **kw)  # noqa: E731
    CX = lambda cid, a, b, g, **kw: contraste(tx, cid, a, b, rng, g, **kw)  # noqa: E731

    prim = com_holm([C("H130 RP-S33@cls", "RP@cls", "S33@cls", "primario"), C("H131 DiD(RP-S33) longabc-cls", "DiD@longabc", "zero", "primario"),
                     CX("H132 DiD(RP-S33) longx-refx", "DiD@longx", "zero", "primario"), C("H133 DiD(RP-S33) short-cls", "DiD@short", "zero", "primario"),
                     C("H134 DiD(RP-S33) acc-cls", "DiD@acc", "zero", "primario")], 5)
    sec = com_holm([C("H135 S33r-S33@acc", "S33r@acc", "S33@acc", "secundario"), C("H136 S33-U@cls", "S33@cls", "U@cls", "secundario"),
                    C("H137 RP-U@cls", "RP@cls", "U@cls", "secundario"), C("H138 HG-U@cls", "HG@cls", "U@cls", "secundario")], 4)
    expl = []
    for r, arms in REG.items():
        tt, ct = (tx, CX) if r == "longx" else (t, C)
        for a in arms:
            if a != "U":
                expl.append(ct(f"{a}-U@{r}", f"{a}@{r}", f"U@{r}", "explor"))
        expl.append(ct(f"RP-S33@{r}", f"RP@{r}", f"S33@{r}", "explor"))
    expl.append(C("RP-S33r@acc", "RP@acc", "S33r@acc", "explor"))
    for a, b in (("RPt", "S33"), ("S65", "S33"), ("RPt", "RP"), ("HG", "RP")):
        expl.append(C(f"{a}-{b}@cls", f"{a}@cls", f"{b}@cls", "explor"))
    for a in ("U", "RP", "S33"):                                              # efeito do regime sobre cada braco (mesmas instancias)
        for r in ("short", "longabc", "acc"):
            expl.append(C(f"{a}@{r}-{a}@cls", f"{a}@{r}", f"{a}@cls", "explor_regime"))
        expl.append(CX(f"{a}@longx-{a}@refx", f"{a}@longx", f"{a}@refx", "explor_regime"))
    for c in sorted(set(conj)):
        sub = t[(conj == c).values]
        for r in ("cls", "short", "longabc", "acc"):
            expl.append(contraste(sub, f"RP-S33@{r}@{c}", f"RP@{r}", f"S33@{r}", rng, "explor_set", conjunto=c))
    ex = pd.DataFrame(expl)
    ex["p_holm"], ex["signif_005"] = ex.p_bruto, ex.p_bruto < 0.05
    r = pd.concat([prim, sec, ex], ignore_index=True)
    print("\n=== primarios (Holm m=5), secundarios (Holm m=4) e exploratorios ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))

    g = lambda cid: r[r.id.str.startswith(cid + " ")].iloc[0]  # noqa: E731
    neg = lambda x: bool(x.p_holm < 0.05 and x.dif_pp < 0)  # noqa: E731
    H = {
        "H130 RP - S33 em A/B/CMT significativamente negativo (primario, Holm m=5)": neg(g("H130")),
        "H131 DiD de (RP - S33), orcamento 20x contra 1x, significativamente negativa (Holm m=5)": neg(g("H131")),
        "H132 DiD de (RP - S33), orcamento 4x contra 1x em X, significativamente negativa (Holm m=5)": neg(g("H132")),
        "H133 DiD de (RP - S33), orcamento 0,25x contra 1x, significativamente negativa (Holm m=5)": neg(g("H133")),
        "H134 DiD de (RP - S33), record-to-record travel contra recozimento, significativamente negativa (Holm m=5)": neg(g("H134")),
        "H135 S33r - S33 sob record-to-record travel significativamente negativo (Holm m=4)": neg(g("H135")),
        "H136 S33 - U em A/B/CMT significativamente negativo (Holm m=4)": neg(g("H136")),
        "H137 RP - U em A/B/CMT significativamente negativo (Holm m=4)": neg(g("H137")),
        "H138 HG - U em A/B/CMT significativamente negativo (Holm m=4)": neg(g("H138")),
    }
    print("\n=== hipoteses (pre-registro, Entrada 138) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de 9 (esperado no pre-registro: 3,75)")

    tam = abc[abc.braco == "S33@cls"].groupby("instancia").n_customers.first().reindex(t.index)
    d_rp = t["RP@cls"] - t["S33@cls"]
    rho_, p_ = stats.spearmanr(tam.values, d_rp.values)
    print(f"\ncorrelacao (Spearman) de RP - S33 @cls com o tamanho: {rho_:+.3f} (p {p_:.3f}), n {len(d_rp)}")
    rows = []
    print("\n=== exploratorio: uso medio de operadores (fracao de iteracoes), destruicoes | reparos ===")
    for r_, arms in REG.items():
        for a in arms:
            h = dados[dados.braco == f"{a}@{r_}"]
            if a in ("RP", "S33", "S33r", "HG"):
                m = {c: pd.DataFrame(h[c].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean() for c in ("uso_d", "uso_r")}
                ud, ur = (m["uso_d"] / m["uso_d"].sum()).round(3).tolist(), (m["uso_r"] / m["uso_r"].sum()).round(3).tolist()
                print(f"  {r_:8s} {a:5s} d {ud}  r {ur}")
                rows.append(dict(regime=r_, braco=a, uso_d="|".join(map(str, ud)), uso_r="|".join(map(str, ur))))
    if not ensaio:
        esc = json.loads((RES / "etapa22_ajuste_escolhidas.json").read_text())
        v = pd.read_csv(RES / "etapa22_validacao.csv")
        v = v[v.viavel]
        print(f"\n=== exploratorio: validacao (instancias de ajuste, seeds 3003-3005), gap medio (%): {v.groupby('braco').gap_pct.mean().round(4).sort_values().to_dict()}")
        print(f"    delta0 escolhido {esc['delta0']}; S33r: gap de ajuste {esc['S33r']['gap_ajuste']:.4f}, pesos_d {esc['S33r']['pesos_d']}, pesos_r {esc['S33r']['pesos_r']}")
        r.to_csv(RES / "analise_etapa22.csv", index=False)
        pd.DataFrame(rows).to_csv(RES / "analise_etapa22_uso.csv", index=False)
        pd.DataFrame([dict(rho=float(rho_), p=float(p_), n=len(d_rp))]).to_csv(RES / "analise_etapa22_tamanho.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa22_hipoteses.csv", index=False)


if __name__ == "__main__":
    main(ensaio="--ensaio" in sys.argv)
