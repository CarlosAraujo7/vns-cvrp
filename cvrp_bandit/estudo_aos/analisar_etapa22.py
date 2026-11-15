#!/usr/bin/env python3
"""
analisar_etapa22.py
===================
Analise do Estudo 22 (deslocamentos de implantacao do conjunto de 21 operadores P21: classe de instancias, orcamento, regra de aceitacao) conforme o PRE-REGISTRO ATUALIZADO pela EMENDA 1
(diario, Entradas 138 e 140; as hipoteses H130-H138 da Entrada 138, sobre P7, foram retiradas antes de qualquer dado). dif = gap(a) - gap(b); negativo = a melhor. Unidade = instancia
(media das seeds viaveis); Wilcoxon bilateral pareado; IC95% bootstrap (10.000, semente 0); empate |dif| < 1e-9; MDE = 2,8 DP / raiz(n); equivalencia = IC90% dentro de +-0,10 pp.
Instancias: ABC = Augerat A (26), B (23) e CMT (4) = 53; X = 21. Regimes (sufixo @): cls (1.500 n^2, recozimento), short (375 n^2), longabc (30.000 n^2, 5 seeds), longx (6.000 n^2, X, 3 seeds),
acc (record-to-record travel); refx = os bracos de P21 em X do Estudo 21 (1.500 n^2, recozimento, 10 seeds). S* = a mistura estatica de maior esforco (S129 se a emenda do Estudo 21 a achou
melhor que S65; senao S65). Se RPt = RP, S65 = S33 ou S129 = S65, o braco nao foi executado e a coluna e copiada.

  Primarios (Holm m=5)    : H141-H145  RP - U em cls, short, longabc, longx, acc
  Secundarios (Holm m=5)  : H146-H150  RP - S* em cls, short, longabc, longx, acc
  Terciarios (Holm m=3)   : H151 HG - U @cls; H152 S* - U @cls; H153 S33 - U @cls (confirmada se POSITIVO: a mistura de 33 avaliacoes e pior que a uniforme)
  Exploratorios (rotular) : cada braco contra U e contra S* em cada regime; DiD de (RP - U) e de (RP - S*) entre cada regime deslocado e a sua referencia (cls para short, longabc e acc;
                            refx para longx); por conjunto (A, B, CMT); correlacao de RP - U com o tamanho; uso de operadores e iteracoes.
Uso: python estudo_aos/analisar_etapa22.py [--ensaio]   (--ensaio fabrica dados para exercitar o codigo; nada e gravado em results/)
"""
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
COLS = ["braco", "conjunto", "instancia", "n_customers", "seed", "viavel", "gap_pct", "uso_d", "uso_r", "iteracoes"]
REG = {"cls": ("U", "RP", "RPt", "S33", "S65", "S129", "HG"), "short": ("U", "RP", "S65", "S129"), "longabc": ("U", "RP", "S65", "S129"), "longx": ("U", "RP", "S65", "S129"),
       "acc": ("U", "RP", "S65", "S129")}
REGS = tuple(REG)


def ler(reg):
    d = pd.read_csv(RES / f"etapa22_eval_{reg}.csv")
    d["braco"] = d.braco.str.replace(f"s22_{reg}_", "", regex=False) + f"@{reg}"
    return d[COLS]


def referencia_x():
    e = pd.read_csv(RES / "etapa21_eval_P21.csv")
    e = e[e.conjunto == "X"]
    out = [e[e.braco == f"s21_P21_{a}"][COLS].assign(braco=f"{a}@refx") for a in ("U", "RP", "S33", "S65")]
    f = RES / "etapa21b_eval_P21.csv"
    if f.exists():
        e2 = pd.read_csv(f)
        out.append(e2[e2.conjunto == "X"][COLS].assign(braco="S129@refx"))
    return pd.concat(out, ignore_index=True)


def fabricar(rng):
    """ENSAIO: dados fabricados (efeitos conhecidos + ruido) para exercitar o codigo."""
    nomes = {c: sorted(p.stem for p in (INST / c).glob("*.vrp")) for c in ("A", "B", "CMT", "X")}
    efeito = {"U": 0.25, "RP": 0.0, "S33": 0.35, "RPt": 0.03, "S65": 0.22, "S129": 0.10, "HG": 0.20}
    out = []
    for reg, arms in REG.items():
        conjs = ("X",) if reg == "longx" else ("A", "B", "CMT")
        for c in conjs:
            for a in arms:
                for n in nomes[c]:
                    for sd in range(3):
                        out.append(dict(braco=f"{a}@{reg}", conjunto=c, instancia=n, n_customers=50 + len(n), seed=1000 + sd, viavel=True,
                                        gap_pct=1.0 + efeito[a] + rng.normal(0, 0.25), uso_d="1|1|1|1|1|1|1|1|1|1|1|1", uso_r="1|1|1|1|1|1|1|1|1", iteracoes=1000))
    for a in ("U", "RP", "S33", "S65", "S129"):
        for n in nomes["X"]:
            for sd in range(3):
                out.append(dict(braco=f"{a}@refx", conjunto="X", instancia=n, n_customers=100, seed=1000 + sd, viavel=True, gap_pct=1.3 + efeito[a] + rng.normal(0, 0.25),
                                uso_d="1|1|1|1|1|1|1|1|1|1|1|1", uso_r="1|1|1|1|1|1|1|1|1", iteracoes=1000))
    return pd.DataFrame(out)


def completar(d):
    """Bracos nao executados (RPt = RP, S65 = S33, S129 = S65): colunas copiadas."""
    nomes = set(d.braco)
    extra = []
    for r in tuple(REGS) + ("refx",):
        arms = REG.get(r, ("U", "RP", "S33", "S65", "S129"))
        for novo, fonte in (("RPt", "RP"), ("S65", "S33"), ("S129", "S65")):
            if novo in arms and f"{novo}@{r}" not in nomes and f"{fonte}@{r}" in nomes:
                extra.append(d[d.braco == f"{fonte}@{r}"].assign(braco=f"{novo}@{r}"))
                print(f"[{r}] {novo} ausente: copiado de {fonte}")
    return pd.concat([d] + extra, ignore_index=True) if extra else d


def main(ensaio=False):
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    dados = fabricar(np.random.default_rng(7)) if ensaio else pd.concat([ler(r) for r in REG] + [referencia_x()], ignore_index=True)
    dados = completar(dados)
    abc = dados[dados.conjunto.isin(["A", "B", "CMT"])]
    tx = pivo(dados[dados.conjunto == "X"])
    t = pivo(abc)
    conj = abc[abc.braco == "U@cls"].groupby("instancia").conjunto.first().reindex(t.index)
    for pv in (t, tx):                                                       # S* = S129 (ja copiado de S65 se nao executado)
        for r_ in REGS + ("refx",):
            if f"S129@{r_}" in pv.columns:
                pv[f"S*@{r_}"] = pv[f"S129@{r_}"]
    print(f"=== ABC: {len(t)} instancias; X: {len(tx)} instancias; gap medio (%) por braco e regime ===")
    for r_ in REGS:
        pv = tx if r_ == "longx" else t
        print(f"  {r_:8s} " + "  ".join(f"{c.split('@')[0]} {pv[c].mean():.3f}" for c in pv.columns if c.endswith(f"@{r_}") and "*" not in c))
    print("  refx     " + "  ".join(f"{c.split('@')[0]} {tx[c].mean():.3f}" for c in tx.columns if c.endswith("@refx") and "*" not in c))
    t, tx = t.copy(), tx.copy()
    for r_ in ("short", "longabc", "acc"):
        t[f"DiDU@{r_}"] = (t[f"RP@{r_}"] - t[f"U@{r_}"]) - (t["RP@cls"] - t["U@cls"])
        t[f"DiDS@{r_}"] = (t[f"RP@{r_}"] - t[f"S*@{r_}"]) - (t["RP@cls"] - t["S*@cls"])
    tx["DiDU@longx"] = (tx["RP@longx"] - tx["U@longx"]) - (tx["RP@refx"] - tx["U@refx"])
    tx["DiDS@longx"] = (tx["RP@longx"] - tx["S*@longx"]) - (tx["RP@refx"] - tx["S*@refx"])
    t["zero"], tx["zero"] = 0.0, 0.0
    C = lambda cid, a, b, g, **kw: contraste(t, cid, a, b, rng, g, **kw)  # noqa: E731
    CX = lambda cid, a, b, g, **kw: contraste(tx, cid, a, b, rng, g, **kw)  # noqa: E731
    cc = {r_: (CX if r_ == "longx" else C) for r_ in REGS}

    prim = com_holm([cc[r_](f"H{141 + i} RP-U@{r_}", f"RP@{r_}", f"U@{r_}", "primario") for i, r_ in enumerate(REGS)], 5)
    sec = com_holm([cc[r_](f"H{146 + i} RP-S*@{r_}", f"RP@{r_}", f"S*@{r_}", "secundario") for i, r_ in enumerate(REGS)], 5)
    ter = com_holm([C("H151 HG-U@cls", "HG@cls", "U@cls", "terciario"), C("H152 S*-U@cls", "S*@cls", "U@cls", "terciario"),
                    C("H153 S33-U@cls", "S33@cls", "U@cls", "terciario")], 3)
    expl = []
    for r_ in REGS:
        pv = tx if r_ == "longx" else t
        for a in REG[r_]:
            if a != "U":
                expl.append(cc[r_](f"{a}-U@{r_}", f"{a}@{r_}", f"U@{r_}", "explor"))
            if a not in ("S129",):
                expl.append(cc[r_](f"{a}-S*@{r_}", f"{a}@{r_}", f"S*@{r_}", "explor"))
    for a, b in (("RPt", "RP"), ("S65", "S33"), ("S129", "S65"), ("HG", "RP")):
        expl.append(C(f"{a}-{b}@cls", f"{a}@cls", f"{b}@cls", "explor"))
    for r_ in ("short", "longabc", "acc"):
        expl.append(C(f"DiD(RP-U) {r_}-cls", f"DiDU@{r_}", "zero", "explor_did"))
        expl.append(C(f"DiD(RP-S*) {r_}-cls", f"DiDS@{r_}", "zero", "explor_did"))
    expl.append(CX("DiD(RP-U) longx-refx", "DiDU@longx", "zero", "explor_did"))
    expl.append(CX("DiD(RP-S*) longx-refx", "DiDS@longx", "zero", "explor_did"))
    for c in sorted(set(conj)):
        sub = t[(conj == c).values]
        for r_ in ("cls", "short", "longabc", "acc"):
            expl.append(contraste(sub, f"RP-U@{r_}@{c}", f"RP@{r_}", f"U@{r_}", rng, "explor_set", conjunto=c))
            expl.append(contraste(sub, f"RP-S*@{r_}@{c}", f"RP@{r_}", f"S*@{r_}", rng, "explor_set", conjunto=c))
    ex = pd.DataFrame(expl)
    ex["p_holm"], ex["signif_005"] = ex.p_bruto, ex.p_bruto < 0.05
    r = pd.concat([prim, sec, ter, ex], ignore_index=True)
    print("\n=== primarios (Holm m=5), secundarios (Holm m=5), terciarios (Holm m=3) e exploratorios ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))

    g = lambda cid: r[r.id.str.startswith(cid + " ")].iloc[0]  # noqa: E731
    neg = lambda x: bool(x.p_holm < 0.05 and x.dif_pp < 0)  # noqa: E731
    nomes_r = {"cls": "A/B/CMT", "short": "orcamento 0,25x", "longabc": "orcamento 20x", "longx": "orcamento 4x em X", "acc": "record-to-record travel"}
    H = {}
    for i, r_ in enumerate(REGS):
        H[f"H{141 + i} RP - U ({nomes_r[r_]}) significativamente negativo (primario, Holm m=5)"] = neg(g(f"H{141 + i}"))
    for i, r_ in enumerate(REGS):
        H[f"H{146 + i} RP - S* ({nomes_r[r_]}) significativamente negativo (secundario, Holm m=5)"] = neg(g(f"H{146 + i}"))
    H["H151 HG - U em A/B/CMT significativamente negativo (terciario, Holm m=3)"] = neg(g("H151"))
    H["H152 S* - U em A/B/CMT significativamente negativo (terciario, Holm m=3)"] = neg(g("H152"))
    H["H153 S33 - U em A/B/CMT significativamente POSITIVO (terciario, Holm m=3)"] = bool(g("H153").p_holm < 0.05 and g("H153").dif_pp > 0)
    print("\n=== hipoteses (pre-registro, Entradas 138 e 140) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de 13 (esperado no pre-registro da emenda: 6,45)")

    tam = abc[abc.braco == "U@cls"].groupby("instancia").n_customers.first().reindex(t.index)
    d_rp = t["RP@cls"] - t["U@cls"]
    rho_, p_ = stats.spearmanr(tam.values, d_rp.values)
    print(f"\ncorrelacao (Spearman) de RP - U @cls com o tamanho: {rho_:+.3f} (p {p_:.3f}), n {len(d_rp)}")
    rows = []
    print("\n=== exploratorio: iteracoes medias por execucao e uso medio (fracao), destruicoes | reparos ===")
    for r_ in REGS:
        for a in REG[r_]:
            h = dados[dados.braco == f"{a}@{r_}"]
            if h.empty:
                continue
            m = {c: pd.DataFrame(h[c].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean() for c in ("uso_d", "uso_r")}
            ud, ur = (m["uso_d"] / m["uso_d"].sum()).round(3).tolist(), (m["uso_r"] / m["uso_r"].sum()).round(3).tolist()
            print(f"  {r_:8s} {a:5s} it {h.iteracoes.mean():8.0f}  d {ud}  r {ur}")
            rows.append(dict(regime=r_, braco=a, iteracoes=float(h.iteracoes.mean()), uso_d="|".join(map(str, ud)), uso_r="|".join(map(str, ur))))
    if not ensaio:
        r.to_csv(RES / "analise_etapa22.csv", index=False)
        pd.DataFrame(rows).to_csv(RES / "analise_etapa22_uso.csv", index=False)
        pd.DataFrame([dict(rho=float(rho_), p=float(p_), n=len(d_rp))]).to_csv(RES / "analise_etapa22_tamanho.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa22_hipoteses.csv", index=False)


if __name__ == "__main__":
    main(ensaio="--ensaio" in sys.argv)
