#!/usr/bin/env python3
"""
analisar_etapa30.py
===================
Analise do Estudo 30 (o que os pesos adaptativos acrescentam a uma faixa de remocao calibrada: 98 instancias novas, P21 do CVRP) conforme o PRE-REGISTRO (diario, Entrada 159). dif = gap(a) - gap(b);
negativo = a melhor. Unidade = instancia (media das 2 seeds viaveis); Wilcoxon bilateral pareado; IC95% bootstrap (10.000, semente 0); empate |dif| < 1e-9; MDE = 2,8 DP / raiz(n); equivalencia = IC90% em +-0,10 pp.
Bracos: do Estudo 26 (reutilizados): U, Uq (uniforme com q x 0,6), RP (pesos de Ropke e Pisinger, padrao), S65, HG; do Estudo 30: RPq (RP com q x c*_RP; se c* = 1 a coluna de RP e copiada) e SRPq (mistura estatica com
o uso medio do RPq nas validacoes, mesma escala de q).
  Primarios (Holm m=3)   : H205 RPq - Uq ; H206 RPq - RP ; H207 RPq - SRPq                                                [significativamente NEGATIVOS]
  Secundarios (Holm m=2) : H208 SRPq - Uq ; H209 Spearman(RPq - Uq, K)                                                   [significativamente NEGATIVOS; K = rotas da solucao de referencia do HGS]
  Exploratorios          : RPq - U, SRPq - U, Uq - U, RP - U, RP - Uq, SRPq - RP, RPq - S65, RPq - HG; por conjunto; participacao da escala de q no ganho do RP (razao de medias, bootstrap); iteracoes e uso.
Uso: python estudo_aos/analisar_etapa30.py [--ensaio]   (--ensaio fabrica dados para exercitar o codigo; nada e gravado em results/)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo, com_holm  # noqa: E402
import analisar_etapa21 as a21  # noqa: E402
import analisar_etapa26 as a26  # noqa: E402

RES = a21.RES
COLS = a26.COLS
ARMS = ("U", "Uq", "RP", "S65", "HG", "RPq", "SRPq")


def ler30(esc):
    d = a26.ler26()
    novo = pd.read_csv(RES / "etapa30_eval.csv")
    novo["braco"] = novo.braco.str.replace("s30_", "", regex=False)
    partes = [d, novo[COLS]]
    if esc["c_estrela"] == 1.0:
        print("[RPq] a escala c* = 1 venceu a calibracao: RPq = RP (a coluna de RP do Estudo 26 e copiada)")
        partes.append(d[d.braco == "RP"].assign(braco="RPq"))
    return pd.concat(partes, ignore_index=True)


def fabricar(rng):
    """ENSAIO: dados fabricados (efeitos conhecidos + ruido) para exercitar o codigo; devolve (dados, K por instancia)."""
    inst = [("X_FRESH", f"X-n{200 + 5 * i}-k{10 + i}", 199 + 5 * i, 10 + 2 * i) for i in range(38)] + [("SYN_REP", f"X-synr{k + 1:02d}-n{n + 1}", n, int(n / (4 + k))) for n in (100, 200, 300) for k in range(20)]
    efeito = {"U": 0.0, "Uq": -0.10, "RP": -0.25, "S65": 0.02, "HG": -0.03, "RPq": -0.28, "SRPq": -0.15}
    out = []
    for c, nome, n, kk in inst:
        base = 1.0 + 0.004 * n + rng.normal(0, 0.5)
        for a in ARMS:
            for sd in (1000, 1001):
                ef = efeito[a] - (0.004 * kk if a in ("RP", "RPq") else 0.0)
                out.append(dict(braco=a, conjunto=c, instancia=nome, n_customers=n, seed=sd, viavel=True, gap_pct=base * (1 + ef) + rng.normal(0, 0.3), uso_d="|".join(["1"] * 12),
                                uso_r="|".join(["1"] * 9), iteracoes=1000))
    return pd.DataFrame(out), pd.Series({nome: float(kk) for _c, nome, _n, kk in inst})


def razao_boot(num, den, rng, B=10000):
    """Razao de medias sum(num)/sum(den) com IC95% por bootstrap de instancias."""
    n = len(num)
    idx = rng.integers(0, n, size=(B, n))
    r = num.values[idx].mean(axis=1) / den.values[idx].mean(axis=1)
    return float(num.mean() / den.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main(ensaio=False):
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    if ensaio:
        dados, kser = fabricar(np.random.default_rng(7))
        esc = {"c_estrela": 0.6}
    else:
        esc = json.loads((RES / "etapa30_selecao.json").read_text())
        dados, kser = ler30(esc), a26.ler_k()
    t = pivo(dados).copy()
    K = kser.reindex(t.index)
    is_x = pd.Series(t.index.str.startswith("X-n"), index=t.index)
    print(f"=== {len(t)} instancias ({int(is_x.sum())} de X_FRESH, {int((~is_x).sum())} de SYN_REP); escala de q do RPq: {esc['c_estrela']}; gap medio (%) por braco ===\n{t.mean().round(3).sort_values().to_string()}")
    C = lambda cid, a, b, g, sub=None, **kw: contraste(t if sub is None else t[sub.values], cid, a, b, rng, g, **kw)  # noqa: E731
    prim = com_holm([C("H205 RPq-Uq", "RPq", "Uq", "primario"), C("H206 RPq-RP", "RPq", "RP", "primario"), C("H207 RPq-SRPq", "RPq", "SRPq", "primario")], 3)
    dif_k = t["RPq"] - t["Uq"]
    rho, p = stats.spearmanr(K, dif_k)
    lc = dict(grupo="secundario", id="H209 Spearman(RPq-Uq, K) [rho]", a="RPq-Uq", b="K", n_inst=len(t), dif_pp=float(rho), ic_lo=float("nan"), ic_hi=float("nan"), vit_a=0, der_a=0, emp=0, p_bruto=float(p),
              equiv_010=False, mde_pp=float("nan"))
    sec = com_holm([C("H208 SRPq-Uq", "SRPq", "Uq", "secundario"), lc], 2)
    expl = []
    for a, b in (("RPq", "U"), ("SRPq", "U"), ("Uq", "U"), ("RP", "U"), ("RP", "Uq"), ("SRPq", "RP"), ("RPq", "S65"), ("RPq", "HG"), ("SRPq", "RPq")):
        expl.append(C(f"{a}-{b}", a, b, "explor"))
    for nome, sub in (("X_FRESH", is_x), ("SYN_REP", ~is_x)):
        for a, b in (("RPq", "Uq"), ("RPq", "RP"), ("RPq", "U"), ("RP", "U"), ("Uq", "U")):
            expl.append(C(f"{a}-{b}@{nome}", a, b, "explor_set", sub, conjunto=nome))
    ex = pd.DataFrame(expl)
    ex["p_holm"], ex["signif_005"] = ex.p_bruto, ex.p_bruto < 0.05
    r = pd.concat([prim, sec, ex], ignore_index=True)
    print("\n=== primarios (Holm m=3), secundarios (Holm m=2) e exploratorios ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    g = lambda cid: r[r.id.str.startswith(cid + " ")].iloc[0]  # noqa: E731
    neg = lambda x: bool(x.p_holm < 0.05 and x.dif_pp < 0)  # noqa: E731
    H = {"H205 RPq - Uq significativamente negativo (primario, Holm m=3)": neg(g("H205")),
         "H206 RPq - RP significativamente negativo (primario, Holm m=3)": neg(g("H206")),
         "H207 RPq - SRPq significativamente negativo (primario, Holm m=3)": neg(g("H207")),
         "H208 SRPq - Uq significativamente negativo (secundario, Holm m=2)": neg(g("H208")),
         "H209 correlacao de Spearman de RPq - Uq com o numero de rotas K significativamente negativa (secundario, Holm m=2)": neg(g("H209"))}
    print("\n=== hipoteses (pre-registro, Entrada 159) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de 5 (esperado no pre-registro: 2,20)")
    # participacao da escala de q e da dinamica no ganho do RP (razao de medias; exploratorio)
    med = []
    for rot, num, den in (("(Uq - U) / (RP - U): parte do ganho do RP que a escala de q sozinha da a uniforme", t["Uq"] - t["U"], t["RP"] - t["U"]),
                          ("(RPq - U) / (RP - U): ganho do RP com a escala de q em relacao ao do RP padrao", t["RPq"] - t["U"], t["RP"] - t["U"]),
                          ("(RPq - Uq) / (Uq - U): o que o RP acrescenta a escala de q, em fracao do ganho da escala", t["RPq"] - t["Uq"], t["Uq"] - t["U"]),
                          ("(SRPq - U) / (RPq - U): parte do ganho do RPq que a alocacao media estatica recupera", t["SRPq"] - t["U"], t["RPq"] - t["U"])):
        est, lo, hi = razao_boot(num, den, rng)
        med.append(dict(razao=rot, estimativa=est, ic_lo=lo, ic_hi=hi))
    med = pd.DataFrame(med)
    print("\n=== exploratorio: razoes de ganhos (razao de medias; IC95% por bootstrap de instancias) ===\n" + med.round(3).to_string(index=False))
    uso = []
    for b in ("U", "Uq", "RP", "RPq", "SRPq", "S65", "HG"):
        h = dados[dados.braco == b]
        mm = {c: pd.DataFrame(h[c].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean() for c in ("uso_d", "uso_r")}
        ud, ur = (mm["uso_d"] / mm["uso_d"].sum()).round(3).tolist(), (mm["uso_r"] / mm["uso_r"].sum()).round(3).tolist()
        uso.append(dict(braco=b, iteracoes=float(h.iteracoes.mean()), uso_d="|".join(map(str, ud)), uso_r="|".join(map(str, ur))))
    uso = pd.DataFrame(uso)
    print("\n=== exploratorio: iteracoes medias e uso ===\n" + uso.to_string(index=False))
    if not ensaio:
        r.to_csv(RES / "analise_etapa30.csv", index=False)
        med.to_csv(RES / "analise_etapa30_razoes.csv", index=False)
        uso.to_csv(RES / "analise_etapa30_uso.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa30_hipoteses.csv", index=False)


if __name__ == "__main__":
    main(ensaio="--ensaio" in sys.argv)
