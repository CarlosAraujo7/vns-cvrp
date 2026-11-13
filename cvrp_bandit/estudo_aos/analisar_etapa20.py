#!/usr/bin/env python3
"""
analisar_etapa20.py
===================
Analise do Estudo 20 (generalidade do indicio dos Estudos 18 e 19 no ALNS do CVRP e no ALNS do TSP) conforme o PRE-REGISTRO (diario, Entrada 132).
dif = gap(a) - gap(b); negativo = a melhor. Unidade = instancia (media das seeds viaveis); Wilcoxon bilateral pareado; IC95% bootstrap (10.000, semente 0);
empate |dif| < 1e-9; MDE = 2,8 DP / raiz(n). Conjuntos: CVRP = X (21 instancias, seeds 1000-1009) + SYN_VAL (30, seeds 1000-1004); TSP = TSPLIB (32, seeds
1000-1009) + SYN_VAL_T (30, seeds 1000-1004); AGREGADO = 113 instancias (cada instancia pesa o mesmo; a diferenca e em pontos percentuais de gap).
Bracos: U e static (a mistura estatica de 33 avaliacoes dos Estudos 12 e 13) REUTILIZADOS dos Estudos 9, 12 e 13; novos: RP (33), tilt (33 + 33), tilt0
(33), static65 (65 avaliacoes) e tilt65 (emenda 1: hibrido padrao sobre static65; 65). Se o ES nao achou melhora, static65 = static e tilt65 = tilt0 (colunas copiadas).

  Primario (sem correcao, alfa 0,05)  : H111 tilt - static, agregado
  Segundo primario (EMENDA 1, Entrada 133, antes de qualquer dado de avaliacao; sem correcao, alfa 0,05; cada afirmacao fica presa a UM teste):
                                        H119 tilt65 - static65, agregado (o hibrido sobre a mistura de 65 avaliacoes, hiperparametros padrao, contra ela: esforco 65 e 65)
  Secundarios (Holm m=4)              : H112 tilt - static65, agregado; H113 tilt0 - static, agregado; H114 tilt - static, CVRP; H115 tilt - static, TSP
  Terciarios (Holm m=3)               : H116 static65 - static, agregado; H117 tilt - RP, agregado; H118 RP - static, agregado (confirmada se RP e PIOR)
  Exploratorios (rotular)             : cada braco contra U; contrastes por conjunto; equivalencia (IC90% em +-0,10pp); MDE; metanalise de efeitos aleatorios
                                        (DerSimonian-Laird) de tilt - static nos quatro conjuntos de replicacao (Estudo 19 SYN e SOL com seeds novas; Estudo 20
                                        CVRP e TSP); validacao nas instancias de ajuste; uso de operadores; correlacao do efeito com o tamanho.
Uso: python estudo_aos/analisar_etapa20.py [--ensaio]   (--ensaio fabrica dados para exercitar o codigo; nada e gravado em results/)
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
COLS = ["braco", "conjunto", "instancia", "n_customers", "seed", "viavel", "gap_pct", "uso_d", "uso_r"]
PROBS = ("CVRP", "TSP")
NOVOS = ("RP", "tilt", "tilt0", "static65", "tilt65")


def referencias():
    e9, e12, e13 = (pd.read_csv(RES / f) for f in ("etapa9_eval.csv", "etapa12_eval.csv", "etapa13_eval_a.csv"))
    sel = lambda d, b, novo: d[d.braco == b][COLS].assign(braco=novo)  # noqa: E731
    return {"CVRP": pd.concat([sel(e9, "s9_U", "U"), sel(e12, "s12_static", "static")], ignore_index=True),
            "TSP": pd.concat([sel(e13, "s13a_U", "U"), sel(e13, "s13a_static", "static")], ignore_index=True)}


def fabricar(ref, rng):
    """ENSAIO: bracos novos fabricados a partir do static (efeitos conhecidos + ruido) para exercitar o codigo."""
    efeito = {"RP": 0.25, "tilt": -0.12, "tilt0": -0.06, "static65": -0.03, "tilt65": -0.05}
    out = {}
    for p in PROBS:
        st = ref[p][ref[p].braco == "static"]
        out[p] = pd.concat([st.assign(braco=b, gap_pct=st.gap_pct + efeito[b] + rng.normal(0, 0.35, len(st))) for b in efeito], ignore_index=True)
    return out


def ler_novos(p):
    d = pd.read_csv(RES / f"etapa20_eval_{p}.csv")
    d["braco"] = d.braco.str.replace(f"s20_{p}_", "", regex=False)
    return d[COLS]


def meta_re(efs, ses):
    """Efeitos aleatorios de DerSimonian-Laird (e fixo): estimativa, IC95%, tau^2, I^2, Q e p de Q."""
    efs, ses = np.asarray(efs, float), np.asarray(ses, float)
    w = 1 / ses ** 2
    mu_f = float((w * efs).sum() / w.sum())
    q = float((w * (efs - mu_f) ** 2).sum())
    k = len(efs)
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / c)
    wr = 1 / (ses ** 2 + tau2)
    mu = float((wr * efs).sum() / wr.sum())
    se = float(np.sqrt(1 / wr.sum()))
    return dict(k=k, mu_fixo=mu_f, mu_re=mu, ic_lo=mu - 1.96 * se, ic_hi=mu + 1.96 * se, tau2=tau2, I2=(max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0),
                Q=q, p_Q=float(stats.chi2.sf(q, k - 1)))


def estudo19():
    """tilt - static18 do Estudo 19 (so a replicacao): SYN_VAL_W (48 instancias novas) e SOLOMON com seeds novas (56)."""
    e = pd.read_csv(RES / "etapa19_eval.csv")
    e["braco"] = e.braco.str.replace("s19_", "", regex=False)
    out = {}
    for rotulo, c in (("E19-SYN", "SYN_VAL_W"), ("E19-SOL", "SOLOMON")):
        t = pivo(e[e.conjunto == c])
        out[rotulo] = (t["tilt"] - t["static18"]).values
    return out


def main(ensaio=False):
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    ref = referencias()
    novos = fabricar(ref, np.random.default_rng(7)) if ensaio else {p: ler_novos(p) for p in PROBS}
    dados = {p: pd.concat([ref[p], novos[p]], ignore_index=True) for p in PROBS}
    piv, conjuntos = {}, {}
    for p in PROBS:
        d = dados[p]
        if "static65" not in set(d.braco):
            print(f"[{p}] static65 ausente: a continuacao do ES nao achou melhora (static65 = static, tilt65 = tilt0); colunas copiadas")
            d = pd.concat([d, d[d.braco == "static"].assign(braco="static65"), d[d.braco == "tilt0"].assign(braco="tilt65")], ignore_index=True)
        t = pivo(d)
        assert set(t.columns) == {"U", "static", "RP", "tilt", "tilt0", "static65", "tilt65"}, (p, list(t.columns))
        piv[p] = t
        conjuntos[p] = d[d.braco == "static"].groupby("instancia").conjunto.first().reindex(t.index)
        print(f"=== {p}: {len(t)} instancias (execucoes inviaveis nos bracos novos: {int((~novos[p].viavel).sum())}); gap medio (%) ===\n"
              f"{t.mean().round(3).sort_values().to_string()}")
    agr = pd.concat([piv["CVRP"], piv["TSP"]], keys=list(PROBS))
    print(f"=== AGREGADO: {len(agr)} instancias; gap medio (%) ===\n{agr.mean().round(3).sort_values().to_string()}")

    prim = pd.DataFrame([contraste(agr, "H111 tilt-static@agregado", "tilt", "static", rng, "primario"),
                         contraste(agr, "H119 tilt65-static65@agregado", "tilt65", "static65", rng, "primario2")])
    prim["p_holm"], prim["signif_005"] = prim.p_bruto, prim.p_bruto < 0.05
    sec = com_holm([contraste(agr, "H112 tilt-static65@agregado", "tilt", "static65", rng, "secundario"),
                    contraste(agr, "H113 tilt0-static@agregado", "tilt0", "static", rng, "secundario"),
                    contraste(piv["CVRP"], "H114 tilt-static@CVRP", "tilt", "static", rng, "secundario"),
                    contraste(piv["TSP"], "H115 tilt-static@TSP", "tilt", "static", rng, "secundario")], 4)
    ter = com_holm([contraste(agr, "H116 static65-static@agregado", "static65", "static", rng, "terciario"),
                    contraste(agr, "H117 tilt-RP@agregado", "tilt", "RP", rng, "terciario"),
                    contraste(agr, "H118 RP-static@agregado", "RP", "static", rng, "terciario")], 3)
    expl = []
    for nome, t in (("agregado", agr), ("CVRP", piv["CVRP"]), ("TSP", piv["TSP"])):
        for a, b in (("tilt", "U"), ("tilt0", "U"), ("static", "U"), ("static65", "U"), ("RP", "U"), ("tilt65", "U"), ("tilt", "tilt0"),
                     ("tilt", "static65"), ("tilt65", "static65"), ("tilt65", "static"), ("tilt65", "tilt0"), ("tilt65", "tilt")):
            if nome == "agregado" and (a, b) == ("tilt65", "static65"):
                continue                                                   # e a H119 (ja esta no bloco primario)
            if (a, b) == ("tilt", "static65") and nome == "agregado":
                continue                                                   # e a H112
            expl.append(contraste(t, f"{a}-{b}@{nome}", a, b, rng, "explor"))
    for p in PROBS:
        for c in sorted(conjuntos[p].unique()):
            t = piv[p][(conjuntos[p] == c).values]
            for a, b in (("tilt", "static"), ("tilt0", "static"), ("RP", "static"), ("static65", "static"), ("tilt65", "static65")):
                expl.append(contraste(t, f"{a}-{b}@{c}", a, b, rng, "explor_conjunto", conjunto=c))
    ex = pd.DataFrame(expl)
    ex["p_holm"], ex["signif_005"] = ex.p_bruto, ex.p_bruto < 0.05
    r = pd.concat([prim, sec, ter, ex], ignore_index=True)
    print("\n=== primario (sem correcao), secundarios (Holm m=4), terciarios (Holm m=3) e exploratorios ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))

    g = lambda cid: r[r.id.str.startswith(cid)].iloc[0]  # noqa: E731
    neg = lambda x: bool(x.p_holm < 0.05 and x.dif_pp < 0)  # noqa: E731
    H = {
        "H111 tilt - static (agregado, 113 inst.) significativamente negativo (primario)": bool(g("H111").p_bruto < 0.05 and g("H111").dif_pp < 0),
        "H112 tilt - static65 (agregado) significativamente negativo (Holm m=4)": neg(g("H112")),
        "H113 tilt0 - static (agregado) significativamente negativo (Holm m=4)": neg(g("H113")),
        "H114 tilt - static no CVRP significativamente negativo (Holm m=4)": neg(g("H114")),
        "H115 tilt - static no TSP significativamente negativo (Holm m=4)": neg(g("H115")),
        "H116 static65 - static (agregado) significativamente negativo (Holm m=3)": neg(g("H116")),
        "H117 tilt - RP (agregado) significativamente negativo (Holm m=3)": neg(g("H117")),
        "H118 RP - static (agregado) significativamente POSITIVO, RP pior (Holm m=3)": bool(g("H118").p_holm < 0.05 and g("H118").dif_pp > 0),
        "H119 tilt65 - static65 (agregado) significativamente negativo (segundo primario, emenda 1, sem correcao)":
            bool(g("H119").p_bruto < 0.05 and g("H119").dif_pp < 0),
    }
    print("\n=== hipoteses (pre-registro, Entrada 132; emenda 1, Entrada 133) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de 9 (esperado no pre-registro e na emenda: 3,75)")

    # metanalise exploratoria de tilt - static: replicacao do Estudo 19 (SYN, SOL com seeds novas) e os dois problemas do Estudo 20
    efs = dict(estudo19())
    efs["E20-CVRP"] = (piv["CVRP"]["tilt"] - piv["CVRP"]["static"]).values
    efs["E20-TSP"] = (piv["TSP"]["tilt"] - piv["TSP"]["static"]).values
    est = pd.DataFrame([dict(conjunto=k, n=len(v), dif_pp=v.mean(), ep=v.std(ddof=1) / np.sqrt(len(v))) for k, v in efs.items()])
    print("\n=== exploratorio: tilt - static nos quatro conjuntos de replicacao ===")
    print(est.round(4).to_string(index=False))
    m_all = meta_re(est.dif_pp, est.ep)
    m_20 = meta_re(est.dif_pp[2:], est.ep[2:])
    print("metanalise (efeitos aleatorios, DL), 4 conjuntos:", {k: round(v, 4) for k, v in m_all.items()})
    print("metanalise (efeitos aleatorios, DL), so Estudo 20:", {k: round(v, 4) for k, v in m_20.items()})
    todas = np.concatenate(list(efs.values()))
    print(f"estimativa bruta sobre as {len(todas)} instancias de replicacao (as do Estudo 19 e as do 20): {todas.mean():+.4f} pp")

    print("\n=== exploratorio: correlacao (Spearman) de tilt - static com o tamanho da instancia ===")
    tam = pd.concat([dados[p][dados[p].braco == "static"].groupby("instancia").n_customers.first() for p in PROBS], keys=list(PROBS))
    d_ts = agr["tilt"] - agr["static"]
    rho_, p_ = stats.spearmanr(tam.reindex(d_ts.index).values, d_ts.values)
    print(f"  rho {rho_:+.3f} (p {p_:.3f}), n {len(d_ts)}")

    print("\n=== exploratorio: uso medio de operadores (fracao), destruicoes | reparos ===")
    for p in PROBS:
        for a in ("static", "RP", "tilt", "tilt0"):
            h = dados[p][dados[p].braco == a]
            m = {c: pd.DataFrame(h[c].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean() for c in ("uso_d", "uso_r")}
            print(f"  {p:4s} {a:8s} d {(m['uso_d'] / m['uso_d'].sum()).round(3).tolist()}  r {(m['uso_r'] / m['uso_r'].sum()).round(3).tolist()}")

    if not ensaio:
        print("\n=== exploratorio: validacao (instancias de ajuste, seeds 3003-3005), gap medio (%) ===")
        for p in PROBS:
            v = pd.read_csv(RES / f"etapa20_validacao_{p}.csv")
            v = v[v.viavel].assign(braco=lambda d: d.braco.str.replace(f"s20_{p}_", "", regex=False))
            print(f"  {p}: {v.groupby('braco').gap_pct.mean().round(4).sort_values().to_dict()}")
            esc = pd.read_json(RES / f"etapa20_tuning_escolhidas_{p}.json", typ="series")
            print(f"     escolhidas: RP {esc['RP']['params']} (gap_tune {esc['RP']['gap_tune']:.4f}, padrao {esc['RP']['gap_padrao']:.4f}); "
                  f"tilt {esc['tilt']['params']} (gap_tune {esc['tilt']['gap_tune']:.4f}, padrao {esc['tilt']['gap_padrao']:.4f}); "
                  f"static65 mudou={esc['static65']['mudou']} (gap_tune {esc['static65']['gap_tune_33']:.4f} -> {esc['static65']['gap_tune_65']:.4f})")
        r.to_csv(RES / "analise_etapa20.csv", index=False)
        est.to_csv(RES / "analise_etapa20_replicacoes.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa20_hipoteses.csv", index=False)


if __name__ == "__main__":
    main(ensaio="--ensaio" in sys.argv)
