#!/usr/bin/env python3
"""
analisar_etapa11.py
===================
Analise do Estudo 11 (os resultados centrais sobrevivem a um SWAP* EFICIENTE?) conforme o PRE-REGISTRO (diario).

  Familia A (Holm m=21): {ucb1c, pmc, hgucbc, ts, exp3, eff, static_F} - U em {CVRPLIB E1, CVRPLIB E2, SYN_VAL_L E1}
  Familia B (Holm m=2) : efeito do conjunto, U(P10F) - U(P5), em CVRPLIB E1 e E2
  Exploratorio         : participacao do SWAP* nas avaliacoes sob U; tendencia de static_F com n (Spearman); equivalencia
dif = gap(a) - gap(b); negativo = a melhor. Wilcoxon bilateral pareado por instancia (media das seeds), IC95% bootstrap
(10.000, semente 0), empate = |dif| < 1e-9.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
N_BOOT = 10000
POL = ["ucb1c", "pmc", "hgucbc", "ts", "exp3", "eff", "static_F"]


def holm(pvals, m):
    ordem = np.argsort(pvals)
    aj = np.empty(len(pvals))
    corrente = 0.0
    for rank, idx in enumerate(ordem):
        corrente = max(corrente, min(1.0, (m - rank) * pvals[idx]))
        aj[idx] = corrente
    return aj


def contraste(piv, cid, a, b, rng, grupo):
    dif = piv[a].values - piv[b].values
    dif = np.where(np.abs(dif) < 1e-9, 0.0, dif)
    idx = rng.integers(0, len(dif), size=(N_BOOT, len(dif)))
    med = dif[idx].mean(axis=1)
    lo, hi = np.percentile(med, [2.5, 97.5])
    lo90, hi90 = np.percentile(med, [5, 95])
    p = stats.wilcoxon(dif).pvalue if np.any(dif != 0) else 1.0
    return dict(grupo=grupo, id=cid, a=a, b=b, n_inst=len(dif), dif_pp=dif.mean(), ic_lo=lo, ic_hi=hi,
                vit_a=int((dif < 0).sum()), der_a=int((dif > 0).sum()), emp=int((dif == 0).sum()), p_bruto=p,
                equiv_010=bool(lo90 > -0.1 and hi90 < 0.1), mde_pp=2.8 * dif.std(ddof=1) / np.sqrt(len(dif)))


def main():
    rng = np.random.default_rng(0)
    e74 = pd.read_csv(RES / "etapa11_eval74.csv")
    eL = pd.read_csv(RES / "etapa11_evalL.csv")
    print(f"eval74: {len(e74)} execucoes, inviaveis {int((~e74.viavel).sum())}; evalL: {len(eL)}, inviaveis {int((~eL.viavel).sum())}")
    linhas = []
    cenarios = [("CVRPLIB E1", e74[e74.nivel == 1]), ("CVRPLIB E2", e74[e74.nivel == 5]), ("SYN_VAL_L E1", eL)]
    gaps = {}
    for nome, d in cenarios:
        d = d[d.viavel]
        t = d.groupby(["braco", "instancia"]).gap_pct.mean().unstack(0).dropna()
        t.columns = [c.replace("s11_", "") for c in t.columns]
        gaps[nome] = t
        print(f"\n=== {nome}: {len(t)} instancias; gap medio ===")
        print(t.mean().round(3).sort_values().to_string())
        for a in POL:
            linhas.append(dict(contraste(t, f"{a}-U@{nome}", a, "U", rng, "familiaA"), cenario=nome))
    r = pd.DataFrame(linhas)
    A = r.grupo == "familiaA"
    r.loc[A, "p_holm"] = holm(r.loc[A, "p_bruto"].values, 21)
    extra = []
    for nome in ("CVRPLIB E1", "CVRPLIB E2"):
        t = gaps[nome]
        extra.append(dict(contraste(t, f"P10F-U vs P5-U@{nome}", "U", "P5_U", rng, "familiaB"), cenario=nome))
    eb = pd.DataFrame(extra)
    eb["p_holm"] = holm(eb.p_bruto.values, 2)
    r = pd.concat([r, eb], ignore_index=True)
    r["signif_005"] = r.p_holm < 0.05
    pd.set_option("display.width", 240)
    print("\n=== FAMILIA A (Holm m=21) e B (Holm m=2) ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa11_confirmatoria.csv", index=False)
    A_ = r[r.grupo == "familiaA"]
    on = A_[A_.a != "static_F"]
    print(f"\nH45 (nenhuma politica online significativamente melhor): {int(((on.dif_pp < 0) & on.signif_005).sum())} melhores -> "
          f"{'CONFIRMADA' if not ((on.dif_pp < 0) & on.signif_005).any() else 'nao confirmada'}")
    sf = A_[A_.a == "static_F"]
    print(f"H46 (static_F significativamente melhor em algum cenario): {int(((sf.dif_pp < 0) & sf.signif_005).sum())} de 3")
    B_ = r[r.grupo == "familiaB"]
    print(f"H47 (efeito do conjunto persiste, >= 0,3pp, ambos significativos): "
          f"{int(((B_.dif_pp <= -0.3) & B_.signif_005).sum())} de 2")
    # SWAP* eficiente: participacao nas avaliacoes sob U
    for nome, d in (("CVRPLIB E1", e74[(e74.nivel == 1) & (e74.braco == "s11_U")]), ("CVRPLIB E2", e74[(e74.nivel == 5) & (e74.braco == "s11_U")]),
                    ("SYN_VAL_L E1", eL[eL.braco == "s11_U"])):
        arr = np.array([[int(x) for x in s.split("|")] for s in d.calls_por_operador])
        pct = 100 * arr.sum(0) / arr.sum()
        print(f"H48 (explor) {nome}: SWAP*-rapido absorve {pct[9]:.1f}% das avaliacoes; CROSS {pct[8]:.1f}%")
    t = gaps["SYN_VAL_L E1"]
    n = eL.drop_duplicates("instancia").set_index("instancia").n_customers.reindex(t.index)
    rho = stats.spearmanr(n, t["static_F"] - t["U"])
    print(f"tendencia de static_F com n (SYN_VAL_L): Spearman {rho.statistic:+.3f} (p={rho.pvalue:.3f})")
    # participacao do SWAP* e CROSS por conjunto, para o texto
    pd.DataFrame([dict(cenario=nome, swap=100 * np.array([[int(x) for x in s.split('|')] for s in d.calls_por_operador]).sum(0)[9]
                       / np.array([[int(x) for x in s.split('|')] for s in d.calls_por_operador]).sum())
                  for nome, d in (("CVRPLIB E1", e74[(e74.nivel == 1) & (e74.braco == "s11_U")]),
                                  ("SYN_VAL_L E1", eL[eL.braco == "s11_U"]))]).to_csv(RES / "analise_etapa11_swap.csv", index=False)


if __name__ == "__main__":
    main()
