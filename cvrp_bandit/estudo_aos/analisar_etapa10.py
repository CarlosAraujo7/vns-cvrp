#!/usr/bin/env python3
"""
analisar_etapa10.py
===================
Analise do Estudo 10 (o valor da selecao aprendida cresce com n?) conforme o PRE-REGISTRO (diario, Entrada 104).

  Confirmatorio (Holm m=7, SYN_VAL_L, E1, 5 seeds): cada braco {static_S, static_L, ucb1c, pmc, hgucbc, ts, eff} - U
  Exploratorio: tendencia com n (Spearman da diferenca por instancia com n, para static_L, static_S e ucb1c), diferenca
                media por tamanho, participacao do SWAP*/CROSS nas avaliacoes sob U por tamanho, equivalencia.
dif = gap(a) - gap(U); negativo = a melhor.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
N_BOOT = 10000
BR = ["static_S", "static_L", "ucb1c", "pmc", "hgucbc", "ts", "eff"]
OPS = ["2-Opt", "Move(1)", "Swap(1,1)", "Reloc(1,0)", "Reloc(2,0)", "2-opt*", "Or-opt", "Swap-intra", "CROSS", "SWAP*"]


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
    e = pd.read_csv(RES / "etapa10_eval.csv")
    print(f"etapa10_eval: {len(e)} execucoes, inviaveis {int((~e.viavel).sum())}")
    esc = json.loads((RES / "etapa10_tuning_escolhida.json").read_text())
    print(f"static_L: config {esc['config_id']}  gap_tune_L {esc['gap_tune']:.3f} (pesos iguais {esc['gap_pesos_iguais']:.3f}); "
          f"pesos {[round(x, 3) for x in esc['params']['pesos']]}")
    e = e[e.viavel]
    t = e.groupby(["braco", "instancia"]).gap_pct.mean().unstack(0).dropna()
    t.columns = [c.replace("s10_", "") for c in t.columns]
    n = e.drop_duplicates("instancia").set_index("instancia").n_customers.reindex(t.index)
    print(f"{len(t)} instancias completas")
    print("\nGap medio (%) por braco:")
    print(t.mean().round(3).sort_values().to_string())
    rows = [contraste(t, f"{a}-U", a, "U", rng, "confirmatorio") for a in BR]
    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r.p_bruto.values, 7)
    r["signif_005"] = r.p_holm < 0.05
    pd.set_option("display.width", 220)
    print("\n=== CONFIRMATORIO (Holm m=7) ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa10_confirmatoria.csv", index=False)
    print("\n=== EXPLORATORIO: tendencia com o tamanho ===")
    exp = []
    for a in ("static_L", "static_S", "ucb1c"):
        dif = (t[a] - t["U"])
        rho = stats.spearmanr(n, dif)
        por = dif.groupby(n.values).mean()
        print(f"  {a}: Spearman(dif, n) = {rho.statistic:+.3f} (p={rho.pvalue:.3f}); dif media por n: "
              + "  ".join(f"{int(k)}:{v:+.3f}" for k, v in por.items()))
        exp.append(dict(braco=a, rho=rho.statistic, p=rho.pvalue, **{f"dif_n{int(k)}": v for k, v in por.items()}))
    pd.DataFrame(exp).to_csv(RES / "analise_etapa10_tamanho.csv", index=False)
    print("\n[exploratorio] participacao (%) do SWAP* e do CROSS nas avaliacoes sob U, por tamanho:")
    g = e[e.braco == "s10_U"]
    for nn, gg in g.groupby("n_customers"):
        arr = np.array([[int(x) for x in s.split("|")] for s in gg.calls_por_operador])
        pct = 100 * arr.sum(0) / arr.sum()
        print(f"  n={nn}: SWAP* {pct[9]:.1f}%  CROSS {pct[8]:.1f}%")
    s = r.set_index("a")
    sl = s.loc["static_L"]
    ss = s.loc["static_S"]
    on = r[r.a.isin(["ucb1c", "pmc", "hgucbc", "ts", "eff"])]
    print(f"\nH40 (static_L significativamente melhor que U, Holm, >= 0,2pp): dif {sl.dif_pp:+.3f}pp p aj. {sl.p_holm:.4f} -> "
          f"{'CONFIRMADA' if sl.p_holm < 0.05 and sl.dif_pp <= -0.2 else 'nao confirmada'}")
    print(f"H41 (static_S significativamente melhor que U, Holm): dif {ss.dif_pp:+.3f}pp p aj. {ss.p_holm:.4f} -> "
          f"{'CONFIRMADA' if ss.p_holm < 0.05 and ss.dif_pp < 0 else 'nao confirmada'}")
    print(f"H42 (nenhuma politica online significativamente melhor): {int(((on.dif_pp < 0) & (on.p_holm < 0.05)).sum())} melhores -> "
          f"{'CONFIRMADA' if not ((on.dif_pp < 0) & (on.p_holm < 0.05)).any() else 'nao confirmada'}")


if __name__ == "__main__":
    main()
