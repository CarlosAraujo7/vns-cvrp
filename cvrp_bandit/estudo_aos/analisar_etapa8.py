#!/usr/bin/env python3
"""
analisar_etapa8.py
==================
Analise do Estudo 8 (controle positivo / pool inflado) conforme o PRE-REGISTRO (diario, Entrada 101). So o que la
esta fixado e confirmatorio; o resto e EXPLORATORIO.

Confirmatorio (74 instancias CVRPLIB, E1, 10 seeds; Holm m=12 = 6 politicas adaptativas x 2 pools com inuteis):
  para cada pool P in {P10+D4x4, P10+D4x8} e politica a in {ucb1c, pmc, hgucbc, ts, exp3, eff}:
      dif = gap(s8_P_a) - gap(s8_P_U)           (negativo = a adaptativa melhor que a uniforme DO MESMO pool)
Exploratorio: as mesmas 6 comparacoes no pool limpo P10 (replicacao dos Estudos 3 e 6; sem correcao);
  recuperacao da perda plantada, rec = 1 - perda(a)/perda(U), perda(x) = gap(x, pool inflado) - gap(x, P10)
  (IC95% bootstrap sobre instancias); desvio das adaptativas em relacao a U do pool LIMPO.
Wilcoxon bilateral pareado por instancia (media das seeds), IC95% bootstrap (10.000, semente 0), empate = |dif|<1e-9.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
N_BOOT = 10000
POLS = ["ucb1c", "pmc", "hgucbc", "ts", "exp3", "eff"]
POOLS_INFL = ["P10+D4x4", "P10+D4x8"]
M_FAMILIA = 12


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
    return dict(grupo=grupo, id=cid, a=a, b=b, n_inst=len(dif), dif_pp=dif.mean(), ic_lo=lo, ic_hi=hi, vit_a=int((dif < 0).sum()),
                der_a=int((dif > 0).sum()), emp=int((dif == 0).sum()), p_bruto=p,
                equiv_010=bool(lo90 > -0.1 and hi90 < 0.1), mde_pp=2.8 * dif.std(ddof=1) / np.sqrt(len(dif)))


def main():
    rng = np.random.default_rng(0)
    d = pd.read_csv(RES / "etapa8_full.csv")
    print(f"etapa8_full: {len(d)} execucoes, inviaveis {int((~d.viavel).sum())}")
    d = d[d.viavel]
    t = d.groupby(["braco", "instancia"]).gap_pct.mean().unstack(0)
    n_inst = t.dropna().shape[0]
    t = t.dropna()
    print(f"{n_inst} instancias completas")
    print("\nGap medio (%) por pool x politica:")
    tab = pd.DataFrame({p: {a: t[f"s8_{p}_{a}"].mean() for a in ["U"] + POLS} for p in ["P10"] + POOLS_INFL})
    print(tab.round(3).to_string())
    linhas = []
    for pool in POOLS_INFL:
        for a in POLS:
            l = contraste(t, f"{pool}:{a}", f"s8_{pool}_{a}", f"s8_{pool}_U", rng, "confirmatorio")
            linhas.append(l)
    for a in POLS:
        linhas.append(contraste(t, f"P10:{a}", f"s8_P10_{a}", "s8_P10_U", rng, "expl_pool_limpo"))
    r = pd.DataFrame(linhas)
    conf = r.grupo == "confirmatorio"
    r.loc[conf, "p_holm"] = holm(r.loc[conf, "p_bruto"].values, M_FAMILIA)
    r["signif_005"] = r.p_holm < 0.05
    pd.set_option("display.width", 230)
    print("\n=== CONFIRMATORIO (dif = gap(a) - gap(U do mesmo pool); Holm m=12) ===")
    print(r[conf].drop(columns=["grupo"]).round(4).to_string(index=False))
    print(f"Significativos: {int(r[conf].signif_005.sum())} de {int(conf.sum())}; "
          f"com a adaptativa MELHOR: {int((r[conf].signif_005 & (r[conf].dif_pp < 0)).sum())}")
    print("\n=== EXPLORATORIO: pool limpo (sem correcao) ===")
    print(r[~conf].drop(columns=["grupo", "p_holm", "signif_005"]).round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa8_confirmatoria.csv", index=False)
    # recuperacao da perda plantada
    rec = []
    for pool in POOLS_INFL:
        perda_u = t[f"s8_{pool}_U"].values - t["s8_P10_U"].values
        for a in POLS:
            perda_a = t[f"s8_{pool}_{a}"].values - t[f"s8_P10_{a}"].values
            idx = rng.integers(0, len(perda_u), size=(N_BOOT, len(perda_u)))
            rr = 1 - perda_a[idx].mean(1) / perda_u[idx].mean(1)
            rec.append(dict(pool=pool, politica=a, perda_U=perda_u.mean(), perda_pol=perda_a.mean(),
                            recuperacao=1 - perda_a.mean() / perda_u.mean(),
                            rec_lo=np.percentile(rr, 2.5), rec_hi=np.percentile(rr, 97.5),
                            gap_vs_U_limpo=(t[f"s8_{pool}_{a}"] - t["s8_P10_U"]).mean()))
    rec = pd.DataFrame(rec)
    print("\n=== RECUPERACAO (exploratorio; IC95% bootstrap sobre instancias) ===")
    print(rec.round(3).to_string(index=False))
    rec.to_csv(RES / "analise_etapa8_recuperacao.csv", index=False)
    # participacao das avaliacoes gastas nos inuteis (operadores 10..13) sob a uniforme
    print("\n[exploratorio] % das avaliacoes gastas nos operadores inuteis (indices >= 10):")
    for pool in POOLS_INFL:
        for a in ["U"] + POLS:
            g = d[d.braco == f"s8_{pool}_{a}"]
            arr = np.array([[int(v) for v in s.split("|")] for s in g.calls_por_operador])
            print(f"  {pool:9s} {a:7s}: {100 * arr[:, 10:].sum() / arr.sum():5.1f}%")


if __name__ == "__main__":
    main()
