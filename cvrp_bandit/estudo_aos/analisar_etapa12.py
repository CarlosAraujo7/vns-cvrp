#!/usr/bin/env python3
"""
analisar_etapa12.py
===================
Analise do Estudo 12 (controle de mistura estatica no ALNS) conforme o PRE-REGISTRO (diario, Entrada 109).

  Confirmatorio (51 instancias X + SYN_VAL; Holm m=5): static - U; reg2 - U; RP - static; RP - reg2; PM - static
  Exploratorio : por conjunto; equivalencia (IC90% em +-0,10pp); fracao do ganho do RP sobre U capturada pela mistura
                 estatica e pelo reparo fixo, rec = (U - x)/(U - RP), com IC95% bootstrap sobre instancias.
dif = gap(a) - gap(b); negativo = a melhor. Wilcoxon bilateral pareado por instancia (media das seeds), IC95% bootstrap
(10.000, semente 0), empate = |dif| < 1e-9. As referencias U, RP, PM... vem de etapa9_eval.csv (mesmo codigo e dados).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
N_BOOT = 10000


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
    e9 = pd.read_csv(RES / "etapa9_eval.csv")
    e12 = pd.read_csv(RES / "etapa12_eval.csv")
    print(f"etapa12_eval: {len(e12)} execucoes, inviaveis {int((~e12.viavel).sum())}")
    esc = json.loads((RES / "etapa12_tuning_escolhida.json").read_text())
    print(f"static_A: gap_tune {esc['gap_tune']:.3f} (pesos iguais {esc['gap_pesos_iguais']:.3f}); destruir "
          f"{[round(x, 3) for x in esc['pesos_d']]} reparar {[round(x, 3) for x in esc['pesos_r']]}")
    d = pd.concat([e9[e9.braco.isin(["s9_U", "s9_RP", "s9_pm", "s9_exp3", "s9_hgucb"])], e12])
    d = d[d.viavel]
    t = d.groupby(["braco", "instancia"]).gap_pct.mean().unstack(0).dropna()
    t.columns = [c.replace("s9_", "").replace("s12_", "") for c in t.columns]
    conj = e9.drop_duplicates("instancia").set_index("instancia").conjunto.reindex(t.index)
    print(f"{len(t)} instancias completas ({(conj == 'X').sum()} X, {(conj == 'SYN_VAL').sum()} SYN_VAL)")
    print("\nGap medio (%):")
    print(pd.DataFrame({"X": t[conj == "X"].mean(), "SYN_VAL": t[conj == "SYN_VAL"].mean(), "todas": t.mean()}).round(3)
          .sort_values("todas").to_string())
    pares = [("static", "U"), ("reg2", "U"), ("RP", "static"), ("RP", "reg2"), ("pm", "static")]
    linhas = [contraste(t, f"{a}-{b}", a, b, rng, "confirmatorio") for a, b in pares]
    for c in ("X", "SYN_VAL"):
        sub = t[conj == c]
        for a, b in pares:
            linhas.append(contraste(sub, f"{a}-{b}@{c}", a, b, rng, "exploratorio"))
    r = pd.DataFrame(linhas)
    cf = r.grupo == "confirmatorio"
    r.loc[cf, "p_holm"] = holm(r.loc[cf, "p_bruto"].values, 5)
    r["signif_005"] = r.p_holm < 0.05
    pd.set_option("display.width", 220)
    print("\n=== CONFIRMATORIO (Holm m=5) ===")
    print(r[cf].drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    print("\n=== EXPLORATORIO (por conjunto, sem correcao) ===")
    print(r[~cf].drop(columns=["grupo", "a", "b", "p_holm", "signif_005"]).round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa12_confirmatoria.csv", index=False)
    # fracao do ganho do RP capturada
    ganho_rp = (t["U"] - t["RP"]).values
    out = []
    for x in ("static", "reg2", "pm"):
        ganho_x = (t["U"] - t[x]).values
        idx = rng.integers(0, len(ganho_rp), size=(N_BOOT, len(ganho_rp)))
        rec = ganho_x[idx].mean(1) / ganho_rp[idx].mean(1)
        out.append(dict(braco=x, ganho_pp=ganho_x.mean(), rec=ganho_x.mean() / ganho_rp.mean(),
                        rec_lo=np.percentile(rec, 2.5), rec_hi=np.percentile(rec, 97.5)))
    out = pd.DataFrame(out)
    print("\n=== FRACAO DO GANHO DO RP SOBRE U capturada (exploratorio) ===")
    print(out.round(3).to_string(index=False))
    out.to_csv(RES / "analise_etapa12_captura.csv", index=False)
    s = r[cf].set_index("id")
    print(f"\nH50 (static significativamente melhor que U, Holm): dif {s.loc['static-U', 'dif_pp']:+.3f}pp p aj. {s.loc['static-U', 'p_holm']:.4f}")
    print(f"H51 (RP - static nao significativo): dif {s.loc['RP-static', 'dif_pp']:+.3f}pp p aj. {s.loc['RP-static', 'p_holm']:.4f} -> "
          f"{'CONFIRMADA' if s.loc['RP-static', 'p_holm'] >= 0.05 else 'nao confirmada'}")
    print(f"H52 (reg2 significativamente melhor que U, Holm): dif {s.loc['reg2-U', 'dif_pp']:+.3f}pp p aj. {s.loc['reg2-U', 'p_holm']:.4f}")
    print(f"H53 (static captura >= 70% do ganho do RP): {out.set_index('braco').loc['static', 'rec']:.2f}")


if __name__ == "__main__":
    main()
