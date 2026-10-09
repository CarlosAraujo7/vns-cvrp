#!/usr/bin/env python3
"""
analisar_etapa4.py
==================
Analise da Etapa 4 conforme o PRE-REGISTRO (diario, Entrada 83). Implementa
SOMENTE o que la esta fixado como confirmatorio; o resto e EXPLORATORIO.

Confirmatorio nas 74 (E1 e E2; Holm m=14, 7 testes por nivel):
  V1-V3: s4_R5_{ucb1t,pmt,hgucbt} vs s3_R5_U
  V4   : s4_R5_static vs s3_R5_U
  V5-V7: s4_R5_{ucb1t,pmt,hgucbt} vs s4_R5_static
  Omnibus: Friedman em {U, ucb1t, pmt, hgucbt, static}.
Confirmatorio na validacao sintetica SYN_VAL (E1, 5 seeds, Holm m=6):
  s3_LO_U vs s3_P5_LO_U; s3_R5_FOC vs s3_R5_U; s4_R5_{ucb1t,pmt,hgucbt} vs s3_R5_U;
  s4_R5_static vs s3_R5_U.
Controle: s4_R5_U_instr reproduz s3_R5_U exatamente (custo identico).

dif = gap(a) - gap(b); negativo = a melhor. Wilcoxon bilateral pareado por
instancia (media das seeds), IC95% bootstrap (10.000, semente 0).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
N_BOOT = 10000
TUNED = ["s4_R5_ucb1t", "s4_R5_pmt", "s4_R5_hgucbt"]
OPS = ["2-Opt", "Move(1)", "Swap(1,1)", "Reloc(1,0)", "Reloc(2,0)", "2-opt*", "Or-opt", "Swap-intra", "CROSS", "SWAP*"]


def holm(pvals, m):
    ordem = np.argsort(pvals)
    aj = np.empty(len(pvals))
    corrente = 0.0
    for rank, idx in enumerate(ordem):
        corrente = max(corrente, min(1.0, (m - rank) * pvals[idx]))
        aj[idx] = corrente
    return aj


def testes(piv, pares, rng, rotulo, m):
    linhas = []
    for cid, a, b in pares:
        x, y = piv[a].values, piv[b].values
        dif = x - y
        dif = np.where(np.abs(dif) < 1e-9, 0.0, dif)   # empates: diferencas de ponto flutuante (~1e-15) nao sao vitorias/derrotas
        idx = rng.integers(0, len(dif), size=(N_BOOT, len(dif)))
        lo, hi = np.percentile(dif[idx].mean(axis=1), [2.5, 97.5])
        lo90, hi90 = np.percentile(dif[idx].mean(axis=1), [5, 95])
        linhas.append(dict(grupo=rotulo, id=cid, a=a, b=b, dif_pp=dif.mean(), ic_lo=lo, ic_hi=hi,
                           vit_a=int((dif < 0).sum()), der_a=int((dif > 0).sum()), emp=int((dif == 0).sum()),
                           p_bruto=stats.wilcoxon(dif).pvalue, equiv_010=bool(lo90 > -0.1 and hi90 < 0.1)))
    return linhas


def main():
    rng = np.random.default_rng(0)
    todas = []
    # ---------------- 74 CVRPLIB ----------------
    e4 = pd.read_csv(RES / "etapa4_eval74.csv")
    e3 = pd.read_csv(RES / "etapa3_full.csv")
    print(f"eval74: {len(e4)} execucoes, inviaveis {int((~e4.viavel).sum())}")
    ctrl = e4[e4.braco == "s4_R5_U_instr"][["instancia", "seed", "custo"]].merge(
        e3[(e3.braco == "s3_R5_U") & (e3.nivel == 1)][["instancia", "seed", "custo"]], on=["instancia", "seed"],
        suffixes=("_i", "_3"))
    print(f"CONTROLE U instrumentado vs s3_R5_U (E1): {int((ctrl.custo_i == ctrl.custo_3).sum())}/{len(ctrl)} identicos")
    pool74 = pd.concat([e4[e4.braco.isin(TUNED + ["s4_R5_static"])][["braco", "nivel", "instancia", "seed", "gap_pct", "viavel"]],
                        e3[e3.braco.isin(["s3_R5_U", "s3_R5_ucb1c", "s3_R5_pmc", "s3_R5_hgucbc"])][
                            ["braco", "nivel", "instancia", "seed", "gap_pct", "viavel"]]])
    pool74 = pool74[pool74.viavel]
    pares = [("V1", TUNED[0], "s3_R5_U"), ("V2", TUNED[1], "s3_R5_U"), ("V3", TUNED[2], "s3_R5_U"),
             ("V4", "s4_R5_static", "s3_R5_U"), ("V5", TUNED[0], "s4_R5_static"),
             ("V6", TUNED[1], "s4_R5_static"), ("V7", TUNED[2], "s4_R5_static")]
    for nv in (1, 5):
        t = pool74[pool74.nivel == nv].groupby(["braco", "instancia"]).gap_pct.mean().unstack(0)
        print(f"\n=== 74 CVRPLIB, E{nv}: gap medio ===")
        print(t.mean().round(3).sort_values().to_string())
        cols = ["s3_R5_U"] + TUNED + ["s4_R5_static"]
        fr = stats.friedmanchisquare(*[t[c].values for c in cols])
        print(f"Friedman (U, ucb1t, pmt, hgucbt, static): chi2={fr.statistic:.2f} p={fr.pvalue:.3g}")
        for l in testes(t, pares, rng, f"74_E{nv}", 14):
            l["nivel"] = nv
            todas.append(l)
        # exploratorio: efeito do ajuste (padrao -> ajustada)
        print("  [exploratorio] ajustada - padrao (mesma familia):")
        for tn, pd_ in (("s4_R5_ucb1t", "s3_R5_ucb1c"), ("s4_R5_pmt", "s3_R5_pmc"), ("s4_R5_hgucbt", "s3_R5_hgucbc")):
            dif = t[tn] - t[pd_]
            print(f"    {tn} - {pd_}: {dif.mean():+.3f}pp  p={stats.wilcoxon(t[tn], t[pd_]).pvalue:.3g}")
    r74 = pd.DataFrame([l for l in todas if l["grupo"].startswith("74")])
    r74["p_holm"] = holm(r74.p_bruto.values, 14)
    # ---------------- validacao sintetica ----------------
    val = pd.read_csv(RES / "etapa4_val.csv")
    val = val[val.viavel]
    tv = val.groupby(["braco", "instancia"]).gap_pct.mean().unstack(0)
    print(f"\n=== SYN_VAL (30 inst., 5 seeds, E1): gap medio ===")
    print(tv.mean().round(3).sort_values().to_string())
    pv = [("W1", "s3_LO_U", "s3_P5_LO_U"), ("W2", "s3_R5_FOC", "s3_R5_U"), ("W3", TUNED[0], "s3_R5_U"),
          ("W4", TUNED[1], "s3_R5_U"), ("W5", TUNED[2], "s3_R5_U"), ("W6", "s4_R5_static", "s3_R5_U")]
    rv = pd.DataFrame(testes(tv, pv, rng, "VAL_E1", 6))
    rv["p_holm"] = holm(rv.p_bruto.values, 6)
    out = pd.concat([r74, rv], ignore_index=True)
    out["signif_005"] = out.p_holm < 0.05
    pd.set_option("display.width", 220)
    print("\n=== CONFIRMATORIO (dif = gap(a) - gap(b)) ===")
    print(out.drop(columns=["grupo"]).round(4).to_string(index=False))
    out.to_csv(RES / "analise_etapa4_confirmatoria.csv", index=False)
    # ---------------- exploratorio: escolhas e eficiencia ----------------
    esc = json.loads((RES / "etapa4_tuning_escolhidas.json").read_text())
    print("\n[exploratorio] configuracoes escolhidas no ajuste (SYN_TUNE):")
    for k, v in esc.items():
        pesos = v["params"].get("pesos")
        extra = f" pesos={[round(x, 3) for x in pesos]}" if pesos else f" {v['params']}"
        print(f"  {k}: gap_tune={v['gap_tune']:.4f}{extra}")
    ins = e4[e4.braco == "s4_R5_U_instr"]
    suc = np.array([[int(v) for v in s.split("|")] for s in ins.suc_por_operador]).sum(0)
    uso = np.array([[int(v) for v in s.split("|")] for s in ins.uso_operadores]).sum(0)
    cal = np.array([[int(v) for v in s.split("|")] for s in ins.calls_por_operador]).sum(0)
    gan = np.array([[float(v) for v in s.split("|")] for s in ins.ganho_por_operador]).sum(0)
    print("\n[exploratorio] eficiencia por operador (U, R5, P10, E1, 74 inst. x 10 seeds):")
    print(f"  {'operador':12s} {'aplic.':>9s} {'sucesso%':>9s} {'ganho/sucesso(%)':>17s} {'ganho/1000 dist':>16s}")
    for j, nome in enumerate(OPS):
        print(f"  {nome:12s} {uso[j]:9d} {100*suc[j]/uso[j]:9.2f} {100*gan[j]/max(suc[j],1):17.4f} {1000*gan[j]/cal[j]*100:16.5f}")


if __name__ == "__main__":
    main()
