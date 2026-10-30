#!/usr/bin/env python3
"""
analisar_etapa15.py
===================
Analise do Estudo 15 (teste prospectivo da regra do "peso morto" do pool) conforme o PRE-REGISTRO (diario, Entrada 118).
Celulas: 4 pools (P5-P8) x 2 problemas (TSP: TSPLIB a E1; CVRPTW: Solomon a E1). Em cada celula, familia de 6 contrastes politica - U (Holm m=6).
Previsao (fixada nas instancias de ajuste): D_k >= 0,30 pp -> pelo menos uma politica significativamente melhor; senao nenhuma.
Exploratorios: replica SYN_VAL_T (TSP); politica - FOC do mesmo pool (Holm m=6 por celula); dose-resposta (Spearman, 10 celulas com P9).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analisar_etapa13 as a13  # noqa: E402

RES = a13.RES
POL = ["ucb1c", "pmc", "hgucbc", "ts", "exp3", "eff"]
POOLS = (5, 6, 7, 8)


def pivo_pool(d, pref, k):
    t = a13.pivo(d[d.braco.str.startswith(f"{pref}_P{k}_")])
    t.columns = [c.replace(f"{pref}_P{k}_", "") for c in t.columns]
    return t


def main():
    rng = np.random.default_rng(0)
    pj = json.loads((RES / "etapa15_predicoes.json").read_text())
    pred = pj["pools"]
    cel = []
    prim, foc, rep = [], [], []
    for prob, arq, pref, conj_prim in (("t", "etapa15_eval_t.csv", "s15t", "TSPLIB"), ("w", "etapa15_eval_w.csv", "s15w", "SOLOMON")):
        e = pd.read_csv(RES / arq)
        print(f"\n=== {prob}: {len(e)} execucoes, inviaveis {int((~e.viavel).sum())}")
        for k in POOLS:
            t = pivo_pool(e[e.conjunto == conj_prim], pref, k)
            print(f"{prob} P{k} ({len(t)} inst): " + ", ".join(f"{c} {t[c].mean():.3f}" for c in ["U"] + POL + ["FOC"]))
            r = a13.com_holm([a13.contraste(t, f"{a}-U@P{k}", a, "U", rng, "primario", prob=prob, pool=f"P{k}") for a in POL], 6)
            f = a13.com_holm([a13.contraste(t, f"{a}-FOC@P{k}", a, "FOC", rng, "explor_foc", prob=prob, pool=f"P{k}") for a in POL], 6)
            prim.append(r)
            foc.append(f)
            nb = int(((r.dif_pp < 0) & r.signif_005).sum())
            nw = int(((r.dif_pp > 0) & r.signif_005).sum())
            ganho = float(np.mean([t["U"].mean() - t[a].mean() for a in POL]))
            D = pred[prob][f"P{k}"]["D"]
            previsao = pred[prob][f"P{k}"]["previsao"]
            obs = "ganho" if nb >= 1 else "nenhum"
            cel.append(dict(prob=prob, pool=f"P{k}", D=D, previsao=previsao, observado=obs, acerto=previsao == obs, n_melhores=nb, n_piores=nw,
                            ganho_medio=ganho, gap_U=float(t["U"].mean()), efeito_pool_aval=float(t["U"].mean() - pivo_pool(e[e.conjunto == conj_prim], pref, 5)["U"].mean())))
            if prob == "t":
                ts = pivo_pool(e[e.conjunto == "SYN_VAL_T"], pref, k)
                rr = a13.com_holm([a13.contraste(ts, f"{a}-U@P{k}", a, "U", rng, "replica", prob=prob, pool=f"P{k}") for a in POL], 6)
                rep.append(rr)
                print(f"   replica SYN_VAL_T: melhores {int(((rr.dif_pp < 0) & rr.signif_005).sum())}, piores {int(((rr.dif_pp > 0) & rr.signif_005).sum())}")
    # P9 (Estudos 13 e 14), para a dose-resposta
    for prob, arq, conj, nivel, pref in (("t", "etapa13_eval_v.csv", "TSPLIB", 1, "s13v_"), ("w", "etapa14_eval_v.csv", "SOLOMON", 1, "s14v_")):
        e = pd.read_csv(RES / arq)
        t = a13.pivo(e[(e.conjunto == conj) & (e.nivel == nivel)])
        t.columns = [c.replace(pref, "") for c in t.columns]
        r = a13.com_holm([a13.contraste(t, f"{a}-U@P9", a, "U", rng, "p9", prob=prob, pool="P9") for a in POL], 21 if prob == "t" else 14)
        nb = int(((r.dif_pp < 0) & r.signif_005).sum())
        ganho = float(np.mean([t["U"].mean() - t[a].mean() for a in POL]))
        D = pred[prob]["P9"]["D"]
        cel.append(dict(prob=prob, pool="P9", D=D, previsao="(ja avaliado)", observado="ganho" if nb >= 1 else "nenhum", acerto=None, n_melhores=nb,
                        n_piores=int(((r.dif_pp > 0) & r.signif_005).sum()), ganho_medio=ganho, gap_U=float(t["U"].mean()),
                        efeito_pool_aval=float(t["U"].mean() - t["P5_U"].mean())))
    c = pd.DataFrame(cel)
    pd.set_option("display.width", 220)
    print("\n=== CELULAS ===")
    print(c.round(3).to_string(index=False))
    P = pd.concat(prim, ignore_index=True)
    F = pd.concat(foc, ignore_index=True)
    P.to_csv(RES / "analise_etapa15_prim.csv", index=False)
    F.to_csv(RES / "analise_etapa15_foc.csv", index=False)
    if rep:
        pd.concat(rep, ignore_index=True).to_csv(RES / "analise_etapa15_replica.csv", index=False)
    pro = c[c.pool != "P9"]
    n_ok = int(pro.acerto.sum())
    bt = stats.binomtest(n_ok, len(pro), 0.5, alternative="greater")
    rho = stats.spearmanr(c.D, c.ganho_medio)
    rho_aval = stats.spearmanr(c.efeito_pool_aval, c.ganho_medio)
    c["rho_D_ganho"] = rho.statistic
    c["rho_p"] = rho.pvalue
    c["rho_aval"] = rho_aval.statistic
    c["n_acertos"], c["n_celulas"], c["binom_p"] = n_ok, len(pro), bt.pvalue
    c.to_csv(RES / "analise_etapa15_celulas.csv", index=False)
    print(f"\nH74 (>= 7 de 8 previsoes corretas): {n_ok} de {len(pro)} (binomial exato contra 50%: p {bt.pvalue:.3f})")
    print(f"H75 (Spearman D x ganho medio >= 0,60 nas 10 celulas): rho {rho.statistic:.2f} (p {rho.pvalue:.3f}); com o efeito de pool da avaliacao: {rho_aval.statistic:.2f}")
    p5 = P[P.pool == "P5"]
    print(f"H76 (nenhuma politica melhor que U nos pools P5): {int(((p5.dif_pp < 0) & p5.signif_005).sum())} melhores")
    print(f"H77 (nenhuma politica melhor que o FOC do mesmo pool): {int(((F.dif_pp < 0) & F.signif_005).sum())} contrastes melhores")
    h78 = P[(P.prob == "t") & (P.pool == "P8") & (P.a == "eff")]
    print(f"H78 (TSP, eff melhor que U no P8): {bool(((h78.dif_pp < 0) & h78.signif_005).any())} ({float(h78.dif_pp.iloc[0]):+.2f} pp, p Holm {float(h78.p_holm.iloc[0]):.3f})")
    print("\n=== contrastes primarios ===")
    print(P.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
