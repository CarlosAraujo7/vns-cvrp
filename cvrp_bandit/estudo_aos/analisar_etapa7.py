#!/usr/bin/env python3
"""
analisar_etapa7.py
==================
Analise da Etapa 7 conforme o PRE-REGISTRO (diario, Entrada 97). So o que la esta fixado e confirmatorio.

Unidade = instancia: media, sobre as 10 seeds, do gap no ponto de controle (geracoes). Populacoes: P1 = X + CMT (25 instancias
CVRPLIB), P2 = SYN_VAL (30 sinteticas). Confirmatorio: {ucb1, ts, pm, hg} - unif e fixed - unif, nos pontos 250 e 1000,
em P1 e P2 = 20 testes, Holm m=20; Wilcoxon bilateral pareado, IC95% bootstrap (10.000, semente 0), empate = |dif| < 1e-9.
Exploratorio: populacao conjunta, pontos 500/2000/4000, equivalencia (IC90% em +-0,05pp), tempo de CPU por geracao,
taxas de sucesso por tipo de movimento, Friedman.

dif = gap(a) - gap(b); negativo = a melhor.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
N_BOOT = 10000
ADAPT = ["ucb1", "ts", "pm", "hg"]
PRIMARIOS = (250, 1000)
EXPLOR = (500, 2000, 4000)
DELTA = 0.05
MOVES = ["Reloc U", "Reloc UX", "Reloc UX rev", "Swap U/V", "Swap UX/V", "Swap UX/VY", "2-opt", "2-opt* a", "2-opt* b"]


def holm(p, m):
    o = np.argsort(p)
    aj = np.empty(len(p))
    c = 0.0
    for r, i in enumerate(o):
        c = max(c, min(1.0, (m - r) * p[i]))
        aj[i] = c
    return aj


def contraste(t, a, b, rng):
    dif = t[a].values - t[b].values
    dif = np.where(np.abs(dif) < 1e-9, 0.0, dif)
    idx = rng.integers(0, len(dif), size=(N_BOOT, len(dif)))
    med = dif[idx].mean(axis=1)
    lo, hi = np.percentile(med, [2.5, 97.5])
    lo90, hi90 = np.percentile(med, [5, 95])
    p = stats.wilcoxon(dif).pvalue if np.any(dif != 0) else 1.0
    return dict(a=a, b=b, dif_pp=dif.mean(), ic_lo=lo, ic_hi=hi, vit_a=int((dif < 0).sum()), der_a=int((dif > 0).sum()),
                emp=int((dif == 0).sum()), p_bruto=p, equiv_005=bool(lo90 > -DELTA and hi90 < DELTA),
                mde_pp=2.8 * dif.std(ddof=1) / np.sqrt(len(dif)))


def main():
    d = pd.read_csv(RES / "etapa7_full.csv")
    print(f"etapa7_full: {len(d)} execucoes; inviaveis {int((~d.viavel).sum())}; seeds por celula "
          f"{d.groupby(['instancia', 'braco']).seed.nunique().min()}-{d.groupby(['instancia', 'braco']).seed.nunique().max()}")
    dv = d[d.viavel]
    d["pop"] = np.where(d.conjunto == "SYN_VAL", "P2", "P1")
    dv = d[d.viavel]
    rng = np.random.default_rng(0)
    linhas = []
    for pop, filtro in (("P1", d["pop"] == "P1"), ("P2", d["pop"] == "P2"), ("P1+P2", d["pop"].notna())):
        sub = dv[filtro.loc[dv.index]]
        for ponto in PRIMARIOS + EXPLOR:
            t = sub.groupby(["braco", "instancia"])[f"gap_{ponto}"].mean().unstack(0).dropna()
            if pop != "P1+P2" and ponto in PRIMARIOS:
                print(f"\n=== {pop}, {ponto} geracoes ({len(t)} instancias): gap medio ===")
                print(t.mean().round(4).sort_values().to_string())
                fr = stats.friedmanchisquare(*[t[c].values for c in t.columns])
                print(f"Friedman: chi2={fr.statistic:.2f} p={fr.pvalue:.3g}")
            for a in ADAPT + ["fixed"]:
                l = contraste(t, a, "unif", rng)
                l.update(pop=pop, ponto=ponto, grupo="confirmatorio" if (pop != "P1+P2" and ponto in PRIMARIOS) else "exploratorio",
                         n_inst=len(t))
                linhas.append(l)
            for a in ADAPT:                       # exploratorio: adaptativas vs ordem original do HGS
                l = contraste(t, a, "fixed", rng)
                l.update(pop=pop, ponto=ponto, grupo="expl_vs_fixed", n_inst=len(t))
                linhas.append(l)
    r = pd.DataFrame(linhas)
    conf = r.grupo == "confirmatorio"
    assert conf.sum() == 20, conf.sum()
    r.loc[conf, "p_holm"] = holm(r.loc[conf, "p_bruto"].values, 20)
    r["signif_005"] = r.p_holm < 0.05
    r["id"] = [f"{a}_{pop}_{p}" for a, pop, p in zip(r.a, r["pop"], r.ponto)]
    pd.set_option("display.width", 230)
    print("\n=== CONFIRMATORIO (dif = gap(a) - gap(b); Holm m=20) ===")
    print(r[conf].drop(columns=["grupo", "b", "id"]).round(4).to_string(index=False))
    print(f"\nSignificativos: {int(r[conf].signif_005.sum())} de 20; melhoras significativas: "
          f"{int(((r[conf].dif_pp < 0) & r[conf].signif_005).sum())}; pioras significativas: "
          f"{int(((r[conf].dif_pp > 0) & r[conf].signif_005).sum())}")
    # sensibilidade: inclui as execucoes que violam o limite inferior de rotas da NOSSA convencao (todas na X-n195-k51) e que
    # sao viaveis para o proprio HGS; os pontos de controle vem do .PG.csv do HGS (melhor solucao viavel para o HGS)
    sens = []
    for pop in ("P1", "P2"):
        sub = d[d["pop"] == pop]
        for ponto in PRIMARIOS:
            t = sub.groupby(["braco", "instancia"])[f"gap_{ponto}"].mean().unstack(0).dropna()
            for a in ADAPT + ["fixed"]:
                l = contraste(t, a, "unif", rng)
                l.update(pop=pop, ponto=ponto)
                sens.append(l)
    sn = pd.DataFrame(sens)
    sn["p_holm"] = holm(sn.p_bruto.values, 20)
    sn["signif_005"] = sn.p_holm < 0.05
    sn.to_csv(RES / "analise_etapa7_sensibilidade.csv", index=False)
    print(f"\n[sensibilidade] com as {int((~d.viavel).sum())} execucoes inviaveis pela convencao incluidas: "
          f"{int(sn.signif_005.sum())} de 20 significativos; maior |dif| = {sn.dif_pp.abs().max():.3f}pp")
    print("\n=== EXPLORATORIO: populacao conjunta e pontos 500/2000/4000 (sem correcao) ===")
    print(r[(~conf) & (r.grupo == "exploratorio")].drop(columns=["grupo", "b", "id", "p_holm", "signif_005"]).round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa7_confirmatoria.csv", index=False)
    # tempo de CPU por geracao
    print("\n[exploratorio] CPU por geracao (ms) e razao a `fixed` (media sobre instancias; 4000 geracoes):")
    c = dv.groupby(["braco", "instancia"]).cpu_s.mean().unstack(0)
    ms = 1000 * c / 4000
    print((ms.mean()).round(2).to_string())
    print((c.div(c["fixed"], axis=0)).mean().round(2).to_string())
    # taxas de sucesso por movimento
    print("\n[exploratorio] taxa de sucesso por tentativa (x1e-4) e % das tentativas, por tipo de movimento:")
    for b in ["unif", "ucb1", "ts", "pm", "hg"]:
        g = d[(d.braco == b) & d.tentativas.notna()]
        ten = np.array([[float(x) for x in s.split("|")] for s in g.tentativas]).sum(0)
        suc = np.array([[float(x) for x in s.split("|")] for s in g.sucessos]).sum(0)
        print(f"  {b:5s} sucesso(1e-4): " + " ".join(f"{1e4 * s / max(t, 1):6.1f}" for s, t in zip(suc, ten)))
        print(f"        % tentativas : " + " ".join(f"{100 * t / ten.sum():6.1f}" for t in ten))
    print("  movimentos:", " | ".join(MOVES))


if __name__ == "__main__":
    main()
