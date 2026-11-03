#!/usr/bin/env python3
"""
gerar_figuras.py
================
Gera as figuras do manuscrito (PDF vetorial) a partir dos CSVs brutos:
  fig_forest.pdf     diferencas pareadas (IC95% em dois niveis) dos contrastes "AOS vs. baseline" (Estudos 1-4 e 6)
  fig_estrutura.pdf  gap medio por estrutura de busca (Estudo 2A) em B=800 e B=4000
  fig_pool.pdf       gap medio dos bracos do Estudo 3 em E1 e E2
  fig_operadores.pdf % das avaliacoes de distancia por operador (Estudo 3) e eficiencia (Estudo 4, se existir)
Uso: python gerar_figuras.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent.parent / "results" / "aos_study"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 150})
OPS = ["2-Opt", "Move(1)", "Swap(1,1)", "Reloc(1,0)", "Reloc(2,0)", "2-opt*", "Or-opt", "Swap-intra", "CROSS", "SWAP*"]


def _pol(nome):
    for k, v in (("ucb1", "UCB1"), ("eps_greedy", "eps-greedy"), ("prob_matching", "Prob. matching"),
                 ("pmc", "Prob. matching"), ("pmt", "Prob. matching"), ("hgucb", "HG-UCB"), ("static", "static mixture"),
                 ("exp3", "EXP3"), ("_ts", "Thompson"), ("swucb", "SW-UCB"), ("_ap", "Adaptive pursuit"),
                 ("eff2", "Efficiency UCB (C=2)"), ("_eff", "Efficiency UCB (C=1)"),
                 ("ctxts", "Contextual TS"), ("ctxeff", "Contextual TS (efficiency)")):
        if k in nome:
            return v
    return nome


def fig_forest():
    r = pd.read_csv(RES / "analise_robustez.csv")
    r = r[~r.a.isin(["hgucb_literal"]) & ~r.a.str.contains("R5_U") & ~r.b.str.contains("hgucb_cego")
          & ~r.a.str.contains("static")]                 # so politicas adaptativas (a mistura estatica tem tabela propria)
    rot = []
    for _, x in r.iterrows():
        if x.estudo == "E1":
            rot.append(f"Study 1, B={x.nivel}: {_pol(x.a)} vs uniform")
        elif x.estudo == "E2":
            rot.append(f"Study 2, B={x.nivel}: {_pol(x.a)} vs random order (VND)")
        elif x.estudo == "E3":
            term = "LO" if "_LO_" in x.a else "R5"
            rot.append(f"Study 3, E{1 if x.nivel == 1 else 2}, {term}: {_pol(x.a)} vs uniform")
        elif x.estudo == "E4":
            rot.append(f"Study 4, E{1 if x.nivel == 1 else 2}: tuned {_pol(x.a)} vs uniform")
        else:
            rot.append(f"Study 6, E{1 if x.nivel == 1 else 2}: {_pol(x.a)} vs uniform")
    r["rotulo"] = rot
    r = r.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.115 * len(r) + 0.8))
    y = np.arange(len(r))
    ax.errorbar(r.dif_pp, y, xerr=[r.dif_pp - r.ic95_2n_lo, r.ic95_2n_hi - r.dif_pp], fmt="none",
                ecolor="0.55", elinewidth=1)
    eq = r["equiv_0.10"].values
    ax.scatter(r.dif_pp[eq], y[eq], s=16, color="k", zorder=3, label=r"equivalent within $\pm0.10$ pp")
    ax.scatter(r.dif_pp[~eq], y[~eq], s=16, facecolors="white", edgecolors="k", zorder=3, label="not equivalent")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(-0.1, 0.1, color="0.92", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(r.rotulo, fontsize=6)
    ax.set_xlabel("Gap difference to baseline (pp); bars: 95% two-level bootstrap")
    ax.legend(loc="lower right", frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(AQUI / "fig_forest.pdf")
    plt.close(fig)


def fig_estrutura():
    e1 = pd.read_csv(RES / "full.csv").rename(columns={"politica": "arm"})
    e2 = pd.read_csv(RES / "etapa2_full.csv").rename(columns={"braco": "arm"})
    d = pd.concat([e1[e1.arm.isin(["gvns_classico", "uniforme"])], e2[e2.arm.str.startswith("s2a")]])
    nome = {"s2a_U_best_LO": "Random/best/LO", "s2a_U_shaken_LO": "Random/shaken/LO", "gvns_classico": "Fixed/best/LO\n(classical GVNS)",
            "s2a_FO_shaken_LO": "Fixed/shaken/LO", "s2a_U_best_F40": "Random/best/F40", "uniforme": "Random/shaken/F40\n(original loop)"}
    t = d.groupby(["arm", "B", "instancia"]).gap_pct.mean().groupby(["arm", "B"]).mean().unstack()
    t = t.loc[list(nome)]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t[800], 0.4, label="B = 800", color="#7aa6c2")
    ax.bar(x + 0.2, t[4000], 0.4, label="B = 4000", color="#2f5d7c")
    ax.set_xticks(x)
    ax.set_xticklabels([nome[a] for a in t.index], fontsize=6)
    ax.set_ylabel("Mean gap (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(AQUI / "fig_estrutura.pdf")
    plt.close(fig)


def fig_pool():
    d = pd.read_csv(RES / "etapa3_full.csv")
    t = d[d.viavel].groupby(["braco", "nivel", "instancia"]).gap_pct.mean().groupby(["braco", "nivel"]).mean().unstack()
    t = t[t.index != "s3_R5_hgucb_cego"].sort_values(5)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    y = np.arange(len(t))
    ax.barh(y - 0.2, t[1], 0.4, label=r"$E_1$", color="#d9a066")
    ax.barh(y + 0.2, t[5], 0.4, label=r"$E_2$", color="#8a4b1c")
    ax.set_yticks(y)
    ax.set_yticklabels([a.replace("s3_", "") for a in t.index], fontsize=6)
    ax.set_xlabel("Mean gap (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(AQUI / "fig_pool.pdf")
    plt.close(fig)


def fig_operadores():
    d = pd.read_csv(RES / "etapa3_full.csv")
    d = d[(d.nivel == 5) & d.braco.isin(["s3_LO_U", "s3_LO_hgucbc", "s3_LO_FOC", "s3_R5_FOC"])]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    w = 0.2
    for i, b in enumerate(["s3_LO_U", "s3_LO_hgucbc", "s3_LO_FOC", "s3_R5_FOC"]):
        arr = np.array([[int(v) for v in s.split("|")] for s in d[d.braco == b].calls_por_operador])
        ax.bar(np.arange(10) + (i - 1.5) * w, 100 * arr.sum(0) / arr.sum(), w, label=b.replace("s3_", ""))
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(OPS, rotation=30, fontsize=6)
    ax.set_ylabel("% of distance evaluations")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(AQUI / "fig_operadores.pdf")
    plt.close(fig)


def fig_sintese():
    """Sintese entre estudos: cada ponto e um contraste confirmatorio 'politica adaptativa (ou mistura estatica) - baseline nao adaptativa'
    (diferenca media em pp; negativo = a politica e melhor); verde/vermelho = significativo apos Holm (melhor/pior), cinza = n.s.;
    losangos = alternativas NAO adaptativas contra a uniforme (ordem fixa de custo crescente e pool menor), onde o estudo as tem."""
    import sys
    sys.path.insert(0, str(AQUI))
    import checar_afirmacoes as ca
    t = ca.aos_vs_baseline()
    linhas = [("S1", "Original loop (S1)"), ("S2", "VND: order, policies (S2)"), ("S3", "Ten-operator pool (S3)"),
              ("S4", "Tuned policies, mixture (S4)"), ("S4val", "Synthetic validation (S4)"), ("S6", "Further policy families (S6)"),
              ("S7", "Inside HGS-CVRP (S7)"), ("S10", "Large instances (S10)"), ("S11", "Efficient SWAP* (S11)"), ("S17", "PyVRP ILS (S17)"),
              ("S13", "TSP (S13)"), ("S14", "CVRPTW (S14)")]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    rng = np.random.default_rng(5)
    nlin = len(linhas)
    for k, (cod, rot) in enumerate(linhas):
        y0 = nlin - 1 - k
        d = t[t.estudo == cod]
        if len(d) == 0:
            continue
        cores = np.where((d.dif_pp < 0) & (d.p_holm < 0.05), "#1b9e77", np.where((d.dif_pp > 0) & (d.p_holm < 0.05), "#d95f02", "0.6"))
        ax.scatter(d.dif_pp, y0 + rng.uniform(-0.22, 0.22, len(d)), c=cores, s=14, alpha=0.9, linewidths=0, zorder=3)
        nb, nw = int(((d.dif_pp < 0) & (d.p_holm < 0.05)).sum()), int(((d.dif_pp > 0) & (d.p_holm < 0.05)).sum())
        ax.text(1.55, y0, f"{nb} better, {nw} worse of {len(d)}", va="center", ha="left", fontsize=6.5, color="0.25")
    # alternativas nao adaptativas (losangos)
    alt = []
    o13 = pd.read_csv(RES / "analise_etapa13_ordem.csv") if (RES / "analise_etapa13_ordem.csv").exists() else None
    if o13 is not None:
        for _, r in o13[(o13.a == "FOC_asc") & (o13.b == "U")].iterrows():
            alt.append(("S13", r.dif_pp))
    v13 = pd.read_csv(RES / "analise_etapa13_v.csv") if (RES / "analise_etapa13_v.csv").exists() else None
    if v13 is not None:
        for _, r in v13[v13.grupo == "familiaB"].iterrows():
            alt.append(("S13", -r.dif_pp))
    v14 = pd.read_csv(RES / "analise_etapa14_v.csv") if (RES / "analise_etapa14_v.csv").exists() else None
    if v14 is not None:
        for _, r in v14[(v14.grupo == "familiaC") & (v14.a == "FOC_custo")].iterrows():
            alt.append(("S14", r.dif_pp))
        for _, r in v14[v14.grupo == "familiaB"].iterrows():
            alt.append(("S14", -r.dif_pp))
    pos = {cod: nlin - 1 - k for k, (cod, _r) in enumerate(linhas)}
    for cod, v in alt:
        ax.scatter([v], [pos[cod] - 0.38], marker="D", s=16, c="k", zorder=4, linewidths=0)
    ax.axvline(0, color="0.3", lw=0.8, zorder=1)
    ax.set_yticks(range(nlin))
    ax.set_yticklabels([r for _c, r in linhas][::-1])
    ax.set_xlim(-2.0, 3.4)
    ax.set_xlabel("Policy minus non-adaptive baseline (pp of gap; negative favors the policy)")
    ax.set_ylim(-0.7, nlin - 0.4)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#1b9e77", label="policy better (Holm $p<0.05$)"),
                       Line2D([], [], marker="o", ls="", color="#d95f02", label="policy worse (Holm $p<0.05$)"),
                       Line2D([], [], marker="o", ls="", color="0.6", label="not significant"),
                       Line2D([], [], marker="D", ls="", color="k", label="fixed order / smaller pool vs. uniform")],
              loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(AQUI / "fig_sintese.pdf")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_forest, fig_estrutura, fig_pool, fig_operadores, fig_sintese):
        f()
        print("ok", f.__name__)
