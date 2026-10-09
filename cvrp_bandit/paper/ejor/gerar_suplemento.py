#!/usr/bin/env python3
"""
gerar_suplemento.py
===================
Gera o material suplementar do manuscrito a partir dos CSVs brutos:
  suplemento/gap_por_instancia_estudo{1,2,3}.csv   gap medio (sobre seeds) por instancia e braco
  suplemento/gap_por_conjunto.csv                  gap medio por conjunto (A/B/CMT/X) e braco
  tab_s2set.tex, tab_s3set.tex                     tabelas LaTeX por conjunto (Estudos 1-2 e 3), suplemento online
Uso: python gerar_suplemento.py
"""
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent.parent / "results" / "aos_study"
SUP = AQUI / "suplemento"
SUP.mkdir(exist_ok=True)


def carregar():
    e1 = pd.read_csv(RES / "full.csv").rename(columns={"politica": "arm"})
    e1["nivel"] = e1.B
    e2 = pd.read_csv(RES / "etapa2_full.csv").rename(columns={"braco": "arm"})
    e2["nivel"] = e2.B
    e3 = pd.read_csv(RES / "etapa3_full.csv").rename(columns={"braco": "arm"})
    return e1[e1.viavel], e2[e2.viavel], e3[e3.viavel]


def por_instancia(d, nome):
    t = d.groupby(["arm", "nivel", "conjunto", "instancia"]).gap_pct.mean().reset_index()
    t["col"] = t.arm + "@" + t.nivel.astype(str)
    w = t.pivot_table(index=["conjunto", "instancia"], columns="col", values="gap_pct").round(4)
    w.to_csv(SUP / f"gap_por_instancia_{nome}.csv")
    return t


def latex_porconjunto(t, titulo, label, niveis, rot, ordem):
    lin = [r"\begin{table}[h]", r"\caption{" + titulo + r"}\label{" + label + r"}", r"\centering\small",
           r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{l" + "cccc" * len(niveis) + "}", r"\toprule"]
    cab = " & ".join(rf"\multicolumn{{4}}{{c}}{{{r}}}" for r in rot)
    lin.append("Arm & " + cab + r" \\")
    lin.append(" & " + " & ".join("A & B & CMT & X" for _ in niveis) + r" \\")
    lin.append(r"\midrule")
    for arm, nome in ordem:
        cel = []
        for nv in niveis:
            for c in ("A", "B", "CMT", "X"):
                v = t[(t.arm == arm) & (t.nivel == nv) & (t.conjunto == c)].gap_pct.mean()
                cel.append(f"{v:.2f}")
        lin.append(nome + " & " + " & ".join(cel) + r" \\")
    lin += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(lin)


def main():
    e1, e2, e3 = carregar()
    t1, t2, t3 = por_instancia(e1, "estudo1"), por_instancia(e2, "estudo2"), por_instancia(e3, "estudo3")
    todas = pd.concat([t1.assign(estudo=1), t2.assign(estudo=2), t3.assign(estudo=3)])
    todas.groupby(["estudo", "arm", "nivel", "conjunto"]).gap_pct.mean().round(4).reset_index().to_csv(
        SUP / "gap_por_conjunto.csv", index=False)
    s2 = pd.concat([t2, t1[t1.arm.isin(["gvns_classico", "uniforme"])]])
    ord2 = [("s2a_U_best_LO", "Random/best/LO"), ("s2a_U_shaken_LO", "Random/shaken/LO"),
            ("gvns_classico", "Fixed/best/LO (GVNS)"), ("s2a_FO_shaken_LO", "Fixed/shaken/LO"),
            ("s2a_U_best_F40", "Random/best/F40"), ("uniforme", "Random/shaken/F40 (orig.)"),
            ("s2b_ucb1", "UCB1 in VND"), ("s2b_prob_matching", "Prob.\\ matching in VND"),
            ("s2b_hgucb_norm", "HG-UCB in VND"), ("s2b_eps_greedy", "$\\varepsilon$-greedy in VND")]
    ord3 = [("s3_R5_ucb1c", "R5, UCB1"), ("s3_R5_U", "R5, uniform"), ("s3_R5_pmc", "R5, PM"),
            ("s3_R5_hgucbc", "R5, HG-UCB"), ("s3_R5_FOC", "R5, FOC"), ("s3_LO_U", "LO, uniform"),
            ("s3_LO_pmc", "LO, PM"), ("s3_LO_ucb1c", "LO, UCB1"), ("s3_LO_hgucbc", "LO, HG-UCB"),
            ("s3_LO_FOC", "LO, FOC"), ("s3_P5_LO_U", "5 operators, LO, uniform"), ("s3_P5_LO_FO", "5 operators, LO, fixed")]
    (AQUI / "tab_s2set.tex").write_text(latex_porconjunto(
        s2, "Studies 1--2: mean gap (\\%) by instance set.", "tab:s2set", (800, 4000), ("$B=800$", "$B=4000$"), ord2) + "\n", encoding="utf-8")
    (AQUI / "tab_s3set.tex").write_text(latex_porconjunto(
        t3, "Study 3: mean gap (\\%) by instance set.", "tab:s3set", (1, 5), ("$E_1$", "$E_2$"), ord3) + "\n", encoding="utf-8")
    # Estudos 4-6 (se os CSVs existirem): gaps por instancia e por conjunto no material suplementar
    extra = []
    for nome, arq in (("estudo4", "etapa4_eval74.csv"), ("estudo5", "etapa5_full.csv"), ("estudo6", "etapa6_full.csv")):
        if (RES / arq).exists():
            d = pd.read_csv(RES / arq).rename(columns={"braco": "arm"})
            d = d[d.viavel]
            t = por_instancia(d, nome)
            extra.append(t.assign(estudo=int(nome[-1])))
    if extra:
        pd.concat([todas] + extra).groupby(["estudo", "arm", "nivel", "conjunto"]).gap_pct.mean().round(4).reset_index().to_csv(
            SUP / "gap_por_conjunto.csv", index=False)
    print("suplemento gerado:", sorted(p.name for p in SUP.iterdir()), "+ tab_s2set.tex, tab_s3set.tex")


if __name__ == "__main__":
    main()
