#!/usr/bin/env python3
"""
gerar_tabelas.py
================
Gera, a partir dos CSVs brutos e das tabelas de analise confirmatoria, TODAS as
tabelas do manuscrito (arquivos tab_*.tex) e o arquivo numeros.tex com os
numeros citados no texto como macros (\\num{chave}). Assim nenhum numero do
manuscrito e digitado a mao: se os dados mudam, rodar este script e
recompilar atualiza tudo. A verificacao independente
(estudo_aos/verificacao_independente.py) confere o conteudo dos arquivos
gerados contra um recalculo feito com a biblioteca padrao.

Uso: python gerar_tabelas.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent.parent / "results" / "aos_study"
NUM = {}          # chave -> string formatada (vira \num{chave})


def f2(x):
    return f"{x:.2f}"


def sgn(x, d=2):
    """diferenca com sinal tipografico: +0.06 / $-$0.02"""
    s = f"{abs(x):.{d}f}"
    if float(s) == 0.0:
        return s                                   # evita "-0.00"
    return ("$+$" + s) if x >= 0 else ("$-$" + s)


def padj(p):
    if p < 1e-4:
        return r"\ensuremath{<10^{-4}}"
    if p >= 0.9995:
        return "1.0"
    if p < 1e-3:
        return f"{p:.4f}"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def plano(x, d=2):
    """numero com sinal de menos tipografico, sem '+': -0.02 -> $-$0.02; 0.004 -> 0.00"""
    s = f"{abs(x):.{d}f}"
    return s if (x >= 0 or float(s) == 0.0) else "$-$" + s


def macros_contraste(pref, chave, r):
    """macros de um contraste: d (com sinal), a (modulo), ci (IC95%), p (Holm), w/l (vitorias/derrotas)"""
    NUM[f"{pref}d_{chave}"] = sgn(r.dif_pp)
    NUM[f"{pref}a_{chave}"] = f"{abs(r.dif_pp):.2f}"
    NUM[f"{pref}ci_{chave}"] = f"[{plano(r.ic_lo)}, {plano(r.ic_hi)}]"
    NUM[f"{pref}p_{chave}"] = padj(r.p_holm)
    NUM[f"{pref}w_{chave}"] = str(int(r.vit_a))
    NUM[f"{pref}l_{chave}"] = str(int(r.der_a))


def tex_escape_label(s):
    return s.replace("_", r"\_")


def cel(row, d=2):
    """'diff [lo, hi]; p' para uma linha de contraste"""
    return f"{sgn(row.dif_pp, d)} [{sgn(row.ic_lo, d)}, {sgn(row.ic_hi, d)}]; {padj(row.p_holm)}"


def media_arm(df, col_arm, arm, col_nivel, nivel):
    sub = df[(df[col_arm] == arm) & (df[col_nivel] == nivel) & df.viavel]
    return sub.groupby("instancia").gap_pct.mean().mean()


def escreve(nome, texto):
    (AQUI / nome).write_text(texto, encoding="utf-8")


def tabela(caption, label, cab, linhas, colspec, small=True, extra_inicio=""):
    t = [r"\begin{table}[t]", r"\caption{" + caption + "}\\label{" + label + "}", r"\centering"]
    if small:
        t.append(r"\small")
    t += [r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{" + colspec + "}", r"\toprule", cab + r" \\", r"\midrule"]
    t += [l if (l.endswith("\\addlinespace") or l.startswith("\\midrule")) else l + r" \\" for l in linhas]
    t += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(t) + "\n"


def uso_pct(df, col, arm, nivel, col_nivel="nivel"):
    """% do total por operador (soma sobre execucoes) de uma coluna pipe-separada."""
    g = df[(df.arm == arm) & (df[col_nivel] == nivel)]
    arr = np.array([[float(v) for v in s.split("|")] for s in g[col]])
    return 100 * arr.sum(0) / arr.sum()


def extras(e1, e2, e3):
    from scipy import stats
    # participacao do 2-Opt (operador 0) na ordem fixa (GVNS classico)
    for B in (800, 4000):
        NUM[f"s2_2opt_share_fixed_{B}"] = f"{uso_pct(e1, 'uso_operadores', 'gvns_classico', B)[0]:.0f}"
    # uso de operadores do HG-UCB no VND de 5 operadores (Estudo 2B)
    u = uso_pct(e2, "uso_operadores", "s2b_hgucb_norm", 4000)
    NUM["s2_hg_use_min"], NUM["s2_hg_use_max"] = f"{u.min():.0f}", f"{u.max():.0f}"
    # fracao das avaliacoes de distancia no SWAP* (indice 9), Estudo 3, E2
    sw = {}
    for a in e3.arm.unique():
        if a.startswith("s3_P5") or a == "s3_R5_hgucb_cego":
            continue
        sw[a] = uso_pct(e3, "calls_por_operador", a, 5)[9]
    nao_foc = [v for k, v in sw.items() if "FOC" not in k]
    foc = [v for k, v in sw.items() if "FOC" in k]
    NUM["s3_swap_min"], NUM["s3_swap_max"] = f"{min(nao_foc):.0f}", f"{max(nao_foc):.0f}"
    NUM["s3_swap_foc_min"], NUM["s3_swap_foc_max"] = f"{min(foc):.0f}", f"{max(foc):.0f}"
    NUM["s3_swap_hg_lo"] = f"{sw['s3_LO_hgucbc']:.0f}"
    cross = [uso_pct(e3, "calls_por_operador", a, 5)[8] for a in sw if "FOC" not in a]
    sw11 = [uso_pct(e3, "calls_por_operador", a, 5)[2] for a in sw if "FOC" not in a]
    NUM["s3_cross_min"], NUM["s3_cross_max"] = f"{min(cross):.0f}", f"{max(cross):.0f}"
    NUM["s3_swap11_min"], NUM["s3_swap11_max"] = f"{min(sw11):.0f}", f"{max(sw11):.0f}"
    # subconjunto exploratorio de B=20000 (22 instancias, seeds 1000-1004; Entradas 69 e 72)
    sub = pd.read_csv(RES / "exploratorio_b20000.csv")
    inst22 = sorted(sub.instancia.unique())
    NUM["sub_n"] = str(len(inst22))

    def pm(df, col, arm, nivel):
        d = df[(df[col] == arm) & (df.B == nivel) & df.instancia.isin(inst22) & (df.seed < 1005) & df.viavel]
        return d.groupby("instancia").gap_pct.mean()

    for B, src, col in ((800, e1, "arm"), (4000, e1, "arm"), (20000, sub.rename(columns={"politica": "arm"}), "arm")):
        h, u_, g = pm(src, col, "hgucb_norm", B), pm(src, col, "uniforme", B), pm(src, col, "gvns_classico", B)
        dif = (h - u_).values
        NUM[f"sub_d_hu_{B}"] = sgn(dif.mean())
        rng = np.random.default_rng(0)
        bm = dif[rng.integers(0, len(dif), size=(10000, len(dif)))].mean(axis=1)
        NUM[f"sub_ci_hu_{B}"] = f"[{plano(np.percentile(bm, 2.5))}, {plano(np.percentile(bm, 97.5))}]"
        NUM[f"sub_d_hg_{B}"] = f"{abs((h - g).mean()):.2f}"
        NUM[f"sub_gap_gvns_{B}"] = f2(g.mean())


OPS = ["2-Opt", "Move(1)", "Swap(1,1)", "Reloc(1,0)", "Reloc(2,0)", "2-opt*", "Or-opt(2-3)", "Swap-intra", "CROSS", "SWAP*"]


def estudo4(e3):
    c4 = pd.read_csv(RES / "analise_etapa4_confirmatoria.csv")
    esc = json.loads((RES / "etapa4_tuning_escolhidas.json").read_text())
    e4 = pd.read_csv(RES / "etapa4_eval74.csv")
    grid = pd.read_csv(RES / "etapa4_tuning_grid.csv")
    stt = pd.read_csv(RES / "etapa4_tuning_static.csv")
    # piso de ruido do ajuste: mesma distribuicao (uniforme) com dois consumos de numeros aleatorios
    NUM["s4_tune_U"] = f2(grid[grid.braco == "tune_U"].gap_pct.mean())
    g0 = stt[stt.config_id == "gen00"].gap_pct.mean()
    NUM["s4_tune_static0"] = f2(g0)
    NUM["s4_tune_noise"] = f2(abs(grid[grid.braco == "tune_U"].gap_pct.mean() - g0))
    NUM["s4_tune_static_best"] = f2(esc["static"]["gap_tune"])
    NUM["s4_tune_static_gain"] = f2(grid[grid.braco == "tune_U"].gap_pct.mean() - esc["static"]["gap_tune"])
    pesos = esc["static"]["params"]["pesos"]
    for j, p in enumerate(pesos):
        NUM[f"s4_w{j}"] = f"{p:.2f}"
        NUM[f"s4_wp{j}"] = f"{100 * p:.1f}"
    NUM["s4_wp_2opt_pair"] = f"{100 * (pesos[0] + pesos[5]):.0f}"
    NUM["s4_wp_expensive"] = f"{100 * (pesos[2] + pesos[8] + pesos[9]):.1f}"
    lin = [f"{OPS[j]} & {pesos[j]:.3f}" for j in range(10)]
    escreve("tab_s4_pesos.tex", tabela("Study 4: probabilities of the tuned static mixture (tuning set).", "tab:s4pesos",
                                       "Operator & Probability", lin, "lc"))
    # tabela de configuracoes e desempenho
    nome_pol = {"ucb1": ("UCB1", "s4_R5_ucb1t", "s3_R5_ucb1c"), "pm": ("Prob. matching", "s4_R5_pmt", "s3_R5_pmc"),
                "hgucb": ("HG-UCB", "s4_R5_hgucbt", "s3_R5_hgucbc")}
    linhas = []
    for fam, (nm, t_arm, d_arm) in nome_pol.items():
        par = esc[fam]["params"]
        ps = ", ".join(f"{k}={v:g}" for k, v in par.items()).replace("gamma", r"$\gamma$").replace("alpha", r"$\alpha$").replace("p_min", r"$p_{\min}$")
        ps = ps.replace("C=", "$C$=")
        v = []
        for nv in (1, 5):
            t_ = media_arm(e4.rename(columns={"braco": "arm"}), "arm", t_arm, "nivel", nv)
            d_ = media_arm(e3, "arm", d_arm, "nivel", nv)
            v.append((t_, d_))
            NUM[f"s4_gap_{fam}_t_{nv}"], NUM[f"s4_gap_{fam}_d_{nv}"] = f2(t_), f2(d_)
        linhas.append(f"{nm} & {ps} & {esc[fam]['gap_tune']:.2f} / {esc[fam]['gap_padrao']:.2f} & "
                      f"{v[0][0]:.2f} / {v[0][1]:.2f} & {v[1][0]:.2f} / {v[1][1]:.2f}")
        NUM[f"s4_tune_{fam}_t"], NUM[f"s4_tune_{fam}_d"] = f2(esc[fam]["gap_tune"]), f2(esc[fam]["gap_padrao"])
    stat = [media_arm(e4.rename(columns={"braco": "arm"}), "arm", "s4_R5_static", "nivel", nv) for nv in (1, 5)]
    uu = [media_arm(e3, "arm", "s3_R5_U", "nivel", nv) for nv in (1, 5)]
    for nv, a, b in ((1, stat[0], uu[0]), (5, stat[1], uu[1])):
        NUM[f"s4_gap_static_{nv}"], NUM[f"s4_gap_U_{nv}"] = f2(a), f2(b)
    linhas.append(f"Static mixture & Table~\\stab{{tab:s4pesos}} & {NUM['s4_tune_static_best']} / {NUM['s4_tune_U']} & "
                  f"{stat[0]:.2f} / {uu[0]:.2f} & {stat[1]:.2f} / {uu[1]:.2f}")
    escreve("tab_s4_config.tex", tabela(
        "Study 4: parameters selected on the tuning set and mean gap (\\%) as ``tuned / reference'': on the tuning set the reference "
        "is the default (static mixture: uniform); on the 74 CVRPLIB instances it is the default policy (static: uniform selection).",
        "tab:s4config", r"Policy & Selected & Tuning set & CVRPLIB $E_1$ & CVRPLIB $E_2$", linhas, "lllcc"))
    rot = {"V1": "UCB1 (tuned) $-$ uniform", "V2": "Prob. matching (tuned) $-$ uniform", "V3": "HG-UCB (tuned) $-$ uniform",
           "V4": "Static mixture $-$ uniform", "V5": "UCB1 (tuned) $-$ static mixture", "V6": "Prob. matching (tuned) $-$ static mixture",
           "V7": "HG-UCB (tuned) $-$ static mixture"}
    lin = []
    for cid, nome in rot.items():
        r1 = c4[(c4.id == cid) & (c4.nivel == 1.0)].iloc[0]
        r5 = c4[(c4.id == cid) & (c4.nivel == 5.0)].iloc[0]
        lin.append(f"{nome} & {cel(r1)} & {cel(r5)}")
        macros_contraste("s4", f"{cid}_1", r1)
        macros_contraste("s4", f"{cid}_5", r5)
    escreve("tab_s4_tests.tex", tabela(
        "Study 4, 74 CVRPLIB instances: confirmatory contrasts (mean paired difference in pp [95\\% CI]; Holm-adjusted $p$ over 14 tests).",
        "tab:s4tests", r"Contrast & $E_1$ & $E_2$", lin, "lcc"))
    rotw = {"W1": "Ten vs five operators (LO, uniform)", "W2": "R5: fixed order by cost $-$ uniform",
            "W3": "UCB1 (tuned) $-$ uniform", "W4": "Prob. matching (tuned) $-$ uniform", "W5": "HG-UCB (tuned) $-$ uniform",
            "W6": "Static mixture $-$ uniform"}
    lin = []
    for cid, nome in rotw.items():
        r = c4[c4.id == cid].iloc[0]
        lin.append(f"{nome} & {cel(r)}")
        macros_contraste("s4", cid, r)
    ev = pd.read_csv(RES / "etapa4_val.csv").rename(columns={"braco": "arm"})
    ev = ev[ev.viavel]
    tv = ev.groupby(["arm", "instancia"]).gap_pct.mean().groupby("arm").mean()
    for a, v in tv.items():
        NUM[f"s4v_gap_{a}"] = f2(v)
    escreve("tab_s4_val.tex", tabela(
        "Study 4, 30 synthetic validation instances at $E_1$: confirmatory contrasts (mean paired difference in pp [95\\% CI]; "
        "Holm-adjusted $p$ over 6 tests).", "tab:s4val", r"Contrast & $E_1$", lin, "lc"))
    NUM["s4_nsig74"] = str(int(c4[c4.nivel.notna()].signif_005.sum()))
    NUM["s4_nsigval"] = str(int(c4[c4.nivel.isna()].signif_005.sum()))
    # eficiencia por operador (U instrumentado, R5, E1, 74 instancias)
    ins = e4[e4.braco == "s4_R5_U_instr"]
    suc = np.array([[int(x) for x in s.split("|")] for s in ins.suc_por_operador]).sum(0)
    uso = np.array([[int(x) for x in s.split("|")] for s in ins.uso_operadores]).sum(0)
    cal = np.array([[int(x) for x in s.split("|")] for s in ins.calls_por_operador]).sum(0)
    gan = np.array([[float(x) for x in s.split("|")] for s in ins.ganho_por_operador]).sum(0)
    lin = []
    for j in np.argsort(-(gan / cal)):
        lin.append(f"{OPS[j]} & {100 * suc[j] / uso[j]:.1f} & {100 * gan[j] / max(suc[j], 1):.2f} & {cal[j] / uso[j] / 1000:.1f} & "
                   f"{1000 * gan[j] / cal[j] * 100:.3f}")
        NUM[f"eff_{j}"] = f"{1000 * gan[j] / cal[j] * 100:.3f}"
        NUM[f"eff_suc_{j}"] = f"{100 * suc[j] / uso[j]:.1f}"
        NUM[f"eff_cost_{j}"] = f"{cal[j] / uso[j] / 1000:.1f}"
    escreve("tab_eficiencia.tex", tabela(
        "Efficiency of each operator under uniform selection (R5, ten operators, $E_1$, 74 instances, 10 seeds): success rate per "
        "application, mean relative gain per success, mean cost per application (thousand distance evaluations) and relative gain "
        "(\\%) per thousand distance evaluations, sorted by the last column.", "tab:eff",
        r"Operator & Success (\%) & Gain/success (\%) & Cost (k eval.) & Gain per 1k eval. (\%)", lin, "lcccc"))
    NUM["eff_best"], NUM["eff_worst"] = OPS[int(np.argmax(gan / cal))], OPS[int(np.argmin(gan / cal))]
    NUM["eff_cost_ratio"] = f"{(cal / uso).max() / (cal / uso).min():.0f}"
    NUM["eff_ratio"] = f"{(gan / cal).max() / (gan / cal).min():.0f}"


def estudo5():
    c5 = pd.read_csv(RES / "analise_etapa5_confirmatoria.csv")
    rot = {"A1": "Add 2-opt*, Or-opt, Swap-intra to five operators", "A2": "Add CROSS, SWAP* to five operators",
           "A3": "Add CROSS, SWAP* to the eight-operator set", "A4": "Ten operators minus Swap(1,1), CROSS, SWAP*",
           "A5": "Seven-operator cheap set minus eight-operator set (drop Swap(1,1))"}
    lin = []
    for cid, nome in rot.items():
        cel_ = []
        for est in ("LO", "R5"):
            for nv in (1, 5):
                r = c5[(c5.id == cid) & (c5.estrutura == est) & (c5.nivel == nv)].iloc[0]
                cel_.append(cel(r))
                macros_contraste("s5", f"{cid}_{est}_{nv}", r)
        lin.append(f"{nome} & " + " & ".join(cel_))
    escreve("tab_s5_tests.tex", tabela(
        "Study 5: attribution of the operator-pool effect (uniform selection; mean paired difference in pp [95\\% CI]; "
        "Holm-adjusted $p$ over 20 tests).", "tab:s5tests", r"Contrast & LO, $E_1$ & LO, $E_2$ & R5, $E_1$ & R5, $E_2$",
        lin, "lcccc", small=True))
    e5 = pd.read_csv(RES / "etapa5_full.csv").rename(columns={"braco": "arm"})
    e3 = pd.read_csv(RES / "etapa3_full.csv").rename(columns={"braco": "arm"})
    d = pd.concat([e5, e3[e3.arm.isin(["s3_P5_LO_U", "s3_LO_U", "s3_R5_U"])]])
    pools = [("P5", "Five original operators", "s3_P5_LO_U", "s5_R5_P5"),
             ("P8c", "P5 + 2-opt*, Or-opt, Swap-intra", "s5_LO_P8c", "s5_R5_P8c"),
             ("P7e", "P5 + CROSS, SWAP*", "s5_LO_P7e", "s5_R5_P7e"),
             ("P7c", "P10 without Swap(1,1), CROSS, SWAP*", "s5_LO_P7c", "s5_R5_P7c"),
             ("P10", "Ten operators", "s3_LO_U", "s3_R5_U")]
    lin = []
    for sigla, nome, a_lo, a_r5 in pools:
        cel_ = []
        for est, a in (("LO", a_lo), ("R5", a_r5)):
            for nv in (1, 5):
                v = media_arm(d, "arm", a, "nivel", nv)
                NUM[f"s5_gap_{a}_{nv}"] = f2(v)
                cel_.append(f2(v))
        lin.append(f"{nome} ({sigla}) & " + " & ".join(cel_))
    escreve("tab_s5.tex", tabela(
        "Study 5: mean gap (\\%) of each operator set under uniform selection, by stopping rule and budget.", "tab:s5",
        r"Operator set & LO, $E_1$ & LO, $E_2$ & R5, $E_1$ & R5, $E_2$", lin, "lcccc"))
    # extras: parte do efeito do conjunto (P5 -> P10) recuperada por P8c e P7e; P10 - P7e (descritivo); uso de SWAP*/CROSS
    for est, p5, p10 in (("LO", "s3_P5_LO_U", "s3_LO_U"), ("R5", "s5_R5_P5", "s3_R5_U")):
        for nv in (1, 5):
            g = lambda a: media_arm(d, "arm", a, "nivel", nv)
            tot = g(p5) - g(p10)
            for sig in ("P8c", "P7e", "P7c"):
                NUM[f"s5_rec_{sig}_{est}_{nv}"] = f"{100 * (g(p5) - g(f's5_{est}_{sig}')) / tot:.0f}"
            NUM[f"s5_P10minusP7e_{est}_{nv}"] = sgn(g(p10) - g(f"s5_{est}_P7e"))
    faixa_int([f"s5_rec_P7e_{e}_{n}" for e in ("LO", "R5") for n in (1, 5)], "s5_rec_P7e_rng")
    faixa_int([f"s5_rec_P8c_{e}_{n}" for e in ("LO", "R5") for n in (1, 5)], "s5_rec_P8c_rng")
    for nv in (1, 5):
        p = uso_pct(e5, "calls_por_operador", "s5_R5_P7e", nv)      # pool [0,1,2,3,4,8,9]: CROSS = 5, SWAP* = 6
        NUM[f"s5_p7e_swap_{nv}"], NUM[f"s5_p7e_cross_{nv}"] = f"{p[6]:.0f}", f"{p[5]:.0f}"
    e4 = pd.read_csv(RES / "etapa4_eval74.csv").rename(columns={"braco": "arm"})
    for nv in (1, 5):
        NUM[f"s4_static_swap_{nv}"] = f"{uso_pct(e4, 'calls_por_operador', 's4_R5_static', nv)[9]:.0f}"
        NUM[f"s3_U_swap_{nv}"] = f"{uso_pct(e3, 'calls_por_operador', 's3_R5_U', nv)[9]:.0f}"
        NUM[f"s3_U_2opt_{nv}"] = f"{uso_pct(e3, 'calls_por_operador', 's3_R5_U', nv)[0]:.0f}"
    NUM["s5_nsig"] = str(int(c5.signif_005.sum()))
    for cid in rot:
        NUM[f"s5_{cid}_nsig"] = str(int(c5[c5.id == cid].signif_005.sum()))


def faixa(chaves, nome):
    v = [float(NUM[k].replace("$-$", "-").replace("$+$", "")) for k in chaves]
    lo, hi = min(v), max(v)
    NUM[nome] = f"{lo:.2f}" if f"{lo:.2f}" == f"{hi:.2f}" else f"{lo:.2f}--{hi:.2f}"


def faixa_int(chaves, nome):
    v = [int(NUM[k]) for k in chaves]
    NUM[nome] = f"{min(v)}" if min(v) == max(v) else f"{min(v)}--{max(v)}"


def faixa1(chaves, nome):
    """faixa com uma casa decimal (resumo): '0.3--0.5' ou '0.5' quando os extremos coincidem"""
    v = [float(NUM[k].replace("$-$", "-").replace("$+$", "")) for k in chaves]
    lo, hi = f"{min(v):.1f}", f"{max(v):.1f}"
    NUM[nome] = lo if lo == hi else f"{lo}--{hi}"


def agrupamentos():
    """faixas de contrastes agrupados (para frases do texto)"""
    B = (800, 4000)
    k_ord = [f"s2a_C{c}_{b}" for c in (5, 6) for b in B]
    k_pool = ["s3a_T1_1", "s3a_T1_5"]
    k_term = [f"s2a_C{c}_{b}" for c in (1, 2) for b in B]
    faixa1(k_ord, "g_ord_r1")
    faixa1(k_pool, "g_pool_r1")
    faixa1(["s1a_hgucb_norm_gvns_classico_800", "s1a_hgucb_norm_gvns_classico_4000"], "g_gvns_r1")
    k_pool_syn = ["s4a_W1"] if "s4a_W1" in NUM else []   # efeito do conjunto nas 30 instancias sinteticas de validacao
    v = [float(NUM[k].replace("$-$", "-").replace("$+$", "")) for k in k_ord + k_pool + k_term + k_pool_syn]
    NUM["g_struct_lo"], NUM["g_struct_hi"] = f"{min(v):.2f}", f"{max(v):.2f}"
    # razao em relacao a 0,14% (meta-analise de Turkes et al., 2021)
    NUM["g_struct_x_lo"], NUM["g_struct_x_hi"] = f"{min(v) / 0.14:.0f}", f"{max(v) / 0.14:.0f}"
    faixa([f"s2a_C{c}_{b}" for c in (1, 2) for b in B], "g_term_abs")
    faixa_int([f"s2w_C{c}_{b}" for c in (1, 2) for b in B], "g_term_w")
    faixa([f"s2a_C{c}_{b}" for c in (3, 4) for b in B], "g_acc_abs")
    faixa([f"s2a_C5_{b}" for b in B], "g_ord_best_abs")
    faixa([f"s2a_C6_{b}" for b in B], "g_ord_shaken_abs")
    faixa([f"s2a_D{c}_{b}" for c in (5, 6, 7, 8) for b in B], "g_vsfixed_abs")
    faixa([f"s2a_D{c}_{b}" for c in (5, 7) for b in B], "g_vsfixed_ucbpm_abs")
    faixa([f"s3a_{t}_{x}_5" for t in ("LO", "R5") for x in ("ucb1cvsFOC", "pmcvsFOC")], "g_s3_vsfoc_abs")
    if "s5a_A1_LO_1" in NUM:
        for cid in ("A1", "A2", "A3", "A4", "A5"):
            faixa([f"s5a_{cid}_{e}_{n}" for e in ("LO", "R5") for n in (1, 5)], f"g_s5_{cid}")


def robustez_macros():
    r = pd.read_csv(RES / "analise_robustez.csv")
    ratio = (r.ic95_2n_hi - r.ic95_2n_lo) / (r.ic95_hi - r.ic95_lo)
    NUM["rb_wider_min"], NUM["rb_wider_max"] = f"{100 * (ratio.min() - 1):.0f}", f"{100 * (ratio.max() - 1):.0f}"
    NUM["rb_wider_med"] = f"{100 * (ratio.median() - 1):.0f}"
    for _, x in r.iterrows():
        k = f"{x.estudo}_{int(x.nivel)}_{x.a}_{x.b}"
        NUM[f"rb2_{k}"] = f"[{plano(x.ic95_2n_lo)}, {plano(x.ic95_2n_hi)}]"
        NUM[f"rb90_{k}"] = f"[{plano(x.ic90_lo)}, {plano(x.ic90_hi)}]"
        NUM[f"rbd_{k}"] = sgn(x.dif_pp)
        NUM[f"rb1_{k}"] = f"[{plano(x.ic95_lo)}, {plano(x.ic95_hi)}]"


PUB_VNS = {"A": 2.040, "B": 1.832, "X": 6.291}          # Kalatzantonakis et al. (2023), Tabelas 10-12
PUB_HGUCB = {"A": 1.610, "B": 1.655, "X": 6.023}
REPRO = [("Ours: classical GVNS", "gvns_baseline_{s}.csv", "gvns"),
         (r"Ours: HG-UCB literal, fixed $x'$", "bandit_vns_validation_{s}.csv", "lit_fixo"),
         (r"Ours: HG-UCB literal, progressive $x'$", "bandit_vns_validation_progressivo_{s}.csv", "lit_prog"),
         (r"Ours: HG-UCB normalized, $\alpha=5$", "bandit_vns_validation_progressivo_normalizado_alpha5.0_{s}.csv", "n5"),
         (r"Ours: HG-UCB normalized, $\alpha=0.5$", "bandit_vns_validation_progressivo_normalizado_alpha0.5_{s}.csv", "n05")]


def reproducao():
    """Tabela de fidelidade da reproducao (Secao 3.5) a partir dos CSVs brutos (A e B reexecutados com solucao inicial fixada)."""
    R = RES.parent
    n_esperado = {"A": 26, "B": 23, "X": 21}
    medias = {}
    for rot, arq, chave in REPRO:
        medias[chave] = {}
        for s in "ABX":
            d = pd.read_csv(R / arq.format(s=s))
            assert d.instancia.nunique() == n_esperado[s] and d.groupby("instancia").seed.nunique().min() == 5, (arq, s)
            medias[chave][s] = d.groupby("instancia").erro_pct.mean().mean()
    mm = lambda v: np.mean([v["A"], v["B"], v["X"]])          # media simples dos tres conjuntos (como no artigo)
    lin = []
    pub = [("Published VNS", PUB_VNS), ("Published HG-UCB", PUB_HGUCB)]
    for rot, v in pub:
        lin.append(f"{rot} & {v['A']:.3f} & {v['B']:.3f} & {v['X']:.3f} & {mm(v):.3f}")
    lin.append(r"\addlinespace")
    for rot, arq, chave in REPRO:
        v = medias[chave]
        lin.append(f"{rot} & {v['A']:.3f} & {v['B']:.3f} & {v['X']:.3f} & {mm(v):.3f}")
        for s in "ABX":
            NUM[f"rep_{chave}_{s}"] = f"{v[s]:.2f}"
        NUM[f"rep_{chave}_m"] = f"{mm(v):.2f}"
    lin.append(r"\midrule")
    d_pub = {s: PUB_HGUCB[s] - PUB_VNS[s] for s in "ABX"}
    lin.append("Published: HG-UCB $-$ VNS & " + " & ".join(sgn(d_pub[s], 3) for s in "ABX") + f" & {sgn(mm(PUB_HGUCB) - mm(PUB_VNS), 3)}")
    for rot, chave in ((r"Ours: literal (progressive) $-$ GVNS", "lit_prog"), (r"Ours: normalized $\alpha=0.5$ $-$ GVNS", "n05")):
        d = {s: medias[chave][s] - medias["gvns"][s] for s in "ABX"}
        lin.append(f"{rot} & " + " & ".join(sgn(d[s], 3) for s in "ABX") + f" & {sgn(mm(medias[chave]) - mm(medias['gvns']), 3)}")
        for s in "ABX":
            NUM[f"rep_d_{chave}_{s}"] = sgn(d[s])
        NUM[f"rep_d_{chave}_m"] = sgn(mm(medias[chave]) - mm(medias["gvns"]))
        NUM[f"rep_da_{chave}_m"] = f"{abs(mm(medias[chave]) - mm(medias['gvns'])):.2f}"
    for s in "ABX":
        NUM[f"rep_gvns_vs_pub_{s}"] = sgn(medias["gvns"][s] - PUB_VNS[s])
    NUM["rep_pub_m"] = f"{mm(PUB_VNS):.2f}"
    NUM["rep_pub_diff"] = sgn(mm(PUB_HGUCB) - mm(PUB_VNS))
    NUM["rep_pub_diffa"] = f"{abs(mm(PUB_HGUCB) - mm(PUB_VNS)):.2f}"
    escreve("tab_repro.tex", tabela(
        "Reproduction fidelity: mean error (\\%) over the instances of each set (mean of per-instance means over five seeds). "
        "Published values from the original study \\citep{kalatzantonakis2023}. ``Literal'' uses the raw-cost reward and "
        "$\\alpha=5$; ``normalized'' uses the dimensionless reward. Last column: unweighted mean of the three sets, as in the original.",
        "tab:repro", " & A & B & X & Mean of sets", lin, "lcccc"))


SUP_ORDEM = ["tab:s1tests", "tab:s2tests", "tab:s3tests", "tab:s4tests", "tab:s4val", "tab:s5tests", "tab:s6tests",
             "tab:s7tests", "tab:s8tests", "tab:s8rec", "tab:s9tests", "tab:s10tests", "tab:s12tests", "tab:s2set", "tab:s3set", "tab:ranks", "tab:s4config", "tab:s4pesos", "tab:s5", "tab:hip",
             "tab:studies", "tab:hyper"]      # mesma ordem de supplement.tex


def suptab():
    """numeracao S<k> das tabelas do suplemento online, para o manuscrito citar 'Table S<k>' (\\stab{label})"""
    l = ["% GERADO por gerar_tabelas.py -- NAO EDITAR A MAO", r"\makeatletter"]
    ordem = [lab for lab in SUP_ORDEM if lab != "tab:s9tests" or (RES / "analise_etapa9_confirmatoria.csv").exists()]
    for k, lab in enumerate(ordem, 1):
        l.append(r"\expandafter\gdef\csname sup@" + lab + r"\endcsname{S" + str(k) + "}")
    l += [r"\makeatother",
          r"\newcommand{\stab}[1]{\ifcsname sup@#1\endcsname\csname sup@#1\endcsname\else\textcolor{red}{??\detokenize{#1}??}\fi}"]
    escreve("suptab.tex", "\n".join(l) + "\n")


ARMS6 = [("s6_R5_exp3", "EXP3"), ("s6_R5_ts", "Thompson sampling"), ("s6_R5_swucb", "Sliding-window UCB"),
         ("s6_R5_ap", "Adaptive pursuit"), ("s6_R5_eff", "Efficiency UCB ($C=1$)"), ("s6_R5_eff2", "Efficiency UCB ($C=2$)"),
         ("s6_R5_ctxts", "Contextual Thompson sampling"), ("s6_R5_ctxeff", "Contextual Thompson, gain per evaluation")]


def estudo6():
    c6 = pd.read_csv(RES / "analise_etapa6_confirmatoria.csv")
    conf = c6[c6.grupo == "confirmatorio"]
    e6 = pd.read_csv(RES / "etapa6_full.csv").rename(columns={"braco": "arm"})
    e3 = pd.read_csv(RES / "etapa3_full.csv").rename(columns={"braco": "arm"})
    lin_main, lin_sup = [], []
    u = [media_arm(e3, "arm", "s3_R5_U", "nivel", nv) for nv in (1, 5)]
    lin_main.append(f"Uniform selection (reference) & {u[0]:.2f} & & & {u[1]:.2f} & & ")
    for i, (arm, nome) in enumerate(ARMS6, 1):
        celulas, sup = [], []
        for nv in (1, 5):
            g = media_arm(e6, "arm", arm, "nivel", nv)
            r = conf[(conf.id == f"W{i}") & (conf.nivel == nv)].iloc[0]
            macros_contraste("s6", f"W{i}_{nv}", r)
            NUM[f"s6_gap_{arm}_{nv}"] = f2(g)
            celulas += [f"{g:.2f}", f"{sgn(r.dif_pp)} [{plano(r.ic_lo)}, {plano(r.ic_hi)}]", padj(r.p_holm)]
            sup.append(cel(r))
        lin_main.append(f"{nome} & " + " & ".join(celulas))
        lin_sup.append(f"{nome} $-$ uniform & {sup[0]} & {sup[1]}")
    escreve("tab_s6.tex", tabela(
        "Study 6: other policy families (R5 structure, ten operators, no tuning): mean gap (\\%), mean paired difference from "
        "uniform selection in pp [95\\% CI] and Holm-adjusted $p$ over 16 tests.", "tab:s6",
        r" & \multicolumn{3}{c}{$E_1$} & \multicolumn{3}{c}{$E_2$} \\ Policy & Gap & Difference & $p$ & Gap & Difference & $p$",
        lin_main, "lcccccc"))
    escreve("tab_s6_tests.tex", tabela(
        "Study 6: confirmatory contrasts against uniform selection (mean paired difference in pp [95\\% CI]; Holm-adjusted $p$ over 16 tests).",
        "tab:s6tests", r"Contrast & $E_1$ & $E_2$", lin_sup, "lcc"))
    NUM["s6_nsig"] = str(int(conf.signif_005.sum()))
    NUM["s6_nworse"] = str(int(((conf.signif_005) & (conf.dif_pp > 0)).sum()))
    NUM["s6_nbetter"] = str(int(((conf.signif_005) & (conf.dif_pp < 0)).sum()))
    # exploratorio: contra a mistura estatica ajustada; participacao de SWAP*/2-Opt na politica de eficiencia
    ex = c6[c6.grupo == "expl_vs_static"]
    for i, (arm, _) in enumerate(ARMS6, 1):
        for nv in (1, 5):
            r = ex[(ex.a == arm) & (ex.nivel == nv)].iloc[0]
            macros_contraste("s6x", f"{i}_{nv}", r)
    for nv in (1, 5):
        for arm, tag in (("s6_R5_eff", "eff"), ("s6_R5_eff2", "eff2")):
            p = uso_pct(e6, "calls_por_operador", arm, nv)
            NUM[f"s6_{tag}_swap_{nv}"], NUM[f"s6_{tag}_2opt_{nv}"] = f"{p[9]:.0f}", f"{p[0]:.0f}"
    # validacao sintetica (exploratoria, 30 instancias, E1, sem correcao de multiplicidade)
    vf = RES / "analise_etapa6_val.csv"
    if vf.exists():
        cv = pd.read_csv(vf)
        ev = pd.read_csv(RES / "etapa6_val.csv").rename(columns={"braco": "arm"})
        ev = ev[ev.viavel]
        tv = ev.groupby(["arm", "instancia"]).gap_pct.mean().groupby("arm").mean()
        for a, v in tv.items():
            NUM[f"s6v_gap_{a}"] = f2(v)
        for i in range(1, len(ARMS6) + 1):
            r = cv[cv.id == f"VAL{i}"].iloc[0]
            NUM[f"s6vd_{i}"] = sgn(r.dif_pp)
            NUM[f"s6vci_{i}"] = f"[{plano(r.ic_lo)}, {plano(r.ic_hi)}]"
            NUM[f"s6vp_{i}"] = padj(r.p_bruto)
        ph = np.array(cv.sort_values("p_bruto").p_bruto)
        adj = {}
        corr = 0.0
        for rank, (idx, pv) in enumerate(cv.sort_values("p_bruto").p_bruto.items()):
            corr = max(corr, min(1.0, (len(cv) - rank) * pv))
            adj[cv.loc[idx, "id"]] = corr
        for i in range(1, len(ARMS6) + 1):
            NUM[f"s6vph_{i}"] = padj(adj[f"VAL{i}"])
        NUM["s6v_nbetter_raw"] = str(int(((cv.dif_pp < 0) & (cv.p_bruto < 0.05)).sum()))
        NUM["s6v_nworse_raw"] = str(int(((cv.dif_pp > 0) & (cv.p_bruto < 0.05)).sum()))
    # melhor e pior braco (media) por nivel
    for nv in (1, 5):
        m = {a: media_arm(e6, "arm", a, "nivel", nv) for a, _ in ARMS6}
        NUM[f"s6_best_{nv}"] = min(m, key=m.get).replace("s6_R5_", "")
        NUM[f"s6_worst_{nv}"] = max(m, key=m.get).replace("s6_R5_", "")


ARMS7 = [("fixed", "Original HGS order"), ("unif", "Uniform random order"), ("ucb1", "UCB1"), ("ts", "Thompson sampling"),
         ("pm", "Probability matching"), ("hg", "HG-UCB (lite)")]


def estudo7():
    c7 = pd.read_csv(RES / "analise_etapa7_confirmatoria.csv")
    d = pd.read_csv(RES / "etapa7_full.csv")
    d["pop"] = np.where(d.conjunto == "SYN_VAL", "P2", "P1")
    dv = d[d.viavel]
    conf = c7[c7.grupo == "confirmatorio"]
    NUM["s7_n_p1"], NUM["s7_n_p2"] = str(d[d["pop"] == "P1"].instancia.nunique()), str(d[d["pop"] == "P2"].instancia.nunique())
    NUM["s7_nruns"] = f"{len(d):,}".replace(",", r"\,")
    NUM["s7_ninv"] = str(int((~d.viavel).sum()))
    # medias por braco, populacao e ponto de controle (media sobre instancias das medias por instancia)
    for pop in ("P1", "P2"):
        for ck in (250, 500, 1000, 2000, 4000):
            t = dv[dv["pop"] == pop].groupby(["braco", "instancia"])[f"gap_{ck}"].mean().groupby("braco").mean()
            for a, v in t.items():
                NUM[f"s7_gap_{a}_{pop}_{ck}"] = f"{v:.3f}"
    for pop in ("P1", "P2"):
        for ck in (250, 1000):
            for a, _ in ARMS7:
                if a == "unif":
                    continue
                r = conf[(conf.a == a) & (conf["pop"] == pop) & (conf.ponto == ck)].iloc[0]
                macros_contraste("s7", f"{a}_{pop}_{ck}", r)
                NUM[f"s7pr_{a}_{pop}_{ck}"] = f"{r.p_bruto:.2f}"   # p bruto (Wilcoxon), antes do Holm
                NUM[f"s7eq_{a}_{pop}_{ck}"] = "yes" if r.equiv_005 else "no"
    NUM["s7_nsig"] = str(int(conf.signif_005.sum()))
    NUM["s7_nbetter"] = str(int(((conf.dif_pp < 0) & conf.signif_005).sum()))
    NUM["s7_nworse"] = str(int(((conf.dif_pp > 0) & conf.signif_005).sum()))
    NUM["s7_neq"] = str(int(conf[conf.a.isin(["ucb1", "ts", "pm", "hg"])].equiv_005.sum()))
    tarde = c7[(c7.grupo == "exploratorio") & c7["pop"].isin(["P1", "P2"]) & c7.ponto.isin([2000, 4000]) & c7.a.isin(["ucb1", "ts", "pm", "hg"])]
    NUM["s7_late_n"], NUM["s7_late_eq"] = str(len(tarde)), str(int(tarde.equiv_005.sum()))
    NUM["s7_mde_med"] = f"{conf.mde_pp.median():.2f}"
    NUM["s7_maxabs"] = f"{conf.dif_pp.abs().max():.2f}"
    NUM["s7_maxabs_adapt"] = f"{conf[conf.a.isin(['ucb1', 'ts', 'pm', 'hg'])].dif_pp.abs().max():.2f}"
    sf = RES / "analise_etapa7_sensibilidade.csv"
    if sf.exists():
        sn = pd.read_csv(sf)
        NUM["s7_sens_nsig"] = str(int(sn.signif_005.sum()))
    # tabela principal: ganho medio (diferenca para a ordem aleatoria) por braco, populacao e ponto de controle
    lin = []
    for a, nome in ARMS7:
        cel_ = []
        for pop in ("P1", "P2"):
            for ck in (250, 1000):
                g = float(NUM[f"s7_gap_{a}_{pop}_{ck}"])
                if a == "unif":
                    cel_.append(f"{g:.2f}")
                else:
                    r = conf[(conf.a == a) & (conf["pop"] == pop) & (conf.ponto == ck)].iloc[0]
                    cel_.append(f"{g:.2f} ({sgn(r.dif_pp)}{'*' if r.signif_005 else ''})")
        lin.append(f"{nome} & " + " & ".join(cel_))
    escreve("tab_s7.tex", tabela(
        "Study 7 (HGS-CVRP): mean gap (\\%) after 250 and 1000 generations on the 25 CVRPLIB instances of sets X and CMT (P1) and on the "
        "30 synthetic validation instances (P2); in parentheses, the difference from the uniform random order in pp "
        "(* marks Holm-significance over 20 tests).", "tab:s7",
        r" & \multicolumn{2}{c}{P1} & \multicolumn{2}{c}{P2} \\ Move-order policy & 250 & 1000 & 250 & 1000", lin, "lcccc"))
    lin = []
    for a, nome in ARMS7:
        if a == "unif":
            continue
        cel_ = []
        for pop in ("P1", "P2"):
            for ck in (250, 1000):
                r = conf[(conf.a == a) & (conf["pop"] == pop) & (conf.ponto == ck)].iloc[0]
                cel_.append(cel(r))
        lin.append(f"{nome} $-$ uniform order & " + " & ".join(cel_))
    escreve("tab_s7_tests.tex", tabela(
        "Study 7 (HGS-CVRP): confirmatory contrasts against the uniform random order of the moves (mean paired difference in pp [95\\% CI]; "
        "Holm-adjusted $p$ over 20 tests).", "tab:s7tests",
        r"Contrast & P1, 250 & P1, 1000 & P2, 250 & P2, 1000", lin, "lcccc"))
    # tempo de CPU por geracao
    c = dv.groupby(["braco", "instancia"]).cpu_s.mean().unstack(0)
    for a, _ in ARMS7:
        NUM[f"s7_cpu_{a}"] = f"{1000 * c[a].mean() / 4000:.1f}"
        NUM[f"s7_cpurat_{a}"] = f"{(c[a] / c['fixed']).mean():.1f}"
    # tentativas e sucessos por tipo de movimento sob a ordem aleatoria
    g = d[(d.braco == "unif") & d.tentativas.notna()]
    ten = np.array([[float(x) for x in s.split("|")] for s in g.tentativas]).sum(0)
    suc = np.array([[float(x) for x in s.split("|")] for s in g.sucessos]).sum(0)
    taxa = suc / np.maximum(ten, 1)
    for j in range(9):
        NUM[f"s7_succ_{j + 1}"] = f"{1e4 * taxa[j]:.1f}"
    NUM["s7_succ_max"], NUM["s7_succ_min"] = str(int(np.argmax(taxa)) + 1), str(int(np.argmin(taxa)) + 1)
    NUM["s7_succ_ratio"] = f"{taxa.max() / taxa.min():.0f}"


ARMS8 = [("U", "Uniform selection"), ("ucb1c", "UCB1 (cost-aware)"), ("pmc", "Probability matching (cost-aware)"),
         ("hgucbc", "HG-UCB (cost-aware)"), ("ts", "Thompson sampling"), ("exp3", "EXP3"), ("eff", "Efficiency UCB")]
POOLS8 = [("P10", "Clean pool"), ("P10+D4x4", "4 useless, $4n^2$"), ("P10+D4x8", "4 useless, $8n^2$")]


def estudo8():
    """Estudo 8 (controle positivo): operadores inuteis plantados. Entrada 101/102 do diario."""
    c8 = pd.read_csv(RES / "analise_etapa8_confirmatoria.csv")
    c8["p_holm"] = c8.p_holm.fillna(c8.p_bruto)          # linhas exploratorias do pool limpo: p bruto
    rc = pd.read_csv(RES / "analise_etapa8_recuperacao.csv")
    d = pd.read_csv(RES / "etapa8_full.csv")
    d = d[d.viavel]
    gap = d.groupby(["braco", "instancia"]).gap_pct.mean().groupby("braco").mean()
    for pool, _ in POOLS8:
        for a, _ in ARMS8:
            NUM[f"s8_gap_{pool}_{a}"] = f2(gap[f"s8_{pool}_{a}"])
    for pool in ("P10+D4x4", "P10+D4x8"):
        NUM[f"s8_loss_{pool}"] = sgn(gap[f"s8_{pool}_U"] - gap["s8_P10_U"])
    # participacao das avaliacoes gastas nos operadores inuteis (indices >= 10)
    for pool in ("P10+D4x4", "P10+D4x8"):
        for a, _ in ARMS8:
            g = d[d.braco == f"s8_{pool}_{a}"]
            arr = np.array([[int(v) for v in s.split("|")] for s in g.calls_por_operador])
            NUM[f"s8_duds_{pool}_{a}"] = f"{100 * arr[:, 10:].sum() / arr.sum():.0f}"
    conf = c8[c8.grupo == "confirmatorio"]
    lin_main, lin_sup = [], []
    for a, nome in ARMS8:
        cel_ = []
        for pool, _ in POOLS8:
            g = float(NUM[f"s8_gap_{pool}_{a}"])
            if a == "U":
                cel_.append(f"{g:.2f}")
                continue
            r = c8[(c8.id == f"{pool}:{a}")].iloc[0]
            macros_contraste("s8", f"{pool}_{a}", r)
            NUM[f"s8pr_{pool}_{a}"] = f"{r.p_bruto:.3f}"
            star = "*" if (r.grupo == "confirmatorio" and r.signif_005) else ""
            cel_.append(f"{g:.2f} ({sgn(r.dif_pp)}{star})")
        lin_main.append(f"{nome} & " + " & ".join(cel_))
        if a != "U":
            sup = [cel(c8[c8.id == f"{pool}:{a}"].iloc[0]) for pool, _ in POOLS8]
            lin_sup.append(f"{nome} $-$ uniform & {sup[1]} & {sup[2]} & {sup[0]}")
    escreve("tab_s8.tex", tabela(
        "Study 8 (positive control): mean gap (\\%) over 74 instances at $E_1$ in the clean pool of ten operators and in pools with four "
        "planted useless operators (each costing $4n^2$ or $8n^2$ distance evaluations and never improving); in parentheses, the difference "
        "from uniform selection \\emph{in the same pool} in pp (* marks Holm-significance over the 12 confirmatory tests; "
        "clean-pool comparisons are exploratory).", "tab:s8",
        r"Policy & " + " & ".join(n for _, n in POOLS8), lin_main, "lccc"))
    escreve("tab_s8_tests.tex", tabela(
        "Study 8: contrasts against uniform selection in the same pool (mean paired difference in pp [95\\% CI]; Holm-adjusted $p$ over 12 "
        "tests for the two pools with useless operators; the clean-pool column is exploratory and reports unadjusted $p$).", "tab:s8tests",
        r"Contrast & 4 useless, $4n^2$ & 4 useless, $8n^2$ & Clean pool (exploratory)", lin_sup, "lccc"))
    NUM["s8_nsig"] = str(int(conf.signif_005.sum()))
    NUM["s8_nbetter"] = str(int(((conf.dif_pp < 0) & conf.signif_005).sum()))
    NUM["s8_nworse"] = str(int(((conf.dif_pp > 0) & conf.signif_005).sum()))
    # recuperacao da perda plantada
    lin = []
    for a, nome in ARMS8[1:]:
        cel_ = []
        for pool in ("P10+D4x4", "P10+D4x8"):
            r = rc[(rc.pool == pool) & (rc.politica == a)].iloc[0]
            NUM[f"s8rec_{pool}_{a}"] = f"{100 * r.recuperacao:.0f}"
            NUM[f"s8reci_{pool}_{a}"] = f"[{100 * r.rec_lo:.0f}, {100 * r.rec_hi:.0f}]"
            cel_.append(f"{100 * r.recuperacao:.0f}\\% [{100 * r.rec_lo:.0f}, {100 * r.rec_hi:.0f}]")
        lin.append(f"{nome} & " + " & ".join(cel_) + f" & {NUM[f's8_duds_P10+D4x8_{a}']}\\%")
    escreve("tab_s8_rec.tex", tabela(
        "Study 8: share of the planted loss recovered, $1-\\mathrm{loss}(\\text{policy})/\\mathrm{loss}(\\text{uniform})$ with "
        "$\\mathrm{loss}(x)$ the gap in the pool with useless operators minus the gap in the clean pool (exploratory; 95\\% bootstrap "
        "interval over instances), and share of the distance evaluations spent on useless operators in the $8n^2$ pool (uniform selection: "
        + NUM["s8_duds_P10+D4x8_U"] + "\\%).", "tab:s8rec",
        r"Policy & Recovery, $4n^2$ & Recovery, $8n^2$ & On useless operators ($8n^2$)", lin, "lccc"))
    rec8 = rc[rc.pool == "P10+D4x8"].recuperacao
    NUM["s8_rec_med_D4x8"] = f"{100 * rec8.median():.0f}"
    NUM["s8_rec_n40_D4x8"] = str(int((rec8 >= 0.4).sum()))
    NUM["s8_rec_excl1_D4x8"] = str(int((rc[rc.pool == "P10+D4x8"].rec_hi < 1).sum()))
    NUM["s8_rec_excl1_D4x4"] = str(int((rc[rc.pool == "P10+D4x4"].rec_hi < 1).sum()))
    best = conf[conf.id.str.startswith("P10+D4x8")].sort_values("dif_pp").iloc[0]
    NUM["s8_best_D4x8"] = best.id.split(":")[1]
    NUM["s8_best_D4x4_a"] = f"{abs(conf[conf.id.str.startswith('P10+D4x4')].dif_pp.min()):.2f}"
    for k in [k for k in NUM if k.startswith("s8") and "P10+" in k]:     # chaves sem '+' (nomes de macro seguros)
        NUM[k.replace("P10+", "")] = NUM.pop(k)


ARMS9 = [("U", "Uniform selection"), ("RP", "Adaptive weights (Ropke--Pisinger)"), ("ucb1", "UCB1"), ("pm", "Probability matching"),
         ("hgucb", "HG-UCB"), ("ts", "Thompson sampling"), ("exp3", "EXP3"), ("eff", "Efficiency UCB")]


def estudo9():
    """Estudo 9 (ALNS): selecao adaptativa com operadores estocasticos. Entradas 103/107 do diario."""
    c = pd.read_csv(RES / "analise_etapa9_confirmatoria.csv")
    e = pd.read_csv(RES / "etapa9_eval.csv")
    e = e[e.viavel]
    vp = RES / "etapa9_vns.csv"
    v = pd.read_csv(vp) if vp.exists() else None
    if v is not None:
        v = v[v.viavel]
    allr = pd.concat([e, v]) if v is not None else e
    allr["set"] = np.where(allr.instancia.str.contains("syn"), "V", "X")
    gm = allr.groupby(["braco", "set", "instancia"]).gap_pct.mean().reset_index()
    for br, g in gm.groupby("braco"):
        a = br.replace("s9_", "")
        NUM[f"s9_gap_{a}"] = f"{g.gap_pct.mean():.3f}"
        for s in ("X", "V"):
            NUM[f"s9{s.lower()}_gap_{a}"] = f"{g[g.set == s].gap_pct.mean():.3f}"
    NUM["s9_n_inst"] = str(e.instancia.nunique())
    NUM["s9_nruns"] = f"{len(e):,}".replace(",", r"\,")
    esc = json.loads((RES / "etapa9_tuning_escolhidas.json").read_text())
    for fam in ("RP", "ucb1", "pm", "hgucb"):
        NUM[f"s9_tune_{fam}"] = f"{esc[fam]['gap_tune']:.2f}"
        NUM[f"s9_tune_{fam}_padrao"] = f"{esc[fam]['gap_padrao']:.2f}"
    c["p_holm"] = c.p_holm.fillna(c.p_bruto)           # linhas primaria/exploratorias: p bruto
    c.loc[c.id == "PRIM:RP", "signif_005"] = c.loc[c.id == "PRIM:RP", "p_bruto"] < 0.05
    pr = c[c.id == "PRIM:RP"].iloc[0]
    NUM["s9prim_d"], NUM["s9prim_a"] = sgn(pr.dif_pp), f"{abs(pr.dif_pp):.2f}"
    NUM["s9prim_ci"] = f"[{plano(pr.ic_lo)}, {plano(pr.ic_hi)}]"
    NUM["s9prim_p"] = f"{pr.p_bruto:.3f}"
    NUM["s9prim_w"], NUM["s9prim_l"] = str(int(pr.vit_a)), str(int(pr.der_a))
    NUM["s9prim_n"] = str(int(pr.n_inst))
    lin_main, lin_sup = [], []
    for a, nome in ARMS9:
        cel_ = []
        for s in ("X", "V"):
            g = float(NUM[f"s9{s.lower()}_gap_{a}"])
            if a == "U":
                cel_.append(f"{g:.2f}")
                continue
            pref = "EXPL" if a == "RP" else "SEC"
            r = c[c.id == f"{pref}:{a}@{'X' if s == 'X' else 'SYN_VAL'}"].iloc[0]
            macros_contraste("s9", f"{a}_{s}", r)
            cel_.append(f"{g:.2f} ({sgn(r.dif_pp)}{'*' if (a != 'RP' and r.signif_005) else ''})")
        lin_main.append(f"{nome} & " + " & ".join(cel_))
        if a != "U":
            pref = "EXPL" if a == "RP" else "SEC"
            rx = c[c.id == f"{pref}:{a}@X"].iloc[0]
            rv = c[c.id == f"{pref}:{a}@SYN_VAL"].iloc[0]
            lin_sup.append(f"{nome} $-$ uniform & {cel(rx)} & {cel(rv)}")
    if "vns_U" in [k.replace("s9_gap_", "") for k in NUM if k.startswith("s9_gap_")]:
        for a, nome in (("vns_U", "VNS, uniform selection (same budget)"), ("vns_static", "VNS, static mixture (same budget)")):
            lin_main.append(f"{nome} & {float(NUM[f's9x_gap_{a}']):.2f} & {float(NUM[f's9v_gap_{a}']):.2f}")
        for ref in ("vns_U", "vns_static"):
            for s, tag in (("X", "X"), ("V", "SYN_VAL")):
                r = c[c.id == f"TERC:ALNS-U vs {ref}@{tag}"].iloc[0]
                macros_contraste("s9t", f"{ref}_{s}", r)
        lin_sup.append(r"\addlinespace")
        for ref, nome in (("vns_U", "ALNS uniform $-$ VNS uniform"), ("vns_static", "ALNS uniform $-$ VNS static mixture")):
            rx = c[c.id == f"TERC:ALNS-U vs {ref}@X"].iloc[0]
            rv = c[c.id == f"TERC:ALNS-U vs {ref}@SYN_VAL"].iloc[0]
            lin_sup.append(f"{nome} & {cel(rx)} & {cel(rv)}")
    escreve("tab_s9.tex", tabela(
        "Study 9 (ALNS with stochastic destroy and repair operators, $1500\\,n^2$ distance evaluations): mean gap (\\%) on the 21 instances of "
        "set X (ten seeds) and the 30 synthetic validation instances (five seeds); in parentheses, the difference from uniform selection in pp "
        "(* marks Holm-significance over the 12 secondary tests; for the adaptive weights, the primary test on the 51 instances is in the text and the per-set differences are exploratory). The last rows are VNS arms at the same budget.", "tab:s9",
        r"Policy & Set X & Synthetic validation", lin_main, "lcc"))
    escreve("tab_s9_tests.tex", tabela(
        "Study 9: secondary and tertiary contrasts (mean paired difference in pp [95\\% CI]; Holm-adjusted $p$ over 12 secondary and, separately, "
        "four tertiary tests).", "tab:s9tests", r"Contrast & Set X & Synthetic validation", lin_sup, "lcc"))
    sec = c[c.grupo == "secundario"]
    NUM["s9_nsig"] = str(int(sec.signif_005.sum()))
    NUM["s9_nbetter"] = str(int(((sec.dif_pp < 0) & sec.signif_005).sum()))
    NUM["s9_nworse"] = str(int(((sec.dif_pp > 0) & sec.signif_005).sum()))
    for pol in ("RP", "ucb1", "pm", "hgucb", "ts", "exp3", "eff", "U"):
        g = e[e.braco == f"s9_{pol}"]
        ur = np.array([[float(x) for x in s.split("|")] for s in g.uso_r])
        ud = np.array([[float(x) for x in s.split("|")] for s in g.uso_d])
        NUM[f"s9_reg2_{pol}"] = f"{100 * np.mean(ur[:, 1] / ur.sum(1)):.0f}"
        NUM[f"s9_rota_{pol}"] = f"{100 * np.mean(ud[:, 3] / ud.sum(1)):.0f}"
        NUM[f"s9_cpu_{pol}"] = f"{g.tempo_s.mean():.0f}"


ARMS12 = [("U", "Uniform selection"), ("static", "Static mixture (tuned offline)"), ("reg2", "Fixed repair: always regret-2"),
          ("RP", "Adaptive weights (Ropke--Pisinger)"), ("pm", "Probability matching (tuned)")]


def estudo12():
    """Estudo 12 (controle estatico no ALNS). Entradas 109/111 do diario."""
    c = pd.read_csv(RES / "analise_etapa12_confirmatoria.csv")
    cap = pd.read_csv(RES / "analise_etapa12_captura.csv")
    e12 = pd.read_csv(RES / "etapa12_eval.csv")
    e9 = pd.read_csv(RES / "etapa9_eval.csv")
    d = pd.concat([e9[e9.braco.isin(["s9_U", "s9_RP", "s9_pm"])], e12])
    d = d[d.viavel]
    d["set"] = np.where(d.instancia.str.contains("syn"), "V", "X")
    gm = d.groupby(["braco", "set", "instancia"]).gap_pct.mean().reset_index()
    for br, g in gm.groupby("braco"):
        a = br.replace("s9_", "").replace("s12_", "")
        NUM[f"s12_gap_{a}"] = f"{g.gap_pct.mean():.3f}"
        for s in ("X", "V"):
            NUM[f"s12{s.lower()}_gap_{a}"] = f"{g[g.set == s].gap_pct.mean():.3f}"
    esc = json.loads((RES / "etapa12_tuning_escolhida.json").read_text())
    for k, w in enumerate(esc["pesos_d"]):
        NUM[f"s12_wd_{k}"] = f"{100 * w:.0f}"
    for k, w in enumerate(esc["pesos_r"]):
        NUM[f"s12_wr_{k}"] = f"{100 * w:.0f}"
    NUM["s12_tune_gap"], NUM["s12_tune_eq"] = f"{esc['gap_tune']:.2f}", f"{esc['gap_pesos_iguais']:.2f}"
    NUM["s12_tune_runs"] = f"{esc['n_configs'] * 72:,}".replace(",", r"\,")
    c["p_holm"] = c.p_holm.fillna(c.p_bruto)
    conf = c[c.grupo == "confirmatorio"]
    for _, r in conf.iterrows():
        k = r.id.replace("-", "_")
        macros_contraste("s12", k, r)
        NUM[f"s12pr_{k}"] = f"{r.p_bruto:.3f}"
    for _, r in cap.iterrows():
        NUM[f"s12cap_{r.braco}"] = f"{100 * r.rec:.0f}"
        NUM[f"s12capci_{r.braco}"] = f"[{100 * r.rec_lo:.0f}, {100 * r.rec_hi:.0f}]"
    lin_main, lin_sup = [], []
    gU = float(NUM["s12_gap_U"])
    for a, nome in ARMS12:
        lin_main.append(f"{nome} & {float(NUM[f's12x_gap_{a}']):.2f} & {float(NUM[f's12v_gap_{a}']):.2f} & {float(NUM[f's12_gap_{a}']):.2f}"
                        + ("" if a == "U" else f" & {sgn(float(NUM[f's12_gap_{a}']) - gU)}"))
    for _, r in conf.iterrows():
        lin_sup.append(f"{r.id.replace('-', ' $-$ ').replace('static', 'static mixture').replace('reg2', 'fixed regret-2').replace('RP', 'adaptive weights').replace('pm', 'probability matching')} & {cel(r)} & {int(r.vit_a)}/{int(r.der_a)}")
    escreve("tab_s12.tex", tabela(
        "Study 12 (static control for the ALNS): mean gap (\\%) on set X (21 instances), the synthetic validation set (30) and both (51), and "
        "the difference from uniform selection over the 51 instances in pp. The static mixture and the fixed repair are new arms; the other "
        "rows are the corresponding arms of Study 9 (same code, instances and seeds).", "tab:s12",
        r"Policy & Set X & Synthetic & All & Difference", lin_main, "lcccc"))
    escreve("tab_s12_tests.tex", tabela(
        "Study 12: confirmatory contrasts on the 51 instances (mean paired difference in pp [95\\% CI]; Holm-adjusted $p$ over five tests; "
        "wins/losses of the first-named arm).", "tab:s12tests", r"Contrast & Difference [CI]; $p$ & Wins/losses", lin_sup, "lcc"))
    NUM["s12_nsig"] = str(int(conf.signif_005.sum()))


ARMS10 = [("U", "Uniform selection"), ("static_S", "Static mixture tuned at $n\\le150$ (Study 4)"),
          ("static_L", "Static mixture tuned at $n=200$--400"), ("ucb1c", "UCB1 (cost-aware)"),
          ("pmc", "Probability matching (cost-aware)"), ("hgucbc", "HG-UCB (cost-aware)"), ("ts", "Thompson sampling"),
          ("eff", "Efficiency UCB")]


def estudo10():
    """Estudo 10 (o valor da selecao aprendida cresce com n?): instancias grandes. Entradas 104/105 do diario."""
    c = pd.read_csv(RES / "analise_etapa10_confirmatoria.csv")
    tam = pd.read_csv(RES / "analise_etapa10_tamanho.csv")
    e = pd.read_csv(RES / "etapa10_eval.csv")
    e = e[e.viavel]
    gap = e.groupby(["braco", "instancia"]).gap_pct.mean().groupby("braco").mean()
    for a, _ in ARMS10:
        NUM[f"s10_gap_{a}"] = f"{gap['s10_' + a]:.3f}"
    NUM["s10_n_inst"] = str(e.instancia.nunique())
    NUM["s10_nruns"] = f"{len(e):,}".replace(",", r"\,")
    lin_main, lin_sup = [], []
    for a, nome in ARMS10:
        if a == "U":
            lin_main.append(f"{nome} & {gap['s10_U']:.2f} & & & ")
            continue
        r = c[c.id == f"{a}-U"].iloc[0]
        macros_contraste("s10", a, r)
        NUM[f"s10pr_{a}"] = f"{r.p_bruto:.3f}"
        NUM[f"s10mde_{a}"] = f"{r.mde_pp:.2f}"
        lin_main.append(f"{nome} & {gap['s10_' + a]:.2f} & {sgn(r.dif_pp)} [{plano(r.ic_lo)}, {plano(r.ic_hi)}] & "
                        f"{r.p_bruto:.3f} & {padj(r.p_holm)}")
        lin_sup.append(f"{nome} $-$ uniform & {cel(r)} & {int(r.vit_a)}/{int(r.der_a)}")
    escreve("tab_s10.tex", tabela(
        "Study 10: instances of 200 to 500 customers (20 synthetic validation instances, five seeds, $E_1$): mean gap (\\%), mean paired "
        "difference from uniform selection in pp [95\\% CI], unadjusted $p$ and Holm-adjusted $p$ over seven tests.", "tab:s10",
        r"Policy & Gap & Difference & $p$ & $p$ (Holm)", lin_main, "lcccc"))
    escreve("tab_s10_tests.tex", tabela(
        "Study 10: confirmatory contrasts against uniform selection on the 20 large synthetic instances (mean paired difference in pp "
        "[95\\% CI]; Holm-adjusted $p$ over seven tests; wins/losses of the first-named arm).", "tab:s10tests",
        r"Contrast & $E_1$ & Wins/losses", lin_sup, "lcc"))
    NUM["s10_nsig"] = str(int(c.signif_005.sum()))
    NUM["s10_nbetter"] = str(int(((c.dif_pp < 0) & c.signif_005).sum()))
    NUM["s10_nworse"] = str(int(((c.dif_pp > 0) & c.signif_005).sum()))
    # tendencia com o tamanho (exploratorio)
    for _, r in tam.iterrows():
        a = r.braco
        NUM[f"s10rho_{a}"] = sgn(r.rho)
        NUM[f"s10rhop_{a}"] = f"{r.p:.2f}"
        for n in (200, 300, 400, 500):
            NUM[f"s10dn{n}_{a}"] = sgn(r[f"dif_n{n}"])
    # participacao do SWAP* e do CROSS sob U por tamanho
    g = e[e.braco == "s10_U"]
    for nn, gg in g.groupby("n_customers"):
        arr = np.array([[int(x) for x in s.split("|")] for s in gg.calls_por_operador])
        pct = 100 * arr.sum(0) / arr.sum()
        NUM[f"s10_swap_n{int(nn)}"], NUM[f"s10_cross_n{int(nn)}"] = f"{pct[9]:.0f}", f"{pct[8]:.0f}"
    esc = json.loads((RES / "etapa10_tuning_escolhida.json").read_text())
    NUM["s10_tune_gap_L"], NUM["s10_tune_gap_eq"] = f"{esc['gap_tune']:.2f}", f"{esc['gap_pesos_iguais']:.2f}"
    # a hipotese pos-hoc dos dados antigos (Entrada 100): estatica do Estudo 4 vs U em CVRPLIB n>120 (E1 e E2)
    if (RES / "etapa4_eval74.csv").exists():
        e3_ = pd.read_csv(RES / "etapa3_full.csv")
        e4_ = pd.read_csv(RES / "etapa4_eval74.csv")
        from scipy import stats
        for nv in (1, 5):
            b = e3_[(e3_.braco == "s3_R5_U") & (e3_.nivel == nv) & e3_.viavel].groupby("instancia").gap_pct.mean()
            s = e4_[(e4_.braco == "s4_R5_static") & (e4_.nivel == nv) & e4_.viavel].groupby("instancia").gap_pct.mean()
            nn = e3_.drop_duplicates("instancia").set_index("instancia").n_customers
            dif = (s - b).dropna()
            gr = dif[nn.reindex(dif.index) > 120]
            NUM[f"s10ph_d_{nv}"] = sgn(gr.mean())
            NUM[f"s10ph_w_{nv}"] = str(int((gr < 0).sum()))
            NUM[f"s10ph_n_{nv}"] = str(len(gr))
            NUM[f"s10ph_p_{nv}"] = f"{stats.wilcoxon(gr).pvalue:.3f}"


def afirmacoes():
    """contagens das afirmacoes universais (mesma logica de checar_afirmacoes.py)"""
    from checar_afirmacoes import aos_vs_baseline
    t = aos_vs_baseline()
    NUM["u_n"] = str(len(t))
    NUM["u_better"] = str(int(((t.dif_pp < 0) & (t.p_holm < 0.05)).sum()))
    NUM["u_worse"] = str(int(((t.dif_pp > 0) & (t.p_holm < 0.05)).sum()))
    hg = t[t.a.str.contains("hgucb") | (t.a == "hg")]   # "hg" = HG-UCB-lite do Estudo 7
    NUM["u_hg_n"] = str(len(hg))
    NUM["u_hg_worse"] = str(int(((hg.dif_pp > 0) & (hg.p_holm < 0.05)).sum()))
    NUM["u_hg_better"] = str(int(((hg.dif_pp < 0) & (hg.p_holm < 0.05)).sum()))


def estouro():
    """estouro medio relativo do orcamento (%) por braco e nivel, Estudos 3-6 (orcamento em avaliacoes de distancia)"""
    v = []
    for arq in ("etapa3_full.csv", "etapa4_eval74.csv", "etapa5_full.csv", "etapa6_full.csv"):
        if (RES / arq).exists() and (RES / arq).stat().st_size > 1000 and not (arq == "etapa6_full.csv" and not (RES / "analise_etapa6_confirmatoria.csv").exists()):
            d = pd.read_csv(RES / arq)
            d["est"] = 100 * (d.calls_usadas - d.orcamento) / d.orcamento
            v += list(d.groupby(["braco", "nivel"]).est.mean().values)
    NUM["overshoot_min"], NUM["overshoot_max"] = f"{min(v):.2f}", f"{max(v):.1f}"
    arq = RES / "verificacao_independente.json"
    if arq.exists():
        j = json.loads(arq.read_text())
        NUM["ver_macros"], NUM["ver_contrastes"] = str(j["medias"]), str(j["contrastes"])


def diagnostico():
    estouro()
    e2 = pd.read_csv(RES / "etapa2_full.csv")
    for b in (800, 4000):
        for arm, nm in (("s2a_U_best_F40", "f40"), ("s2a_U_best_LO", "ubestlo"), ("s2a_FO_shaken_LO", "foshakenlo")):
            NUM[f"sh_{nm}_{b}"] = f"{e2[(e2.braco == arm) & (e2.B == b)].shakes.mean():.0f}"
    arq = RES / "diag_estrutura.csv"
    d = pd.read_csv(arq) if arq.exists() and arq.stat().st_size > 0 else pd.DataFrame()
    if len(d) == 740:                       # 74 instancias x 10 seeds (arquivo completo)
        NUM["diag_n"] = str(len(d))
        NUM["diag_ultima"] = f"{100 * d.apos_ultima_melhora.sum() / d.aplicacoes.sum():.0f}"
        NUM["diag_otimo"] = f"{100 * d.apos_otimo_local.sum() / d.aplicacoes.sum():.0f}"
        por_inst = d.groupby("instancia")[["apos_otimo_local", "aplicacoes"]].sum()
        r = 100 * por_inst.apos_otimo_local / por_inst.aplicacoes
        NUM["diag_otimo_min"], NUM["diag_otimo_max"] = f"{r.min():.0f}", f"{r.max():.0f}"


def complementaridade():
    arq = RES / "diag_complementaridade.csv"
    if not arq.exists():
        return
    d = pd.read_csv(arq)
    NUM["compl_n"] = f"{len(d):,}".replace(",", r"\,")
    for tag, c in (("2opt", "2-opt*"), ("oropt", "Or-opt"), ("swapintra", "Swap-intra"), ("cross", "CROSS"), ("swapstar", "SWAP*")):
        NUM[f"compl_{tag}"] = f"{100 * d[f'ok_{c}'].mean():.1f}"
    caros = (d["ok_CROSS"] + d["ok_SWAP*"]) > 0
    baratos = (d["ok_2-opt*"] + d["ok_Or-opt"] + d["ok_Swap-intra"]) > 0
    NUM["compl_so_caros"] = f"{100 * (caros & ~baratos).mean():.1f}"
    NUM["compl_so_baratos"] = f"{100 * (~caros & baratos).mean():.1f}"
    NUM["compl_algum_caro"] = f"{100 * caros.mean():.1f}"
    NUM["compl_algum_barato"] = f"{100 * baratos.mean():.1f}"


def referencias():
    """tetos externos (HGS-CVRP, Kool zero-shot) e faixa de gap das variantes VNS estudadas"""
    R = RES.parent
    for nome, arq in (("matched", "hgs_74_matched.csv"), ("n01", "hgs_74_n01.csv")):
        h = pd.read_csv(R / arq)
        NUM[f"hgs_{nome}"] = f"{h[h.viavel].groupby('instancia').gap_pct.mean().mean():.3f}"
        NUM[f"hgs_{nome}_viab"] = f"{int(h.viavel.sum())}/{len(h)}"
        NUM[f"hgs_{nome}_t_min"] = f"{h.groupby('instancia').time_limit_s.mean().min():.1f}"
        NUM[f"hgs_{nome}_t_max"] = f"{h.groupby('instancia').time_limit_s.mean().max():.1f}"
    k = pd.read_csv(R / "kool_zeroshot_74.csv")
    NUM["kool_gap"] = f"{k[k.viavel_sem_restricao_k].gap_pct.mean():.2f}"
    NUM["kool_gap_strict"] = f"{k[k.viavel_estrito].gap_pct.mean():.2f}"
    NUM["kool_nstrict"] = str(int(k.viavel_estrito.sum()))
    NUM["kool_n"] = str(len(k))
    gaps = [float(v) for c, v in NUM.items()
            if c.startswith(("s1_gap_", "s2a_gap_", "s3_gap_")) and "literal" not in c]
    NUM["vns_gap_min"], NUM["vns_gap_max"] = f"{min(gaps):.1f}", f"{max(gaps):.1f}"


def main():
    e1 = pd.read_csv(RES / "full.csv").rename(columns={"politica": "arm"})
    e1["nivel"] = e1.B
    e2 = pd.read_csv(RES / "etapa2_full.csv").rename(columns={"braco": "arm"})
    e2["nivel"] = e2.B
    e3 = pd.read_csv(RES / "etapa3_full.csv").rename(columns={"braco": "arm"})
    c1 = pd.read_csv(RES / "analise_confirmatoria.csv").rename(
        columns={"dif_media_pp": "dif_pp", "signif_holm_005": "signif_005"})
    c2 = pd.read_csv(RES / "analise_etapa2_confirmatoria.csv")
    c3 = pd.read_csv(RES / "analise_etapa3_confirmatoria.csv")
    rb = pd.read_csv(RES / "analise_robustez.csv")

    # ------------------------------------------------ Estudo 1
    ordem1 = [("gvns_classico", "Classical GVNS (fixed-order VND)"), ("uniforme", "Uniform"), ("ucb1", "UCB1"),
              ("prob_matching", "Probability matching"), ("hgucb_norm", "HG-UCB (normalized)"),
              ("eps_greedy", r"$\varepsilon$-greedy"), ("hgucb_literal", "HG-UCB literal")]
    lin = []
    for a, nome in ordem1:
        v8, v40 = media_arm(e1, "arm", a, "nivel", 800), media_arm(e1, "arm", a, "nivel", 4000)
        NUM[f"s1_gap_{a}_800"], NUM[f"s1_gap_{a}_4000"] = f2(v8), f2(v40)
        lin.append(f"{nome} & {f2(v8)} & {f2(v40)}")
    escreve("tab_s1.tex", tabela("Study 1: mean gap (\\%) over 74 instances, original loop.", "tab:s1",
                                 r"Policy & $B=800$ & $B=4000$", lin, "lcc"))
    # contrastes do Estudo 1 (Holm m=12)
    nomes_c = {("hgucb_norm", "gvns_classico"): "HG-UCB $-$ classical GVNS", ("hgucb_norm", "uniforme"): "HG-UCB $-$ uniform",
               ("hgucb_norm", "ucb1"): "HG-UCB $-$ UCB1", ("hgucb_norm", "eps_greedy"): r"HG-UCB $-$ $\varepsilon$-greedy",
               ("hgucb_norm", "prob_matching"): "HG-UCB $-$ prob. matching", ("hgucb_literal", "hgucb_norm"): "Literal $-$ normalized HG-UCB"}
    lin = []
    for (a, b), nome in nomes_c.items():
        r8 = c1[(c1.B == 800) & (c1.a == a) & (c1.b == b)].iloc[0]
        r40 = c1[(c1.B == 4000) & (c1.a == a) & (c1.b == b)].iloc[0]
        lin.append(f"{nome} & {cel(r8)} & {cel(r40)}")
        for B, r in ((800, r8), (4000, r40)):
            macros_contraste("s1", f"{a}_{b}_{B}", r)
    escreve("tab_s1_tests.tex", tabela(
        "Study 1: confirmatory contrasts (paired mean difference in pp [95\\% CI]; Holm-adjusted $p$ over 12 tests).",
        "tab:s1tests", r"Contrast & $B=800$ & $B=4000$", lin, "lcc"))
    NUM["s1_nsig"] = str(int(c1.signif_005.sum()))
    NUM["s1_minp_others_4000"] = padj(c1[(c1.B == 4000) & c1.b.isin(["ucb1", "eps_greedy", "prob_matching"])].p_holm.min())

    # ------------------------------------------------ Estudo 2
    ordem2 = [("s2a_U_best_LO", "Random / best / LO", "best structure"), ("s2a_U_shaken_LO", "Random / shaken / LO", ""),
              ("gvns_classico", "Fixed / best / LO", "classical GVNS"), ("s2a_FO_shaken_LO", "Fixed / shaken / LO", ""),
              ("s2a_U_best_F40", "Random / best / F40", ""), ("uniforme", "Random / shaken / F40", "original loop, uniform")]
    lin = []
    for a, nome, papel in ordem2:
        src = e1 if a in ("gvns_classico", "uniforme") else e2
        v8, v40 = media_arm(src, "arm", a, "nivel", 800), media_arm(src, "arm", a, "nivel", 4000)
        NUM[f"s2a_gap_{a}_800"], NUM[f"s2a_gap_{a}_4000"] = f2(v8), f2(v40)
        lin.append(f"{nome} & {f2(v8)} & {f2(v40)} & {papel}")
    escreve("tab_s2a.tex", tabela("Study 2A: mean gap (\\%) over 74 instances by search structure (five operators).",
                                  "tab:s2a", r"Order / acceptance / termination & $B=800$ & $B=4000$ & Role", lin, "lccc"))
    rot2 = {"C1": "Termination, random/best: LO $-$ F40", "C2": "Termination, random/shaken: LO $-$ F40",
            "C3": "Acceptance, random/LO: best $-$ shaken", "C4": "Acceptance, random/F40: best $-$ shaken",
            "C5": "Order, best/LO: fixed $-$ random", "C6": "Order, shaken/LO: fixed $-$ random",
            "D1": "UCB1 $-$ random order (VND)", "D2": r"$\varepsilon$-greedy $-$ random order (VND)",
            "D3": "Prob. matching $-$ random order (VND)", "D4": "HG-UCB $-$ random order (VND)",
            "D5": "UCB1 $-$ fixed order (VND)", "D6": r"$\varepsilon$-greedy $-$ fixed order (VND)",
            "D7": "Prob. matching $-$ fixed order (VND)", "D8": "HG-UCB $-$ fixed order (VND)"}
    lin = []
    for cid, nome in rot2.items():
        r8 = c2[(c2.B == 800) & (c2.id == cid)].iloc[0]
        r40 = c2[(c2.B == 4000) & (c2.id == cid)].iloc[0]
        lin.append(f"{nome} & {cel(r8)} & {cel(r40)}")
        for B, r in ((800, r8), (4000, r40)):
            macros_contraste("s2", f"{cid}_{B}", r)
        if cid == "C5":
            lin.append(r"\addlinespace")
    escreve("tab_s2tests.tex", tabela(
        "Study 2: confirmatory contrasts (mean paired difference in pp [95\\% CI]; Holm-adjusted $p$ over 28 tests). "
        "C: structure; D: adaptive policies inside the best structure.", "tab:s2tests",
        r"Contrast & $B=800$ & $B=4000$", lin, "lcc"))
    NUM["s2_nsig"] = str(int(c2.signif_005.sum()))

    # ------------------------------------------------ Estudo 3
    ordem3 = [("s3_R5_ucb1c", "R5, UCB1 (cost-aware)"), ("s3_R5_U", "R5, uniform"),
              ("s3_R5_pmc", "R5, probability matching (cost-aware)"), ("s3_LO_U", "LO, uniform"),
              ("s3_LO_pmc", "LO, probability matching (cost-aware)"), ("s3_LO_ucb1c", "LO, UCB1 (cost-aware)"),
              ("s3_R5_hgucbc", "R5, HG-UCB (cost-aware)"), ("s3_LO_hgucbc", "LO, HG-UCB (cost-aware)"),
              ("s3_R5_FOC", "R5, fixed order by cost"), ("s3_LO_FOC", "LO, fixed order by cost"),
              ("s3_P5_LO_U", "Five operators, LO, uniform"), ("s3_P5_LO_FO", "Five operators, LO, fixed order")]
    ordem3.sort(key=lambda t: media_arm(e3, "arm", t[0], "nivel", 5) if not t[0].startswith("s3_P5") else 99)
    lin = []
    for a, nome in ordem3:
        v1, v5 = media_arm(e3, "arm", a, "nivel", 1), media_arm(e3, "arm", a, "nivel", 5)
        NUM[f"s3_gap_{a}_1"], NUM[f"s3_gap_{a}_5"] = f2(v1), f2(v5)
        lin.append(f"{nome} & {f2(v1)} & {f2(v5)}")
        if a == "s3_LO_FOC":
            lin.append(r"\addlinespace")
    escreve("tab_s3.tex", tabela("Study 3: mean gap (\\%) over 74 instances (budgets $E_1$ and $E_2$); the last two rows use the "
                                 "original five operators with the same budget.", "tab:s3",
                                 r"Arm & $E_1$ & $E_2$", lin, "lcc"))
    rot3 = {"T1": "Ten vs five operators (LO, uniform)", "T2": "R5 $-$ LO (uniform)", "T3": "Fixed $-$ random order (five operators)",
            "LO_FOCvsU": "LO: fixed order by cost $-$ uniform", "LO_ucb1cvsU": "LO: UCB1 $-$ uniform",
            "LO_pmcvsU": "LO: prob. matching $-$ uniform", "LO_hgucbcvsU": "LO: HG-UCB $-$ uniform",
            "R5_FOCvsU": "R5: fixed order by cost $-$ uniform", "R5_ucb1cvsU": "R5: UCB1 $-$ uniform",
            "R5_pmcvsU": "R5: prob. matching $-$ uniform", "R5_hgucbcvsU": "R5: HG-UCB $-$ uniform",
            "LO_ucb1cvsFOC": "LO: UCB1 $-$ fixed order by cost", "LO_pmcvsFOC": "LO: prob. matching $-$ fixed order by cost",
            "LO_hgucbcvsFOC": "LO: HG-UCB $-$ fixed order by cost", "R5_ucb1cvsFOC": "R5: UCB1 $-$ fixed order by cost",
            "R5_pmcvsFOC": "R5: prob. matching $-$ fixed order by cost", "R5_hgucbcvsFOC": "R5: HG-UCB $-$ fixed order by cost"}
    lin = []
    for cid, nome in rot3.items():
        r1 = c3[(c3.E == 1) & (c3.id == cid)].iloc[0]
        r5 = c3[(c3.E == 5) & (c3.id == cid)].iloc[0]
        lin.append(f"{nome} & {cel(r1)} & {cel(r5)}")
        for E, r in ((1, r1), (5, r5)):
            macros_contraste("s3", f"{cid}_{E}", r)
        if cid in ("T3", "LO_hgucbcvsU", "R5_hgucbcvsU", "LO_hgucbcvsFOC"):
            lin.append(r"\addlinespace")
    escreve("tab_s3tests.tex", tabela(
        "Study 3: confirmatory contrasts (mean paired difference in pp [95\\% CI]; Holm-adjusted $p$ over 34 tests).",
        "tab:s3tests", r"Contrast & $E_1$ & $E_2$", lin, "lcc"))
    NUM["s3_nsig"] = str(int(c3.signif_005.sum()))
    NUM["s3_nsig_E1"] = str(int(c3[c3.E == 1].signif_005.sum()))
    NUM["s3_nsig_E2"] = str(int(c3[c3.E == 5].signif_005.sum()))

    # ------------------------------------------------ postos medios e tempo (Estudo 3)
    principais = [a for a, _ in ordem3 if not a.startswith("s3_P5")]
    nome3 = dict(ordem3)
    lin = []
    for nv in (1, 5):
        piv = e3[(e3.nivel == nv) & e3.viavel & e3.arm.isin(principais)].groupby(["arm", "instancia"]).gap_pct.mean().unstack(0)
        rk = piv.rank(axis=1).mean()
        for a in principais:
            NUM[f"s3_rank_{a}_{nv}"] = f"{rk[a]:.2f}"
    for a in principais:
        t1 = e3[(e3.arm == a) & (e3.nivel == 1)].tempo_s.mean()
        t5 = e3[(e3.arm == a) & (e3.nivel == 5)].tempo_s.mean()
        lin.append(f"{nome3[a]} & {NUM[f's3_rank_{a}_1']} & {NUM[f's3_rank_{a}_5']} & {t1:.1f} & {t5:.1f}")
    escreve("tab_ranks_tempo.tex", tabela(
        "Study 3: mean rank over 74 instances (1 = best; ten arms) and mean wall-clock time per run (s; eleven processes "
        "in parallel, so descriptive only).", "tab:ranks",
        r"Arm & Rank $E_1$ & Rank $E_2$ & Time $E_1$ & Time $E_2$", lin, "lcccc"))

    # ------------------------------------------------ robustez / equivalencia
    n = len(rb)
    for d in ("0.05", "0.10", "0.15", "0.20"):
        NUM[f"eq_{d.replace('.', '')}"] = str(int(rb[f"equiv_{d}"].sum()))
    NUM["eq_n"] = str(n)
    NUM["eq_shapiro"] = str(int((rb.shapiro_p < 0.05).sum()))
    NUM["mde_med"] = f"{rb.mde_pp.median():.2f}"
    NUM["mde_min"], NUM["mde_max"] = f"{rb.mde_pp.min():.2f}", f"{rb.mde_pp.max():.2f}"
    aos = rb[~((rb.estudo == "E3") & (rb.b.str.contains("R5_U|LO_U") & rb.a.str.contains("R5_U"))) & ~rb.a.isin(["hgucb_literal"])
             & ~rb.b.str.contains("hgucb_cego") & ~rb.a.str.contains("static")]    # so politicas adaptativas
    NUM["eq_n_aos"] = str(len(aos))
    NUM["eq_best_lo90"] = f"{abs(aos.ic90_lo.min()):.2f}"
    escreve("tab_equiv.tex", tabela(
        f"Equivalence audit of {n} contrasts between an adaptive policy and its baseline (Studies 1--3): number equivalent "
        "at each margin $\\delta$ (90\\% bootstrap interval within $\\pm\\delta$).", "tab:equiv",
        r"$\delta$ (pp) & 0.05 & 0.10 & 0.15 & 0.20",
        [f"Equivalent contrasts (of {n}) & {NUM['eq_005']} & {NUM['eq_010']} & {NUM['eq_015']} & {NUM['eq_020']}"], "lcccc"))
    # componentes de variancia (braco R5/U do Estudo 3)
    for nv, tag in ((1, "E1"), (5, "E2")):
        piv = e3[(e3.arm == "s3_R5_U") & (e3.nivel == nv) & e3.viavel].pivot(index="instancia", columns="seed", values="gap_pct")
        M = piv.values
        dentro = M.var(axis=1, ddof=1).mean()
        entre = M.mean(axis=1).var(ddof=1) - dentro / M.shape[1]
        NUM[f"var_entre_{tag}"] = f"{100 * entre / (entre + dentro):.0f}"
        NUM[f"var_dentro_{tag}"] = f"{100 * dentro / (entre + dentro):.0f}"
        NUM[f"seed_sd_{tag}"] = f"{np.median(M.std(axis=1, ddof=1)):.2f}"
        NUM[f"seed_se_{tag}"] = f"{np.sqrt(dentro / M.shape[1]):.2f}"

    extras(e1, e2, e3)
    if (RES / "analise_etapa4_confirmatoria.csv").exists():
        estudo4(e3)
    if (RES / "analise_etapa5_confirmatoria.csv").exists():
        estudo5()
    if (RES / "analise_etapa6_confirmatoria.csv").exists():
        estudo6()
    if (RES / "analise_etapa7_confirmatoria.csv").exists():
        estudo7()
    if (RES / "analise_etapa8_confirmatoria.csv").exists():
        estudo8()
    if (RES / "analise_etapa9_confirmatoria.csv").exists():
        estudo9()
    if (RES / "analise_etapa10_confirmatoria.csv").exists():
        estudo10()
    if (RES / "analise_etapa12_confirmatoria.csv").exists():
        estudo12()
    referencias()
    complementaridade()
    afirmacoes()
    suptab()
    log_rep = RES / "reproducao_AB_log.txt"
    if log_rep.exists() and "REPRODUCAO_AB_COMPLETA" in log_rep.read_text(errors="ignore"):
        reproducao()
    diagnostico()
    agrupamentos()
    robustez_macros()

    # ------------------------------------------------ numeros.tex
    linhas = [r"% GERADO por gerar_tabelas.py -- NAO EDITAR A MAO", r"\makeatletter"]
    for k in sorted(NUM):
        linhas.append(r"\expandafter\gdef\csname num@" + k + r"\endcsname{" + NUM[k] + "}")
    linhas += [r"\makeatother",
               r"\newcommand{\num}[1]{\ifcsname num@#1\endcsname\csname num@#1\endcsname\else\textcolor{red}{??\detokenize{#1}??}\fi}"]
    escreve("numeros.tex", "\n".join(linhas) + "\n")
    json.dump(NUM, open(AQUI / "numeros.json", "w"), indent=0, sort_keys=True)
    print(f"geradas 5 tabelas + numeros.tex ({len(NUM)} numeros)")


if __name__ == "__main__":
    main()
