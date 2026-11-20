#!/usr/bin/env python3
"""
gerar_estudos21_23.py
=====================
Macros (\\num{chave}) e tabelas dos Estudos 21 (tamanho do conjunto de operadores no ALNS do CVRP, com a emenda S129 e o diagnostico S_RP), 22 (deslocamentos de implantacao do conjunto de 21
operadores) e 23 (fidelidade da pontuacao e mecanismo da vantagem dos pesos adaptativos), no modelo de `estudo20()` de gerar_tabelas.py. Chamado por gerar_tabelas.main(); le os CSVs brutos e os
CSVs de analise (analise_etapa21.csv, analise_etapa21b.csv, analise_etapa22.csv, analise_etapa23.csv e os de hipoteses, robustez, tendencia, uso, curva). Cada secao so roda se os arquivos existem.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent / "estudo_aos"))
sys.path.insert(0, str(AQUI.parent.parent / "src"))
import __main__ as _principal  # noqa: E402
# quando gerar_tabelas.py roda como script, `import gerar_tabelas` criaria uma SEGUNDA copia do modulo (com outro dicionario NUM): usa-se o modulo em execucao
gt = _principal if str(getattr(_principal, "__file__", "")).endswith("gerar_tabelas.py") else __import__("gerar_tabelas")

RES = gt.RES
NUM = gt.NUM
COLS = ["braco", "conjunto", "instancia", "n_customers", "seed", "viavel", "gap_pct", "uso_d", "uso_r", "iteracoes", "calls_usadas"]
POOL_N = {"P7": 7, "P12": 12, "P21": 21}


def chave(i):
    """Chave de macro de um contraste: o numero H da hipotese ou o identificador sanitizado."""
    m = re.match(r"(H\d+) ", i)
    if m:
        return m.group(1)
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", i.replace("*", "star").replace("@", "_"))).strip("_")


def macros_linhas(pref, aa):
    for _, r in aa.iterrows():
        k = chave(r.id)
        gt.macros_contraste(pref, k, r)
        NUM[f"{pref}apr_{k}"] = f"{r.p_bruto:.3f}"
        NUM[f"{pref}mde_{k}"] = f"{r.mde_pp:.2f}"
        NUM[f"{pref}loa_{k}"], NUM[f"{pref}hia_{k}"] = f"{abs(r.ic_lo):.2f}", f"{abs(r.ic_hi):.2f}"


def flags_hipoteses(pref, caminho, primeiro):
    hy = pd.read_csv(caminho)
    for i, (_, r) in enumerate(hy.iterrows()):
        NUM[f"{pref}_h{primeiro + i}_ok"] = "yes" if bool(r.confirmada) else "no"
    NUM[f"{pref}_hconf"] = str(int(hy.confirmada.sum()))
    NUM[f"{pref}_hn"] = str(len(hy))


def piv(d):
    return d[d.viavel].groupby(["braco", "instancia"]).gap_pct.mean().unstack(0).dropna()


def milhar(n):
    return f"{n:,}".replace(",", r"\,")


# ------------------------------------------------------------------------------------------------ Estudo 21
def dados21():
    import analisar_etapa21 as a21
    ref = a21.referencias().assign(iteracoes=np.nan, calls_usadas=np.nan)
    parts = [ref]
    for p in ("P12", "P21"):
        f = RES / f"etapa21_eval_{p}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["braco"] = d.braco.str.replace(f"s21_{p}_", "", regex=False) + f"@{p}"
            parts.append(d[COLS])
    for arq, pref, novo in (("etapa21b_eval_P21.csv", "s21_P21_", "S129@P21"), ("etapa21c_eval_P21.csv", "s21_P21_", "SRP@P21")):
        f = RES / arq
        if f.exists():
            parts.append(pd.read_csv(f)[COLS].assign(braco=novo))
    d = pd.concat(parts, ignore_index=True)
    nomes = set(d.braco)
    extra = []
    for p in ("P12", "P21"):
        if f"RP@{p}" in nomes and f"RPt@{p}" not in nomes:
            extra.append(d[d.braco == f"RP@{p}"].assign(braco=f"RPt@{p}"))
    if "S65@P21" in nomes and "S129@P21" not in nomes:
        extra.append(d[d.braco == "S65@P21"].assign(braco="S129@P21"))
    return pd.concat([d] + extra, ignore_index=True) if extra else d


def estudo21():
    """Estudo 21 (tamanho do conjunto de operadores). Entradas 136-137 (plano e pre-registro), 139 (emenda 1), 140 (diagnostico S_RP) e as de resultados."""
    if not (RES / "analise_etapa21.csv").exists():
        print("estudo21: sem analise_etapa21.csv, nada gerado")
        return
    d = dados21()
    t = piv(d)
    NUM["s21_n_inst"] = str(len(t))
    ev = {p: pd.read_csv(RES / f"etapa21_eval_{p}.csv") for p in ("P12", "P21") if (RES / f"etapa21_eval_{p}.csv").exists()}
    val = {p: pd.read_csv(RES / f"etapa21_validacao_{p}.csv") for p in ev if (RES / f"etapa21_validacao_{p}.csv").exists()}
    esc = {p: json.loads((RES / f"etapa21_ajuste_escolhidas_{p}.json").read_text()) for p in ev if (RES / f"etapa21_ajuste_escolhidas_{p}.json").exists()}
    NUM["s21_nruns"] = milhar(sum(len(v) for v in ev.values()))
    NUM["s21_valid_runs"] = milhar(sum(len(v) for v in val.values()))
    NUM["s21_tune_runs"] = milhar(sum(len(pd.read_csv(f)) for f in RES.glob("etapa21_ajuste_*_P*.csv")))
    NUM["s21_ninv"] = str(int(sum((~v.viavel).sum() for v in ev.values())))
    for a in t.columns:
        arm, p = a.split("@")
        NUM[f"s21_gap_{p}_{arm}"] = f"{t[a].mean():.3f}"
        NUM[f"s21_g2_{p}_{arm}"] = gt.f2(t[a].mean())
        if f"U@{p}" in t.columns:
            NUM[f"s21_dU_{p}_{arm}"] = gt.sgn((t[a] - t[f"U@{p}"]).mean())
    aa = pd.read_csv(RES / "analise_etapa21.csv")
    macros_linhas("s21", aa)
    flags_hipoteses("s21", RES / "analise_etapa21_hipoteses.csv", 120)
    if (RES / "analise_etapa21b.csv").exists():
        ab = pd.read_csv(RES / "analise_etapa21b.csv")
        macros_linhas("s21", ab)
        hb = pd.read_csv(RES / "analise_etapa21b_hipoteses.csv")
        for i, (_, r) in enumerate(hb.iterrows()):
            NUM[f"s21_h{139 + i}_ok"] = "yes" if bool(r.confirmada) else "no"
        NUM["s21b_hconf"] = str(int(hb.confirmada.sum()))
    rb = pd.read_csv(RES / "analise_etapa21_robustez.csv")
    for _, r in rb[rb.grupo == "robustez"].iterrows():
        h = r.id.split(" ")[0]
        NUM[f"s21rob_{h}_pt"], NUM[f"s21rob_{h}_pperm"] = gt.padj(r.p_t), gt.padj(r.p_perm)
        if not pd.isna(r.get("ic2_lo", np.nan)):
            NUM[f"s21rob_{h}_ci2"] = f"[{gt.plano(r.ic2_lo)}, {gt.plano(r.ic2_hi)}]"
    for h in ("H120", "H121"):
        x = rb[(rb.grupo == "robustez_sem_conjunto") & rb.id.str.startswith(h)]
        if len(x):
            NUM[f"s21rob_{h}_loo_lo"], NUM[f"s21rob_{h}_loo_hi"] = gt.sgn(x.dif_pp.min()), gt.sgn(x.dif_pp.max())
            NUM[f"s21rob_{h}_loo_minp"] = gt.padj(x.p_wilcoxon.max())
    if (RES / "analise_etapa21_tendencia.csv").exists():
        tr = pd.read_csv(RES / "analise_etapa21_tendencia.csv").iloc[0]
        NUM["s21_trend_slope"], NUM["s21_trend_p"] = gt.sgn(tr.media), gt.padj(tr.p_wilcoxon)
        for p in ("P7", "P12", "P21"):
            NUM[f"s21_delta_{p}"] = gt.sgn(tr[f"delta_{p}"])
    # ajuste, selecao, validacao, curva de aprendizado
    for p, e in esc.items():
        traj = {x["geracao"]: x for x in e["trajetoria"]}
        NUM[f"s21_es_{p}_gap0"], NUM[f"s21_es_{p}_gap33"] = f"{traj[0]['gap_ajuste']:.3f}", f"{traj[8]['gap_ajuste']:.3f}"
        NUM[f"s21_es_{p}_gap65"] = f"{traj[16]['gap_ajuste']:.3f}"
        NUM[f"s21_hp_{p}_rho"], NUM[f"s21_hp_{p}_seg"] = f"{e['RPt']['params']['rho']:g}", f"{e['RPt']['params']['segmento']:g}"
        NUM[f"s21_hp_{p}_gap"], NUM[f"s21_hp_{p}_gap0"] = f"{e['RPt']['gap_ajuste']:.3f}", f"{e['RPt']['gap_padrao']:.3f}"
        NUM[f"s21_hp_{p}_changed"] = "yes" if e["RPt"]["mudou"] else "no"
        v = val.get(p)
        if v is not None:
            for a, g in v[v.viavel].groupby("braco").gap_pct.mean().items():
                NUM[f"s21_val_{p}_{a.replace(f's21_{p}_', '').replace(f'val21_{p}_', 'v')}"] = f"{g:.3f}"
    f_b = RES / "etapa21b_ajuste_escolhidas_P21.json"
    if f_b.exists():
        eb = json.loads(f_b.read_text())
        trb = {x["geracao"]: x for x in eb["trajetoria"]}
        NUM["s21_es_P21_gap129"] = f"{trb[32]['gap_ajuste']:.3f}"
        NUM["s21_es_P21_changed129"] = "yes" if eb["S129"]["mudou"] else "no"
    # iteracoes e avaliacoes por iteracao (P21)
    for arm in ("U", "RP", "RPt", "S33", "S65", "HG", "S129", "SRP"):
        h = d[d.braco == f"{arm}@P21"]
        if len(h) and h.iteracoes.notna().all():
            NUM[f"s21_it_{arm}"] = milhar(int(round(h.iteracoes.mean())))
            NUM[f"s21_evit_{arm}"] = milhar(int(round(h.calls_usadas.mean() / h.iteracoes.mean())))
    # uso de operadores (P12 e P21)
    for p in ("P12", "P21"):
        for arm in sorted({a.split("@")[0] for a in d.braco.unique() if a.endswith("@" + p)}):
            g = d[d.braco == f"{arm}@{p}"]
            for col, tag in (("uso_d", "d"), ("uso_r", "r")):
                m = pd.DataFrame(g[col].str.split("|").apply(lambda x: [float(y) for y in x]).tolist()).mean()
                m = 100 * m / m.sum()
                for j, w in enumerate(m):
                    NUM[f"s21_u{tag}_{p}_{arm}_{j}"] = f"{w:.0f}"
    fg = RES / "analise_etapa21_expl_folga.csv"
    if fg.exists():
        for _, r in pd.read_csv(fg).iterrows():
            NUM[f"s21_hr_rho_{r.pool}"], NUM[f"s21_hr_p_{r.pool}"] = f"{r.rho_gap:+.2f}".replace("-", "$-$"), gt.padj(r.p_gap)
            for lab in ("baixo", "medio", "alto"):
                NUM[f"s21_hr_d_{lab}_{r.pool}"], NUM[f"s21_hr_g_{lab}_{r.pool}"] = gt.sgn(r[f"dif_{lab}"]), f"{r[f"gapU_{lab}"]:.2f}"
    # folga corrigida (Entrada 145): correlacao em metades das seeds, folga medida por bracos independentes, reducao relativa e validacao por tamanho
    fm = RES / "analise_folga_metades.csv"
    if fm.exists():
        neg = lambda x: f"{x:+.2f}".replace("-", "$-$")  # noqa: E731
        for _, r in pd.read_csv(fm).iterrows():
            p = r.pool
            NUM[f"hm_half_{p}"], NUM[f"hm_lo_{p}"], NUM[f"hm_hi_{p}"] = neg(r.rho_metades), neg(r.rho_metades_lo), neg(r.rho_metades_hi)
            NUM[f"hm_ind_{p}"], NUM[f"hm_indp_{p}"] = neg(r.rho_outros), gt.padj(r.p_outros)
            NUM[f"hm_rel_{p}"] = f"{-100 * r.rel_mediana:.0f}"
            NUM[f"hm_relrho_{p}"], NUM[f"hm_relp_{p}"] = neg(r.rho_rel), gt.padj(r.p_rel)
    ft = RES / "analise_folga_tamanho.csv"
    if ft.exists():
        f = pd.read_csv(ft)
        v = f[f.conjunto.str.startswith("validacao P21") & (f.classe == "TODAS")].iloc[0]
        NUM["ft_val_dif"], NUM["ft_val_p"], NUM["ft_val_gap"], NUM["ft_val_rel"] = gt.sgn(v.dif_pp), gt.padj(v.p_wilcoxon), f"{v.gap_u:.2f}", f"{-100 * v.rel:.0f}"
    # diagnostico de operadores nulos e perfil de custo
    dg = RES / "etapa21_diag_noop.csv"
    if dg.exists():
        for _, r in pd.read_csv(dg).iterrows():
            NUM[f"s21_noop_{r.destruicao}"] = f"{r.mesma_solucao_pct:.0f}"
            NUM[f"s21_qrem_{r.destruicao}"] = f"{r.clientes_removidos:.0f}"
    tabelas21(d, t, aa, esc, val)


def tabelas21(d, t, aa, esc, val):
    import alns_ext as ax
    # (1) conjuntos e operadores: definicao curta, conjunto de entrada, desempenho sozinho no piloto, taxa de reconstrucao da mesma solucao
    pil = pd.read_csv(RES / "etapa21_piloto.csv")
    sozinho = pil.groupby("rotulo").gap.mean()
    noop = {}
    if (RES / "etapa21_diag_noop.csv").exists():
        noop = pd.read_csv(RES / "etapa21_diag_noop.csv").set_index("destruicao").mesma_solucao_pct.to_dict()
    defs = {"aleatoria": "random customers", "pior": "worst removal gain, $p=3$", "relacionada": "Shaw (distance, demand), $p=6$", "rota": "one random route (at most $q_{\\max}$)",
            "radial": "seed and its nearest customers", "sequencia": "contiguous strings (length $\\le10$) of nearby routes", "demanda": "customers of demand closest to the seed's",
            "aleatoria_leve": "random, $0.4q$", "aleatoria_forte": "random, $2q$", "pior_leve": "worst, $0.4q$", "relacionada_forte": "Shaw, $1.7q$",
            "rota_pequena": "the route with fewest customers", "guloso": "greedy (regret-1)", "regret2": "regret-2", "barato": "random order, best position",
            "regret3": "regret-3", "demanda_dec": "decreasing demand, best position", "regret4": "regret-4", "longe_primeiro": "farthest from depot first",
            "barato_ruidoso": "random order, noisy cost", "blink": "random/demand/far/near order, 1\\% blink"}
    entra = {}
    for p, (D, R) in ax.POOLS.items():
        for i, (n, _f) in enumerate(D):
            entra.setdefault(("D", n), p)
        for i, (n, _f) in enumerate(R):
            entra.setdefault(("R", n), p)
    lin = []
    for tipo, tab in (("D", ax.DESTRUIR_P21), ("R", ax.REPARAR_P21)):
        for n, _f in tab:
            pg = sozinho.get(f"{'D' if tipo == 'D' else 'R'}:{n}", np.nan)
            nz = noop.get(n, np.nan) if tipo == "D" else np.nan
            lin.append(" & ".join([("Destroy" if tipo == "D" else "Repair"), n.replace("_", r"\_"), entra[(tipo, n)], defs[n], f"{pg:.2f}" if pg == pg else "---",
                                   f"{nz:.0f}" if nz == nz else "---"]))
    gt.escreve("tab_s21p.tex", gt.tabela(
        "Study 21: the 21 operators of the nested pools (P7 = the first four destroy and three repair operators of Study 9; P12 adds those marked P12; P21 all), the mean gap (\\%) of each "
        "operator used alone in the pilot on the 24 tuning instances (one seed; destroy with regret-2 repair, repair with random destroy) and the share (\\%) of its applications that rebuild "
        "exactly the current solution (diagnostic on three tuning instances).", "tab:s21p", r"Type & Operator & Pool & Definition & Alone (\%) & Same solution (\%)", lin, "llllcc"))
    # (2) bracos por conjunto: esforco, ajuste, validacao, avaliacao, diferenca contra a uniforme
    ARM = [("U", "Uniform selection", 0), ("RP", "Adaptive weights, default", 0), ("RPt", "Adaptive weights, tuned", 33), ("S33", "Static mixture, 33 evaluations", 33),
           ("S65", "Static mixture, 65 evaluations", 65), ("S129", "Static mixture, 129 evaluations", 129), ("SRP", "Static mixture = usage of RP", 0), ("HG", "HG-UCB (Study 9)", 0)]
    lin = []
    s129_proprio = (RES / "etapa21b_eval_P21.csv").exists()
    for p in ("P7", "P12", "P21"):
        if f"U@{p}" not in t.columns:
            continue
        lin.append(r"\multicolumn{6}{l}{\emph{" + f"{p}: {POOL_N[p]} operators, {len(t)} instances" + r"}}")
        for a, nome, nev in ARM:
            if f"{a}@{p}" not in t.columns or (a == "S129" and not s129_proprio):
                continue
            col = t[f"{a}@{p}"]
            u = t[f"U@{p}"]
            dd = col - u
            vg = ""
            if p in val:
                vv = val[p][val[p].viavel & (val[p].braco == f"s21_{p}_{a}")]
                vg = f"{vv.gap_pct.mean():.2f}" if len(vv) else "---"
            lin.append(" & ".join([nome, str(nev), vg if vg else "---", f"{col.mean():.2f}"] + (["---", "---"] if a == "U" else [gt.sgn(dd.mean()), f"{int((dd < -1e-9).sum())}/{int((dd > 1e-9).sum())}"])))
    gt.escreve("tab_s21a.tex", gt.tabela(
        "Study 21: tuning effort (evaluations of the mixture or of the hyperparameters), validation gap (tuning instances, three new seeds; descriptive), evaluation gap (\\%) of each arm in "
        "each pool, difference from uniform selection (pp, descriptive) and wins/losses. The arms of P7 are reused from Studies 9, 12 and 20; S129 and the usage mixture SRP exist only in P21. "
        "An arm that was not run because the search found nothing better is a copy of the arm it equals.", "tab:s21a",
        r"Arm & Evaluations & Validation & Evaluation & $\Delta$ vs.\ uniform & W/L", lin, "lccccc"))
    # (3) contrastes
    rot = [("H120", "Primary", r"RP $-$ S33, P21"), ("H121", "Primary", r"difference of (RP $-$ S33): P21 $-$ P7"), ("H122", "Secondary", r"RPt $-$ S33, P21"),
           ("H123", "Secondary", r"S65 $-$ S33, P21"), ("H124", "Secondary", r"RP $-$ U, P21"), ("H125", "Secondary", r"S33 $-$ U, P21"), ("H126", "Secondary", r"HG $-$ U, P21"),
           ("H127", "Tertiary", r"RP $-$ S33, P12"), ("H128", "Tertiary", r"RPt $-$ S33, P12"), ("H129", "Tertiary", r"difference of (RP $-$ S33): P12 $-$ P7")]
    lin = []
    for cid, g, nome in rot:
        mm = aa[aa.id.str.startswith(cid + " ")]
        if not len(mm):
            continue
        r = mm.iloc[0]
        lin.append(" & ".join([g, nome, f"{gt.sgn(r.dif_pp)} [{gt.sgn(r.ic_lo)}, {gt.sgn(r.ic_hi)}]", gt.padj(r.p_holm), f"{int(r.vit_a)}/{int(r.der_a)}", f"{r.mde_pp:.2f}"]))
    if (RES / "analise_etapa21b.csv").exists():
        ab = pd.read_csv(RES / "analise_etapa21b.csv")
        for cid, nome in (("H139", r"S129 $-$ RP, P21"), ("H140", r"S129 $-$ S65, P21")):
            r = ab[ab.id.str.startswith(cid + " ")].iloc[0]
            lin.append(" & ".join(["Amendment", nome, f"{gt.sgn(r.dif_pp)} [{gt.sgn(r.ic_lo)}, {gt.sgn(r.ic_hi)}]", gt.padj(r.p_bruto), f"{int(r.vit_a)}/{int(r.der_a)}", f"{r.mde_pp:.2f}"]))
    for cid, nome in (("RP-S65@P21", r"RP $-$ S65, P21"), ("RP-U@P12", r"RP $-$ U, P12"), ("S33-U@P12", r"S33 $-$ U, P12"), ("RP-U@P7", r"RP $-$ U, P7"),
                      ("RP-S33@P7", r"RP $-$ S33, P7")):
        m = aa[aa.id == cid]
        if len(m):
            r = m.iloc[0]
            lin.append(" & ".join(["Exploratory", nome, f"{gt.sgn(r.dif_pp)} [{gt.sgn(r.ic_lo)}, {gt.sgn(r.ic_hi)}]", gt.padj(r.p_bruto), f"{int(r.vit_a)}/{int(r.der_a)}", f"{r.mde_pp:.2f}"]))
    gt.escreve("tab_s21f.tex", gt.tabela(
        "Study 21: confirmatory contrasts (mean paired difference in pp [95\\% CI]; Holm-adjusted $p$ for the primary (over two), secondary (over five) and tertiary (over three) tests, "
        "unadjusted for the amendment and the exploratory contrasts; wins/losses of the first-named arm; MDE: minimum detectable effect at 80\\% power).", "tab:s21f",
        r"Test & Contrast & $\Delta$ [CI] & $p$ & W/L & MDE", lin, "llcccc"))
    # (4) curva de aprendizado do ajuste (validacao com seeds novas)
    cv = RES / "analise_etapa21_curva.csv"
    if cv.exists():
        c = pd.read_csv(cv)
        cb = RES / "analise_etapa21b_curva.csv"
        lin = []
        for p in ("P12", "P21"):
            cc = c[c.pool == p]
            for _, r in cc.iterrows():
                lin.append(" & ".join([p, str(int(r.geracao)), str(int(r.n_avaliacoes)), f"{r.gap_ajuste:.3f}", f"{r.gap_validacao:.3f}", f"{r.gap_validacao_RP:.3f}", f"{r.gap_validacao_U:.3f}"]))
        if cb.exists():
            for _, r in pd.read_csv(cb).iterrows():
                if int(r.geracao) > 16:
                    lin.append(" & ".join(["P21", str(int(r.geracao)), str(int(r.n_avaliacoes)), f"{r.gap_ajuste:.3f}", f"{r.gap_validacao:.3f}", "---", "---"]))
        gt.escreve("tab_s21c.tex", gt.tabela(
            "Study 21 (exploratory): learning curve of the tuning of the static mixture. For each generation of the evolution strategy, the evaluations spent, the tuning gap of the incumbent "
            "(selection-biased), its gap on the tuning instances with three new seeds (validation), and the validation gap of the default adaptive weights (RP, no tuning) and of uniform selection.",
            "tab:s21c", r"Pool & Generation & Evaluations & Tuning & Validation & RP & U", lin, "lcccccc"))
    # (5) pesos das misturas e uso do RP (P21)
    if "P21" in esc:
        nd = [n for n, _ in ax.DESTRUIR_P21]
        nr = [n for n, _ in ax.REPARAR_P21]
        lin = []
        mis = [("S33", esc["P21"]["S33"]), ("S65", esc["P21"]["S65"])]
        f_b = RES / "etapa21b_ajuste_escolhidas_P21.json"
        if f_b.exists():
            mis.append(("S129", json.loads(f_b.read_text())["S129"]))
        fs = RES / "etapa21c_srp_pesos.json"
        if fs.exists():
            mis.append(("SRP", json.loads(fs.read_text())))
        for nome, e in mis:
            lin.append(" & ".join([nome, "destroy"] + [f"{100 * w:.0f}" for w in e["pesos_d"]]))
            lin.append(" & ".join([nome, "repair"] + [f"{100 * w:.0f}" for w in e["pesos_r"]] + [""] * (12 - len(e["pesos_r"]))))
        for arm in ("RP",):
            for col, tag in (("d", "destroy"), ("r", "repair")):
                k = 12 if tag == "destroy" else 9
                lin.append(" & ".join([f"{arm} usage", tag] + [NUM.get(f"s21_u{col}_P21_{arm}_{j}", "") for j in range(k)] + [""] * (12 - k)))
        gt.escreve("tab_s21w.tex", gt.tabela(
            "Study 21: probabilities (\\%) of the 12 destroy operators (random, worst, related, route, radial, string, demand, light random, heavy random, light worst, heavy related, smallest route) "
            "and of the nine repair operators (greedy, regret-2, cheapest, regret-3, decreasing demand, regret-4, farthest first, noisy, blink) in the tuned mixtures of P21, and mean usage of the "
            "default adaptive weights in the evaluation runs.", "tab:s21w", "Arm & Kind & " + " & ".join(str(i + 1) for i in range(12)), lin, "ll" + "c" * 12))


# ------------------------------------------------------------------------------------------------ Estudo 22
def estudo22():
    if not (RES / "analise_etapa22.csv").exists():
        print("estudo22: sem analise_etapa22.csv, nada gerado")
        return
    import analisar_etapa22 as a22
    d = a22.completar(pd.concat([a22.ler(r) for r in a22.REG] + [a22.referencia_x()], ignore_index=True))
    abc = d[d.conjunto.isin(["A", "B", "CMT"])]
    tabc, tx = piv(abc), piv(d[d.conjunto == "X"])
    NUM["s22_nruns"] = milhar(sum(len(pd.read_csv(RES / f"etapa22_eval_{r}.csv")) for r in a22.REG))
    NUM["s22_ninv"] = str(int(sum((~pd.read_csv(RES / f"etapa22_eval_{r}.csv").viavel).sum() for r in a22.REG)))
    NUM["s22_n_abc"], NUM["s22_n_x"] = str(len(tabc)), str(len(tx))
    if (RES / "etapa22_rrt_delta0.json").exists():
        NUM["s22_delta0"] = f"{json.loads((RES / 'etapa22_rrt_delta0.json').read_text())['delta0']:g}"
    for pv in (tabc, tx):
        for a in pv.columns:
            arm, reg = a.split("@")
            NUM[f"s22_gap_{reg}_{arm}"] = f"{pv[a].mean():.3f}"
            NUM[f"s22_g2_{reg}_{arm}"] = gt.f2(pv[a].mean())
    aa = pd.read_csv(RES / "analise_etapa22.csv")
    macros_linhas("s22", aa)
    flags_hipoteses("s22", RES / "analise_etapa22_hipoteses.csv", 141)
    tm = pd.read_csv(RES / "analise_etapa22_tamanho.csv").iloc[0]
    NUM["s22_size_rho"], NUM["s22_size_p"] = f"{tm.rho:+.2f}".replace("-", "$-$"), gt.padj(tm.p)
    for _, r in pd.read_csv(RES / "analise_etapa22_uso.csv").iterrows():
        NUM[f"s22_it_{r.regime}_{r.braco}"] = milhar(int(round(r.iteracoes)))
    # tabela de bracos por regime
    nomes = {"cls": "A/B/CMT, $1500n^2$", "short": "A/B/CMT, $375n^2$", "longabc": "A/B/CMT, $30000n^2$", "longx": "X, $6000n^2$", "acc": "A/B/CMT, record-to-record"}
    arms = ["U", "RP", "RPt", "S33", "S65", "S129", "HG"]
    lin = []
    for reg in a22.REGS:
        pv = tx if reg == "longx" else tabc
        cel = []
        for a in arms:
            c = f"{a}@{reg}"
            cel.append(f"{pv[c].mean():.2f}" if c in pv.columns else "---")
        lin.append(" & ".join([nomes[reg]] + cel))
    lin.append(" & ".join(["X, $1500n^2$ (Study 21)"] + [f"{tx[f'{a}@refx'].mean():.2f}" if f"{a}@refx" in tx.columns else "---" for a in arms]))
    gt.escreve("tab_s22a.tex", gt.tabela(
        "Study 22: mean gap (\\%) of the frozen arms of Study 21 (pool of 21 operators) in each deployment regime; the regimes that omit an arm did not run it (RPt, S33 and HG only in the first regime). "
        "A copy of S65 stands for S129 when the continued search found nothing better.", "tab:s22a", "Regime & " + " & ".join(arms), lin, "l" + "c" * len(arms)))
    rot = [(f"H{141 + i}", "Primary", f"RP $-$ U, {nomes[r]}") for i, r in enumerate(a22.REGS)] + [(f"H{146 + i}", "Secondary", f"RP $-$ S*, {nomes[r]}") for i, r in enumerate(a22.REGS)] + [
        ("H151", "Tertiary", "HG $-$ U, A/B/CMT"), ("H152", "Tertiary", "S* $-$ U, A/B/CMT"), ("H153", "Tertiary", "S33 $-$ U, A/B/CMT")]
    lin = []
    for cid, g, nome in rot:
        r = aa[aa.id.str.startswith(cid + " ")].iloc[0]
        lin.append(" & ".join([g, nome, f"{gt.sgn(r.dif_pp)} [{gt.sgn(r.ic_lo)}, {gt.sgn(r.ic_hi)}]", gt.padj(r.p_holm), f"{int(r.vit_a)}/{int(r.der_a)}", f"{r.mde_pp:.2f}"]))
    gt.escreve("tab_s22f.tex", gt.tabela(
        "Study 22: confirmatory contrasts (mean paired difference in pp [95\\% CI]; Holm-adjusted $p$ over five for the primary and the secondary tests and over three for the tertiary ones; "
        "wins/losses of the first-named arm; MDE: minimum detectable effect at 80\\% power). S*: the static mixture of greatest tuning effort.", "tab:s22f",
        r"Test & Contrast & $\Delta$ [CI] & $p$ & W/L & MDE", lin, "llcccc"))


# ------------------------------------------------------------------------------------------------ Estudo 23
def estudo23():
    if not (RES / "analise_etapa23.csv").exists():
        print("estudo23: sem analise_etapa23.csv, nada gerado")
        return
    import analisar_etapa23 as a23
    esc = json.loads((RES / "etapa23_selecao.json").read_text())
    NUM["s23_cstar"] = f"{esc['c_estrela']:g}"
    for k, v in esc["gap_ajuste_Uq"].items():
        NUM[f"s23_calq_{k.replace('.', '')}"] = f"{v:.3f}"
    NUM["s23_prun_d"], NUM["s23_prun_r"] = str(len(esc["podado_destruir"])), str(len(esc["podado_reparar"]))
    NUM["s23_prun_dnames"] = ", ".join(n.replace("_", r"\_") for n in esc["podado_destruir"])
    NUM["s23_prun_rnames"] = ", ".join(n.replace("_", r"\_") for n in esc["podado_reparar"])
    d = pd.concat([a23.referencias(), a23.ler_novo("p21", a23.NOVOS_P21), a23.ler_novo("p7", ("RPh7",))], ignore_index=True)
    if "Uq@P21" not in set(d.braco):
        d = pd.concat([d, d[d.braco == "U@P21"].assign(braco="Uq@P21")], ignore_index=True)
    if "S129@P21" not in set(d.braco):
        d = pd.concat([d, d[d.braco == "S65@P21"].assign(braco="S129@P21")], ignore_index=True)
    t = piv(d)
    NUM["s23_nruns"] = milhar(sum(len(pd.read_csv(RES / f"etapa23_eval_{r}.csv")) for r in ("p21", "p7")))
    for a in t.columns:
        arm, p = a.split("@")
        NUM[f"s23_gap_{p}_{arm}"] = f"{t[a].mean():.3f}"
        NUM[f"s23_g2_{p}_{arm}"] = gt.f2(t[a].mean())
    aa = pd.read_csv(RES / "analise_etapa23.csv")
    macros_linhas("s23", aa)
    flags_hipoteses("s23", RES / "analise_etapa23_hipoteses.csv", 154)
    for arm in ("RPh", "RPsr", "Upr", "Uq", "RP", "U"):
        h = d[d.braco == f"{arm}@P21"]
        if len(h) and h.iteracoes.notna().all():
            NUM[f"s23_it_{arm}"] = milhar(int(round(h.iteracoes.mean())))
        if len(h):
            for col, tag in (("uso_d", "d"), ("uso_r", "r")):
                m = pd.DataFrame(h[col].str.split("|").apply(lambda x: [float(y) for y in x]).tolist()).mean()
                m = 100 * m / m.sum()
                for j, w in enumerate(m):
                    NUM[f"s23_u{tag}_{arm}_{j}"] = f"{w:.0f}"
    arms = [("U", "Uniform selection"), ("S33", "Static mixture, 33 evaluations"), ("S65", "Static mixture, 65 evaluations"), ("S129", "Static mixture, 129 evaluations"),
            ("SRP", "Static mixture = usage of RP"), ("Uq", f"Uniform, removal size $\\times${esc['c_estrela']:g}"), ("Upr", "Uniform on the pruned pool"),
            ("RP", "Adaptive weights (scoring of Studies 9--22)"), ("RPsr", "Adaptive weights without the smallest-route operator"), ("RPh", "Adaptive weights, faithful scoring")]
    lin = [r"\multicolumn{4}{l}{\emph{P21: 21 operators}}"]
    s129_proprio = (RES / "etapa21b_eval_P21.csv").exists()
    for a, nome in arms:
        c = f"{a}@P21"
        if c in t.columns and not (a == "S129" and not s129_proprio):
            dd = t[c] - t["U@P21"]
            lin.append(" & ".join([nome, f"{t[c].mean():.2f}", "---" if a == "U" else gt.sgn(dd.mean()), "---" if a == "U" else f"{int((dd < -1e-9).sum())}/{int((dd > 1e-9).sum())}"]))
    lin.append(r"\multicolumn{4}{l}{\emph{P7: 7 operators}}")
    for a, nome in (("U", "Uniform selection"), ("S33", "Static mixture, 33 evaluations"), ("RP", "Adaptive weights (scoring of Studies 9--22)"), ("RPh7", "Adaptive weights, faithful scoring")):
        c = f"{a}@P7"
        if c in t.columns:
            dd = t[c] - t["U@P7"]
            lin.append(" & ".join([nome, f"{t[c].mean():.2f}", "---" if a == "U" else gt.sgn(dd.mean()), "---" if a == "U" else f"{int((dd < -1e-9).sum())}/{int((dd > 1e-9).sum())}"]))
    gt.escreve("tab_s23a.tex", gt.tabela(
        "Study 23: mean gap (\\%) on the 51 evaluation instances of the arms that test the fidelity of the scoring and the mechanism of the advantage of the adaptive weights, difference from "
        "uniform selection (pp, descriptive) and wins/losses.", "tab:s23a", r"Arm & Gap (\%) & $\Delta$ vs.\ uniform & W/L", lin, "lccc"))
    rot = [("H154", "Primary", r"RPh $-$ U, P21"), ("H155", "Primary", r"RPh $-$ S*, P21"), ("H156", "Primary", r"RPh7 $-$ S33, P7"), ("H157", "Secondary", r"RPh $-$ RP, P21"),
           ("H158", "Secondary", r"RPh7 $-$ RP, P7"), ("H159", "Secondary", r"RP $-$ SRP, P21"), ("H160", "Secondary", r"RP $-$ Upr, P21"), ("H161", "Secondary", r"RP $-$ Uq, P21"),
           ("H162", "Secondary", r"RPsr $-$ RP, P21"), ("H163", "Tertiary", r"SRP $-$ U, P21"), ("H164", "Tertiary", r"Upr $-$ U, P21"), ("H165", "Tertiary", r"Uq $-$ U, P21"),
           ("H166", "Tertiary", r"RPsr $-$ U, P21")]
    lin = []
    for cid, g, nome in rot:
        r = aa[aa.id.str.startswith(cid + " ")].iloc[0]
        lin.append(" & ".join([g, nome, f"{gt.sgn(r.dif_pp)} [{gt.sgn(r.ic_lo)}, {gt.sgn(r.ic_hi)}]", gt.padj(r.p_holm), f"{int(r.vit_a)}/{int(r.der_a)}", f"{r.mde_pp:.2f}"]))
    gt.escreve("tab_s23f.tex", gt.tabela(
        "Study 23: confirmatory contrasts (mean paired difference in pp [95\\% CI]; Holm-adjusted $p$ over three for the primary, six for the secondary and four for the tertiary tests; "
        "wins/losses of the first-named arm; MDE: minimum detectable effect at 80\\% power). S*: the static mixture of greatest tuning effort.", "tab:s23f",
        r"Test & Contrast & $\Delta$ [CI] & $p$ & W/L & MDE", lin, "llcccc"))


# ------------------------------------------------------------------------------------------------ Estudo 24
def estudo24():
    if not (RES / "analise_etapa24.csv").exists():
        print("estudo24: sem analise_etapa24.csv, nada gerado")
        return
    import analisar_etapa23 as a23
    import analisar_etapa24 as a24
    d = pd.concat([a23.referencias(), a23.ler_novo("p21", a23.NOVOS_P21), a24.ler24()], ignore_index=True)
    if "Uq@P21" not in set(d.braco):
        d = pd.concat([d, d[d.braco == "U@P21"].assign(braco="Uq@P21")], ignore_index=True)
    if "S129@P21" not in set(d.braco):
        d = pd.concat([d, d[d.braco == "S65@P21"].assign(braco="S129@P21")], ignore_index=True)
    t = piv(d[d.braco.str.endswith("@P21")])
    NUM["s24_nruns"] = milhar(len(pd.read_csv(RES / "etapa24_eval_P21.csv")))
    for a in t.columns:
        arm = a.split("@")[0]
        NUM[f"s24_gap_{arm}"] = f"{t[a].mean():.3f}"
        NUM[f"s24_g2_{arm}"] = gt.f2(t[a].mean())
        NUM[f"s24_dU_{arm}"] = gt.sgn((t[a] - t["U@P21"]).mean())
    aa = pd.read_csv(RES / "analise_etapa24.csv")
    macros_linhas("s24", aa)
    flags_hipoteses("s24", RES / "analise_etapa24_hipoteses.csv", 167)
    for arm in a24.POL:
        h = d[d.braco == f"{arm}@P21"]
        if len(h) and h.iteracoes.notna().all():
            NUM[f"s24_it_{arm}"] = milhar(int(round(h.iteracoes.mean())))
        for col, tag in (("uso_d", "d"), ("uso_r", "r")):
            m = pd.DataFrame(h[col].str.split("|").apply(lambda x: [float(y) for y in x]).tolist()).mean()
            m = 100 * m / m.sum()
            for j, w in enumerate(m):
                NUM[f"s24_u{tag}_{arm}_{j}"] = f"{w:.0f}"
    for arm in ("U", "RP", "RPh"):
        h = d[d.braco == f"{arm}@P21"]
        if len(h) and h.iteracoes.notna().all():
            NUM[f"s24_it_{arm}"] = milhar(int(round(h.iteracoes.mean())))
    nomes = [("U", "Uniform selection"), ("RPh", "Adaptive weights (faithful scoring)"), ("PM", "Probability matching"), ("EXP3", "EXP3"), ("UCB1", "UCB1"),
             ("TS", "Thompson sampling"), ("EFF", "Efficiency UCB"), ("HG", "HG-UCB (Study 9)"), ("S65", "Static mixture, 65 evaluations")]
    lin = []
    for a, nome in nomes:
        c = f"{a}@P21"
        if c in t.columns:
            dd = t[c] - t["U@P21"]
            lin.append(" & ".join([nome, f"{t[c].mean():.2f}", "---" if a == "U" else gt.sgn(dd.mean()), "---" if a == "U" else f"{int((dd < -1e-9).sum())}/{int((dd > 1e-9).sum())}"]))
    gt.escreve("tab_s24a.tex", gt.tabela(
        "Study 24: mean gap (\\%) on the 51 evaluation instances of the pool of 21 operators of the policy families of Study 9 without tuning and with the faithful scoring, the adaptive "
        "weights, HG-UCB and the mixture of greatest effort; difference from uniform selection (pp, descriptive) and wins/losses.", "tab:s24a", r"Arm & Gap (\%) & $\Delta$ vs.\ uniform & W/L", lin, "lccc"))
    rot = [(f"H{167 + i}", "Primary", f"{p} $-$ U") for i, p in enumerate(a24.POL)] + [(f"H{172 + i}", "Secondary", f"RPh $-$ {p}") for i, p in enumerate(a24.POL)]
    lin = []
    for cid, g, nome in rot:
        r = aa[aa.id.str.startswith(cid + " ")].iloc[0]
        lin.append(" & ".join([g, nome + ", P21", f"{gt.sgn(r.dif_pp)} [{gt.sgn(r.ic_lo)}, {gt.sgn(r.ic_hi)}]", gt.padj(r.p_holm), f"{int(r.vit_a)}/{int(r.der_a)}", f"{r.mde_pp:.2f}"]))
    gt.escreve("tab_s24f.tex", gt.tabela(
        "Study 24: confirmatory contrasts (mean paired difference in pp [95\\% CI]; Holm-adjusted $p$ over five for each family; wins/losses of the first-named arm; MDE: minimum detectable effect "
        "at 80\\% power). RPh: adaptive weights with the faithful scoring.", "tab:s24f", r"Test & Contrast & $\Delta$ [CI] & $p$ & W/L & MDE", lin, "llcccc"))


# ------------------------------------------------------------------------------------------------ Estudo 25
def estudo25():
    if not (RES / "analise_etapa25.csv").exists():
        print("estudo25: sem analise_etapa25.csv, nada gerado")
        return
    import analisar_etapa25 as a25
    d = pd.concat([a25.referencia()] + [a25.ler25(r) for r in a25.REGS], ignore_index=True)
    t = piv(d)
    NUM["s25_nruns"] = milhar(sum(len(pd.read_csv(RES / f"etapa25_eval_{r}.csv")) for r in a25.REGS))
    NUM["s25_n_inst"] = str(len(t))
    for a in t.columns:
        arm, reg = a.split("@")
        NUM[f"s25_gap_{reg}_{arm}"] = f"{t[a].mean():.3f}"
        NUM[f"s25_g2_{reg}_{arm}"] = gt.f2(t[a].mean())
    aa = pd.read_csv(RES / "analise_etapa25.csv")
    macros_linhas("s25", aa)
    flags_hipoteses("s25", RES / "analise_etapa25_hipoteses.csv", 177)
    for _, r in pd.read_csv(RES / "analise_etapa25_folga.csv").iterrows():
        NUM[f"s25_hr_rho_{r.regime}"], NUM[f"s25_hr_p_{r.regime}"] = f"{r.rho_gap:+.2f}".replace("-", "$-$"), gt.padj(r.p_gap)
        NUM[f"s25_hr_gap_{r.regime}"] = f"{r.gap_U:.2f}"
        NUM[f"s25_hr_half_{r.regime}"], NUM[f"s25_hr_halflo_{r.regime}"], NUM[f"s25_hr_halfhi_{r.regime}"] = (f"{r.rho_metades:+.2f}".replace("-", "$-$"), f"{r.rho_metades_lo:+.2f}".replace("-", "$-$"),
                                                                                                          f"{r.rho_metades_hi:+.2f}".replace("-", "$-$"))
        NUM[f"s25_hr_ind_{r.regime}"], NUM[f"s25_hr_indp_{r.regime}"] = f"{r.rho_indep:+.2f}".replace("-", "$-$"), gt.padj(r.p_indep)
        NUM[f"s25_hr_rel_{r.regime}"] = f"{-100 * r.rel_mediana:.0f}"
    nomes = {"ref": "Reference (Study 21): simulated annealing, $1500n^2$", "acc": "Record-to-record travel, $1500n^2$", "short": "Simulated annealing, $375n^2$"}
    lin = []
    for reg in ("ref", "acc", "short"):
        cel = []
        for a in ("U", "RP", "S65"):
            c = f"{a}@{reg}"
            cel.append(f"{t[c].mean():.2f}" if c in t.columns else "---")
        dd = t[f"RP@{reg}"] - t[f"U@{reg}"]
        lin.append(" & ".join([nomes[reg]] + cel + [gt.sgn(dd.mean()), f"{int((dd < -1e-9).sum())}/{int((dd > 1e-9).sum())}"]))
    gt.escreve("tab_s25a.tex", gt.tabela(
        "Study 25: mean gap (\\%) of the frozen arms of Study 21 (pool of 21 operators) on the 51 evaluation instances in two deployment regimes and in the reference regime of Study 21, and the "
        "difference of the adaptive weights from uniform selection (pp) with its wins/losses.", "tab:s25a", r"Regime & U & RP & S65 & RP $-$ U & W/L", lin, "lccccc"))
    rot = [("H177", "Primary", "RP $-$ U, record-to-record"), ("H178", "Primary", "RP $-$ U, $375n^2$"), ("H179", "Secondary", "RP $-$ S65, record-to-record"), ("H180", "Secondary", "RP $-$ S65, $375n^2$")]
    lin = []
    for cid, g, nome in rot:
        r = aa[aa.id.str.startswith(cid + " ")].iloc[0]
        lin.append(" & ".join([g, nome, f"{gt.sgn(r.dif_pp)} [{gt.sgn(r.ic_lo)}, {gt.sgn(r.ic_hi)}]", gt.padj(r.p_holm), f"{int(r.vit_a)}/{int(r.der_a)}", f"{r.mde_pp:.2f}"]))
    gt.escreve("tab_s25f.tex", gt.tabela(
        "Study 25: confirmatory contrasts (mean paired difference in pp [95\\% CI]; Holm-adjusted $p$ over two for each family; wins/losses of the first-named arm; MDE: minimum detectable effect at "
        "80\\% power).", "tab:s25f", r"Test & Contrast & $\Delta$ [CI] & $p$ & W/L & MDE", lin, "llcccc"))
    f_eq = RES / "analise_equivalencia_orcamento.csv"
    if f_eq.exists():
        eq = pd.read_csv(f_eq)
        for tag, sub, arm in (("51", "51 instancias", "RP"), ("51S", "51 instancias", "S65"), ("X", "X (21", "RP"), ("XS", "X (21", "S65"), ("ABC", "A, B e CMT (53): 1x contra 0,25x", "RP")):
            r = eq[(eq.grupo.str.contains(sub, regex=False)) & (eq.braco == arm)].iloc[0]
            NUM[f"s25_eq{tag}_f"] = f"{r.fator:.2f}"
            NUM[f"s25_eq{tag}_ci"] = f"[{r.ic_lo:.2f}, {r.ic_hi:.2f}]"


# ------------------------------------------------------------------------------------------------ Estudo 26
def estudo26():
    if not (RES / "analise_etapa26.csv").exists():
        print("estudo26: sem analise_etapa26.csv, nada gerado")
        return
    import analisar_etapa26 as a26
    d = a26.ler26()
    t = piv(d)
    is_x = t.index.str.startswith("X-n")
    NUM["s26_nruns"] = milhar(len(pd.read_csv(RES / "etapa26_eval.csv")))
    NUM["s26_n_inst"], NUM["s26_nx"], NUM["s26_ns"] = str(len(t)), str(int(is_x.sum())), str(int((~is_x).sum()))
    for a in t.columns:
        NUM[f"s26_gap_{a}"] = f"{t[a].mean():.3f}"
        NUM[f"s26_gx_{a}"], NUM[f"s26_gs_{a}"] = gt.f2(t[a][is_x].mean()), gt.f2(t[a][~is_x].mean())
        NUM[f"s26_dU_{a}"] = gt.sgn((t[a] - t["U"]).mean())
    aa = pd.read_csv(RES / "analise_etapa26.csv")
    macros_linhas("s26", aa)
    flags_hipoteses("s26", RES / "analise_etapa26_hipoteses.csv", 181)
    neg = lambda x: f"{x:+.2f}".replace("-", "$-$")  # noqa: E731
    for _, r in pd.read_csv(RES / "analise_etapa26_folga.csv").iterrows():
        c = r.conjunto
        NUM[f"s26_rel_{c}"] = f"{-100 * r.rel_mediana:.0f}"
        NUM[f"s26_rhoind_{c}"], NUM[f"s26_pind_{c}"] = neg(r.rho_folga_indep), gt.padj(r.p_folga_indep)
        NUM[f"s26_rhosame_{c}"], NUM[f"s26_psame_{c}"] = neg(r.rho_folga_mesmas), gt.padj(r.p_folga_mesmas)
        NUM[f"s26_rhon_{c}"], NUM[f"s26_pn_{c}"] = neg(r.rho_tamanho), gt.padj(r.p_tamanho)
        NUM[f"s26_relrho_{c}"], NUM[f"s26_relp_{c}"] = neg(r.rho_rel_folga), gt.padj(r.p_rel_folga)
        NUM[f"s26_gapU_{c}"] = f"{r.gap_U:.2f}"
    f_meta = RES / "analise_etapa26_meta.csv"
    if f_meta.exists():
        for k, (_, r) in zip(("meta", "meta3", "metaS", "metaHG", "metaM", "metaUq"), pd.read_csv(f_meta).iterrows()):
            NUM[f"s26_{k}_est"], NUM[f"s26_{k}_lo"], NUM[f"s26_{k}_hi"] = gt.sgn(r.estimativa), gt.sgn(r.ic_lo), gt.sgn(r.ic_hi)
            NUM[f"s26_{k}_ci"] = f"[{gt.plano(r.ic_lo)}, {gt.plano(r.ic_hi)}]"
            NUM[f"s26_{k}_i2"], NUM[f"s26_{k}_pq"] = f"{100 * r.I2:.0f}", gt.padj(r.p_Q)
    nomes_mod = {"K": "K", "n": "n", "clientes por rota": "cpr", "log razao de iteracoes RP/U": "lr", "K | n": "Kn", "n | K": "nK", "corr(K, n)": "Kcorr"}
    for _, r in pd.read_csv(RES / "analise_etapa26_moderadores.csv").iterrows():
        k = nomes_mod.get(r.moderador)
        if k is not None and pd.notna(r.rho):
            NUM[f"s26_mod_{r.conjunto}_{k}"], NUM[f"s26_modp_{r.conjunto}_{k}"] = neg(r.rho), gt.padj(r.p)
    for _, r in pd.read_csv(RES / "analise_etapa26_uso.csv").iterrows():
        NUM[f"s26_it_{r.conjunto}_{r.braco}"] = milhar(int(round(r.iteracoes)))
    # tabelas
    nomes = [("U", "Uniform selection"), ("Uq", "Uniform, scaled removal size"), ("RP", "Adaptive weights"), ("S65", "Static mixture, 65 evaluations"), ("HG", "HG-UCB")]
    lin = []
    for a, nome in nomes:
        dd = t[a] - t["U"]
        lin.append(" & ".join([nome, f"{t[a][is_x].mean():.2f}", f"{t[a][~is_x].mean():.2f}", f"{t[a].mean():.2f}", "---" if a == "U" else gt.sgn(dd.mean()), "---" if a == "U" else f"{int((dd < -1e-9).sum())}/{int((dd > 1e-9).sum())}"]))
    gt.escreve("tab_s26a.tex", gt.tabela(
        "Study 26: mean gap (\\%) on the 98 new instances (\\num{s26_nx} instances X of 200 to 401 nodes and \\num{s26_ns} synthetic instances) of the frozen arms of Study~21; difference from uniform selection "
        "(pp, descriptive) and wins/losses over the 98 instances.", "tab:s26a", r"Arm & X (\%) & Synthetic (\%) & All (\%) & $\Delta$ vs.\ uniform & W/L", lin, "lccccc"))
    rot = [("H181", "Primary", "RP $-$ U, 98 instances"), ("H182", "Primary", "RP $-$ S65, 98 instances"), ("H183", "Primary", "HG-UCB $-$ U, 98 instances"), ("H184", "Secondary", "RP $-$ U, instances X"),
           ("H185", "Secondary", "RP $-$ U, synthetic"), ("H186", "Secondary", "Scaled removal $-$ U, 98 instances"), ("H187", "Secondary", "Spearman correlation of RP $-$ U with the number of routes"),
           ("H188", "Secondary", "Partial correlation of RP $-$ U with the number of routes given the size")]
    lin = []
    for cid, g, nome in rot:
        r = aa[aa.id.str.startswith(cid + " ")].iloc[0]
        if cid in ("H187", "H188"):
            lin.append(" & ".join([g, nome, f"$\\rho={gt.sgn(r.dif_pp)}$", gt.padj(r.p_holm), "---", "---"]))
        else:
            lin.append(" & ".join([g, nome, f"{gt.sgn(r.dif_pp)} [{gt.sgn(r.ic_lo)}, {gt.sgn(r.ic_hi)}]", gt.padj(r.p_holm), f"{int(r.vit_a)}/{int(r.der_a)}", f"{r.mde_pp:.2f}"]))
    gt.escreve("tab_s26f.tex", gt.tabela(
        "Study 26: confirmatory tests (mean paired difference in pp [95\\% CI]; Holm-adjusted $p$ over three for the primary and five for the secondary family; wins/losses of the first-named arm; MDE: "
        "minimum detectable effect at 80\\% power).", "tab:s26f", r"Test & Contrast & $\Delta$ [CI] & $p$ & W/L & MDE", lin, "llcccc"))


# ------------------------------------------------------------------------------------------------ Estudo 27
def estudo27():
    if not (RES / "analise_etapa27.csv").exists():
        print("estudo27: sem analise_etapa27.csv, nada gerado")
        return
    aa = pd.read_csv(RES / "analise_etapa27.csv")
    macros_linhas("s27", aa)
    flags_hipoteses("s27", RES / "analise_etapa27_hipoteses.csv", 189)
    NUM["s27_nruns"] = milhar(len(pd.read_csv(RES / "etapa27_eval.csv")))
    NUM["s27_nlay"] = str(int(aa[aa.id.str.startswith("H189 ")].iloc[0].n_inst))
    neg = lambda x: f"{x:+.2f}".replace("-", "$-$")  # noqa: E731
    lin = []
    for _, r in pd.read_csv(RES / "analise_etapa27_resumo.csv").iterrows():
        k = int(r.r)
        NUM[f"s27_K_{k}"], NUM[f"s27_gap_{k}"], NUM[f"s27_dif_{k}"] = f"{r.K:.0f}", f"{r.gap_U:.2f}", gt.sgn(r.dif)
        NUM[f"s27_rel_{k}"], NUM[f"s27_it_{k}"], NUM[f"s27_usorp_{k}"] = f"{-100 * r.rel:.0f}", f"{r.razao_it:.2f}", f"{r.uso_rota_pequena:.0f}"
        lin.append(" & ".join([f"{k}", f"{r.K:.0f}", f"{r.gap_U:.2f}", gt.sgn(r.dif), f"{-100 * r.rel:.0f}\\%", f"{r.razao_it:.2f}"]))
    gt.escreve("tab_s27a.tex", gt.tabela(
        "Study 27: effect of the mean route size at fixed layout and demands (\\num{s27_nlay} layouts): mean number of routes of the reference solution, mean gap of uniform selection (\\%), difference of "
        "the adaptive weights from uniform selection (pp), relative reduction of the gap and ratio of iterations.", "tab:s27a", r"Route size & Routes & Gap U (\%) & RP $-$ U & Reduction & Iterations RP/U", lin, "cccccc"))
    tr = pd.read_csv(RES / "analise_etapa27_tendencia.csv").iloc[0]
    NUM["s27_trend_rho"], NUM["s27_trend_p"] = neg(tr.rho_medio), gt.padj(tr.p_wilcoxon)
    rot = [("H189", "Primary", "RP $-$ U at $r=4$ minus at $r=25$"), ("H190", "Primary", "RP $-$ U at $r=4$"), ("H191", "Secondary", "RP $-$ U at $r=4$ minus at $r=10$"), ("H192", "Secondary", "RP $-$ U at $r=10$ minus at $r=25$")]
    lin = []
    for cid, g, nome in rot:
        r = aa[aa.id.str.startswith(cid + " ")].iloc[0]
        lin.append(" & ".join([g, nome, f"{gt.sgn(r.dif_pp)} [{gt.sgn(r.ic_lo)}, {gt.sgn(r.ic_hi)}]", gt.padj(r.p_holm), f"{int(r.vit_a)}/{int(r.der_a)}", f"{r.mde_pp:.2f}"]))
    gt.escreve("tab_s27f.tex", gt.tabela(
        "Study 27: confirmatory contrasts (mean paired difference in pp [95\\% CI] over the layouts; Holm-adjusted $p$ over two for each family; wins/losses; MDE: minimum detectable effect at 80\\% power).",
        "tab:s27f", r"Test & Contrast & $\Delta$ [CI] & $p$ & W/L & MDE", lin, "llcccc"))


# ------------------------------------------------------------------------------------------------ Estudo 28
def estudo28():
    if not (RES / "analise_etapa28.csv").exists():
        print("estudo28: sem analise_etapa28.csv, nada gerado")
        return
    import analisar_etapa28 as a28
    d = pd.concat([a28.ler28(), a28.referencia_p7()], ignore_index=True)
    if "S65@P21" not in set(d.braco):
        d = pd.concat([d, d[d.braco == "S33@P21"].assign(braco="S65@P21")], ignore_index=True)
    t = piv(d)
    NUM["s28_nruns"] = milhar(len(pd.read_csv(RES / "etapa28_eval.csv")))
    NUM["s28_n_inst"] = str(len(t))
    for a in t.columns:
        arm, pool = a.split("@")
        NUM[f"s28_gap_{pool}_{arm}"], NUM[f"s28_g2_{pool}_{arm}"] = f"{t[a].mean():.3f}", gt.f2(t[a].mean())
    aa = pd.read_csv(RES / "analise_etapa28.csv")
    macros_linhas("s28", aa)
    flags_hipoteses("s28", RES / "analise_etapa28_hipoteses.csv", 193)
    neg = lambda x: f"{x:+.2f}".replace("-", "$-$")  # noqa: E731
    r198 = aa[aa.id.str.startswith("H198 ")].iloc[0]
    NUM["s28_rho_K"], NUM["s28_n_RC"] = neg(r198.dif_pp), str(int(r198.n_inst))
    for _, r in pd.read_csv(RES / "analise_etapa28_classes.csv").iterrows():
        c = r.classe
        NUM[f"s28_cl_K_{c}"], NUM[f"s28_cl_gapU_{c}"] = f"{r.K:.0f}", f"{r.gap_U:.2f}"
        for col in ("RP_U", "S33_U", "HG_U", "RP_S33"):
            NUM[f"s28_cl_{col}_{c}"] = gt.sgn(r[col])
        NUM[f"s28_cl_it_{c}"] = f"{r.razao_it:.2f}"
    for _, r in pd.read_csv(RES / "analise_etapa28_uso.csv").iterrows():
        NUM[f"s28_it_{r.braco}"] = milhar(int(round(r.iteracoes)))
    uso28 = pd.read_csv(RES / "analise_etapa28_uso.csv").set_index("braco")
    vet28 = lambda x: [float(v) for v in x.split("|")]  # noqa: E731
    dd, rr = vet28(uso28.loc["RP", "uso_d"]), vet28(uso28.loc["RP", "uso_r"])
    NUM["s28_rp_d_lo"], NUM["s28_rp_d_hi"] = f"{min(dd):.3f}", f"{max(dd):.3f}"
    NUM["s28_rp_r_lo"], NUM["s28_rp_r_hi"] = f"{min(rr):.3f}", f"{max(rr):.3f}"
    sd, sr = vet28(uso28.loc["S33", "uso_d"]), vet28(uso28.loc["S33", "uso_r"])
    NUM["s28_s33_d_min1"], NUM["s28_s33_d_min2"] = f"{sorted(sd)[0]:.3f}", f"{sorted(sd)[1]:.3f}"
    NUM["s28_s33_r_max"] = f"{max(sr):.2f}"
    NUM["s28_it_ratio_RP"] = f"{uso28.loc['RP', 'iteracoes'] / uso28.loc['U', 'iteracoes']:.2f}"
    f_esc = RES / "etapa28_ajuste_escolhidas.json"
    if f_esc.exists():
        esc = json.loads(f_esc.read_text())
        NUM["s28_es_gap33"], NUM["s28_es_gap65"] = f"{esc['S33']['gap_ajuste']:.3f}", f"{esc['S65']['gap_ajuste']:.3f}"
        NUM["s28_es_changed65"] = "yes" if esc["S65"]["mudou"] else "no"
        tr = {x["geracao"]: x for x in esc["trajetoria"]}
        NUM["s28_es_gap0"] = f"{tr[0]['gap_ajuste']:.3f}"
    f_val = RES / "etapa28_validacao.csv"
    if f_val.exists():
        v = pd.read_csv(f_val)
        for arm, g in v[v.viavel].groupby("braco").gap_pct.mean().items():
            NUM[f"s28_val_{arm[len('s28_'):]}"] = f"{g:.3f}"
    nomes = [("U", "Uniform selection"), ("RP", "Adaptive weights"), ("S33", "Static mixture, 33 evaluations"), ("S65", "Static mixture, 65 evaluations"), ("HG", "HG-UCB")]
    lin = []
    for a, nome in nomes:
        c21, c7 = f"{a}@P21", f"{a}@P7"
        lin.append(" & ".join([nome, f"{t[c7].mean():.2f}" if c7 in t.columns else "---", f"{t[c21].mean():.2f}", "---" if a == "U" else gt.sgn((t[c21] - t["U@P21"]).mean()),
                               "---" if a == "U" else f"{int(((t[c21] - t['U@P21']) < -1e-9).sum())}/{int(((t[c21] - t['U@P21']) > 1e-9).sum())}"]))
    gt.escreve("tab_s28a.tex", gt.tabela(
        "Study 28: mean gap (\\%) on the 56 Solomon instances in the ALNS of the CVRPTW with 7 operators (Study 16, the same five seeds) and with 21; difference from uniform selection in the pool of 21 "
        "(pp, descriptive) and wins/losses.", "tab:s28a", r"Arm & 7 operators (\%) & 21 operators (\%) & $\Delta$ vs.\ uniform (21) & W/L", lin, "lcccc"))
    rot = [("H193", "Primary", "RP $-$ S33, pool of 21"), ("H194", "Primary", "Difference of (RP $-$ S33) between the pools of 21 and 7"), ("H195", "Primary", "RP $-$ U, pool of 21"),
           ("H196", "Secondary", "RP $-$ S65, pool of 21"), ("H197", "Secondary", "HG-UCB $-$ U, pool of 21"), ("H198", "Secondary", "Spearman correlation of RP $-$ U with the number of routes (classes R and RC)")]
    lin = []
    for cid, g, nome in rot:
        r = aa[aa.id.str.startswith(cid + " ")].iloc[0]
        if cid == "H198":
            lin.append(" & ".join([g, nome, f"$\\rho={gt.sgn(r.dif_pp)}$", gt.padj(r.p_holm), "---", "---"]))
        else:
            lin.append(" & ".join([g, nome, f"{gt.sgn(r.dif_pp)} [{gt.sgn(r.ic_lo)}, {gt.sgn(r.ic_hi)}]", gt.padj(r.p_holm), f"{int(r.vit_a)}/{int(r.der_a)}", f"{r.mde_pp:.2f}"]))
    gt.escreve("tab_s28f.tex", gt.tabela(
        "Study 28: confirmatory tests (mean paired difference in pp [95\\% CI]; Holm-adjusted $p$ over three for each family; wins/losses of the first-named arm; MDE: minimum detectable effect at 80\\% power).",
        "tab:s28f", r"Test & Contrast & $\Delta$ [CI] & $p$ & W/L & MDE", lin, "llcccc"))


def main():
    fd = RES / "etapa22_rrt_delta0.json"
    if fd.exists():
        NUM["s22_delta0"] = f"{json.loads(fd.read_text())['delta0']:g}"
    for f in (estudo21, estudo22, estudo23, estudo24, estudo25, estudo26, estudo27, estudo28):
        f()
