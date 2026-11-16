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
import gerar_tabelas as gt  # noqa: E402

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
    for p in ("P7", "P12", "P21"):
        lin.append(r"\multicolumn{7}{l}{\emph{" + f"{p}: {POOL_N[p]} operators, {len(t)} instances" + r"}}")
        for a, nome, nev in ARM:
            if f"{a}@{p}" not in t.columns:
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
        r = aa[aa.id.str.startswith(cid + " ")].iloc[0]
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
    for a, nome in arms:
        c = f"{a}@P21"
        if c in t.columns:
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


def main():
    for f in (estudo21, estudo22, estudo23):
        f()
