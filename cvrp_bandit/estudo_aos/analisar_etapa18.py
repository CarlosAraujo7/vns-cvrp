#!/usr/bin/env python3
"""
analisar_etapa18.py
===================
Analise do Estudo 18 (politicas com estado e esforco de ajuste igual, ALNS no CVRPTW; 56 instancias de Solomon, 10 seeds) conforme o
PRE-REGISTRO (diario, Entrada 127). dif = gap(a) - gap(b); negativo = a melhor. Unidade = instancia (media das seeds); Wilcoxon bilateral
pareado; IC95% bootstrap (10.000, semente 0); empate |dif| < 1e-9; Holm POR FAMILIA. Referencias do Estudo 16 (U, RP16, static16, fixrep, ucb1,
pm, hgucb, ts, exp3, eff) lidas de etapa16_eval_a.csv (mesmas instancias e seeds 1000-1009).

  Primario (sem correcao)     : MELHOR online com estado - MELHOR offline, ambos escolhidos pelo gap de VALIDACAO (seeds novas nas instancias de
                                ajuste; nunca pelos dados de avaliacao): online em {ctxucb, linucb, qlearn}, offline em {static18, sched, loglin}
  Familia A (Holm m=3)        : static18 - static16, RP18 - RP16, static18 - RP18                                   (esforco de ajuste igual)
  Familia B (Holm m=2)        : sched - static18, loglin - static18                                                  (estrutura offline)
  Familia C (Holm m=6)        : {ctxucb, linucb, qlearn} - {U, RP18}                                                 (estado vs sem estado)
  Familia D (Holm m=15)       : {RP18, tilt, ctxucb, linucb, qlearn} - {static18, sched, loglin}                     (online vs melhor offline)
  Exploratorios (rotular)     : por classe de Solomon; equivalencia (IC90% em +-0,10pp); MDE; otimismo do ajuste (ajuste -> validacao -> avaliacao);
                                uso de operadores por fase; ganho de sched e loglin sobre static18 dentro e fora da amostra de ajuste.
Uso: python estudo_aos/analisar_etapa18.py            |   python estudo_aos/analisar_etapa18.py --ensaio   (dados fabricados, sem escrever)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import holm, contraste, pivo, com_holm, classe  # noqa: E402

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
ONLINE_EST = ["ctxucb", "linucb", "qlearn"]
OFFLINE = ["static18", "sched", "loglin"]
ONLINE_D = ["RP18", "tilt", "ctxucb", "linucb", "qlearn"]
REF16 = {"RP": "RP16", "static": "static16"}


def tabela_gaps(e16, e18):
    t16 = pivo(e16)
    t16.columns = [REF16.get(c.replace("s16a_", ""), c.replace("s16a_", "")) for c in t16.columns]
    t18 = pivo(e18)
    t18.columns = [c.replace("s18_", "") for c in t18.columns]
    return t16.join(t18, how="inner").dropna()


def sig_neg(r):
    return bool(r.p_holm < 0.05 and r.dif_pp < 0)


def analisar(e16, e18, valid, escolhas, rng, escrever=True):
    t = tabela_gaps(e16, e18)
    print(f"=== {len(t)} instancias de Solomon; gap medio (%) ===")
    print(t.mean().round(3).sort_values().to_string())
    # primario: escolha pelos gaps de validacao
    v = valid[valid.viavel].groupby("braco").gap_pct.mean()
    v.index = [i.replace("s18_", "") for i in v.index]
    on = min(ONLINE_EST, key=lambda a: v[a])
    off = min(OFFLINE, key=lambda a: v[a])
    print(f"\nvalidacao (instancias de ajuste, seeds novas): {v.round(4).sort_values().to_dict()}")
    print(f"primario: melhor online com estado = {on}, melhor offline = {off} (pelo gap de validacao)")
    prim = pd.DataFrame([contraste(t, f"{on}-{off}", on, off, rng, "primario")])
    prim["p_holm"], prim["signif_005"] = prim.p_bruto, prim.p_bruto < 0.05
    A = com_holm([contraste(t, "static18-static16", "static18", "static16", rng, "A"),
                  contraste(t, "RP18-RP16", "RP18", "RP16", rng, "A"),
                  contraste(t, "static18-RP18", "static18", "RP18", rng, "A")], 3)
    B = com_holm([contraste(t, f"{a}-static18", a, "static18", rng, "B") for a in ("sched", "loglin")], 2)
    C = com_holm([contraste(t, f"{a}-{b}", a, b, rng, "C") for a in ONLINE_EST for b in ("U", "RP18")], 6)
    D = com_holm([contraste(t, f"{a}-{b}", a, b, rng, "D") for a in ONLINE_D for b in OFFLINE], 15)
    r = pd.concat([prim, A, B, C, D], ignore_index=True)
    print("\n=== primario, A (m=3), B (m=2), C (m=6), D (m=15) ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))

    g = lambda fam, cid: r[(r.grupo == fam) & (r.id == cid)].iloc[0]  # noqa: E731
    H = {
        "H96 static18 - static16 nao significativo": not g("A", "static18-static16").signif_005,
        "H97 RP18 - RP16 nao significativamente negativo": not sig_neg(g("A", "RP18-RP16")),
        "H98 static18 - RP18 significativamente negativo": sig_neg(g("A", "static18-RP18")),
        "H99 sched - static18 significativamente negativo": sig_neg(g("B", "sched-static18")),
        "H100 loglin - static18 significativamente negativo": sig_neg(g("B", "loglin-static18")),
        "H101 algum {ctxucb,linucb,qlearn} - U significativamente negativo": any(sig_neg(g("C", f"{a}-U")) for a in ONLINE_EST),
        "H102 algum {ctxucb,linucb,qlearn} - RP18 significativamente negativo": any(sig_neg(g("C", f"{a}-RP18")) for a in ONLINE_EST),
        "H103 nenhum dos 15 contrastes online - offline significativamente negativo": not any(sig_neg(x) for _, x in D.iterrows()),
        "H104 tilt - static18 significativamente negativo": sig_neg(g("D", "tilt-static18")),
        "H105 primario nao significativamente negativo": not sig_neg(prim.iloc[0]),
    }
    print("\n=== hipoteses (pre-registro, Entrada 127) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de {len(H)} (esperado no pre-registro: 5,0)")
    n_melhor = int(sum(sig_neg(x) for _, x in D.iterrows()))
    n_pior = int(sum((x.p_holm < 0.05 and x.dif_pp > 0) for _, x in D.iterrows()))
    print(f"  familia D: {n_melhor} contrastes online - offline significativamente melhores, {n_pior} significativamente piores, de 15")

    # exploratorio: por classe
    cl = pd.Series({n: classe(n) for n in t.index})
    pares = [("static18", "static16"), ("RP18", "RP16"), ("static18", "RP18"), ("sched", "static18"), ("loglin", "static18"),
             ("tilt", "static18"), (on, off), ("ctxucb", "RP18"), ("linucb", "RP18"), ("qlearn", "RP18")]
    linhas = []
    for c in sorted(cl.unique()):
        s = t[cl == c]
        for a, b in pares:
            linhas.append(contraste(s, f"{a}-{b}@{c}", a, b, rng, "explor_classe", classe=c))
    ex = pd.DataFrame(linhas)
    print("\n=== exploratorio: por classe de Solomon (sem correcao; rotular) ===")
    print(ex[["id", "n_inst", "dif_pp", "ic_lo", "ic_hi", "vit_a", "der_a", "p_bruto"]].round(4).to_string(index=False))

    # exploratorio: otimismo do ajuste (ajuste -> validacao -> avaliacao) e ganho dentro/fora da amostra de sched e loglin
    linhas = []
    for a in BRACOS:
        if escolhas is None or a not in escolhas:
            continue
        linhas.append(dict(braco=a, gap_ajuste=escolhas[a]["gap_tune"], gap_validacao=float(v[a]), gap_avaliacao=float(t[a].mean())))
    ot = pd.DataFrame(linhas)
    if len(ot):
        ot["otimismo_ajuste_validacao"] = ot.gap_validacao - ot.gap_ajuste
        print("\n=== exploratorio: gap do ajuste (33 avaliacoes, melhor configuracao), da validacao (seeds novas, mesmas instancias) e da avaliacao ===")
        print(ot.round(4).to_string(index=False))
        if "static18" in escolhas:
            for a in ("sched", "loglin"):
                if a in escolhas:
                    print(f"   {a} - static18: ajuste {escolhas[a]['gap_tune'] - escolhas['static18']['gap_tune']:+.4f}  validacao "
                          f"{v[a] - v['static18']:+.4f}  avaliacao {t[a].mean() - t['static18'].mean():+.4f} pp")

    # exploratorio: uso de operadores por fase (terços do orcamento)
    print("\n=== exploratorio: fracao de uso por fase (3 tercos do orcamento): destruicoes (aleat., pior, relac., rota) | reparos (guloso, regret2, barato) ===")
    for a in BRACOS:
        h = e18[e18.braco == f"s18_{a}"]
        if not len(h):
            continue
        def fase_media(col):
            m = np.array([[[float(x) for x in ph.split("|")] for ph in s.split(";")] for s in h[col]]).mean(axis=0)
            return (m / m.sum(axis=1, keepdims=True)).round(2).tolist()
        print(f"  {a:9s} D {fase_media('uso_d_fase')}  R {fase_media('uso_r_fase')}")

    if escrever:
        r.to_csv(RES / "analise_etapa18.csv", index=False)
        ex.to_csv(RES / "analise_etapa18_classe.csv", index=False)
        ot.to_csv(RES / "analise_etapa18_ajuste.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa18_hipoteses.csv", index=False)
    return r, H


BRACOS = ["static18", "RP18", "tilt", "sched", "loglin", "ctxucb", "linucb", "qlearn"]


def ensaio(e16):
    """Dados FABRICADOS a partir do Estudo 16 so para exercitar o codigo (nao e resultado; nada e escrito)."""
    rng = np.random.default_rng(1)
    mapa = {"static18": "static", "RP18": "RP", "tilt": "RP", "sched": "static", "loglin": "static", "ctxucb": "U", "linucb": "pm", "qlearn": "ts"}
    partes = []
    for novo, velho in mapa.items():
        d = e16[e16.braco == f"s16a_{velho}"].copy()
        d["braco"] = f"s18_{novo}"
        d["gap_pct"] = d.gap_pct + rng.normal(0, 0.05, len(d))
        d["uso_d_fase"] = "10|20|30|40;10|20|30|40;10|20|30|40"
        d["uso_r_fase"] = "10|20|30;10|20|30;10|20|30"
        partes.append(d)
    e18 = pd.concat(partes, ignore_index=True)
    valid = pd.DataFrame({"braco": [f"s18_{a}" for a in BRACOS for _ in range(4)], "viavel": True,
                          "gap_pct": [rng.normal(1.0 + 0.1 * i, 0.01) for i, _ in enumerate(BRACOS) for _ in range(4)]})
    return e18, valid


def main():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    e16 = pd.read_csv(RES / "etapa16_eval_a.csv")
    if "--ensaio" in sys.argv:
        e18, valid = ensaio(e16)
        analisar(e16, e18, valid, None, rng, escrever=False)
        return
    e18 = pd.read_csv(RES / "etapa18_eval.csv")
    valid = pd.read_csv(RES / "etapa18_validacao.csv")
    escolhas = json.loads((RES / "etapa18_tuning_escolhidas.json").read_text())
    print(f"eval Estudo 18: {len(e18)} execucoes, inviaveis {int((~e18.viavel).sum())}")
    analisar(e16, e18, valid, escolhas, rng)


if __name__ == "__main__":
    main()
