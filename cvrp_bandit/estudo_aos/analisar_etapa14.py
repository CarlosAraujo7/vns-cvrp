#!/usr/bin/env python3
"""
analisar_etapa14.py
===================
Analise do Estudo 14 (CVRPTW, regime VND) conforme o PRE-REGISTRO (diario, Entrada 116). dif = gap(a) - gap(b); negativo = a melhor. Wilcoxon
bilateral pareado por instancia (media das seeds), IC95% bootstrap (10.000, semente 0), empate |dif| < 1e-9; funcoes de analisar_etapa13.

  Familia A (Holm m=14): {ucb1c, pmc, hgucbc, ts, exp3, eff, static} - U em E1 e E2
  Familia B (Holm m=2) : efeito do conjunto, U(P9) - U(P5), em E1 e E2
  Familia C (Holm m=6) : {FOC_custo, FOC_idx, FOC_rev} - U em E1 e E2
  Controle positivo (Holm m=12): 6 politicas - U no mesmo pool (D_LO, D_HI), Solomon E1
  Exploratorios: E3 (A + B + C, Holm m=11); {eff, ucb1c, hgucbc} - {FOC_custo, P5_U, static} em E1 e E2 (Holm m=18); recuperacao da perda plantada
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analisar_etapa13 as a13  # noqa: E402

RES = a13.RES
POL = ["ucb1c", "pmc", "hgucbc", "ts", "exp3", "eff", "static"]
FOC = ["FOC_custo", "FOC_idx", "FOC_rev"]


def main():
    rng = np.random.default_rng(0)
    e = pd.read_csv(RES / "etapa14_eval_v.csv")
    print(f"eval VND: {len(e)} execucoes, inviaveis {int((~e.viavel).sum())}")
    piv = {}
    for nome, nv in (("E1", 1), ("E2", 5), ("E3", 25)):
        t = a13.pivo(e[e.nivel == nv])
        t.columns = [c.replace("s14v_", "") for c in t.columns]
        piv[nome] = t
        print(f"\n=== Solomon {nome}: {len(t)} instancias; gap medio ===")
        print(t.mean().round(3).sort_values().to_string())
    A = a13.com_holm([a13.contraste(piv[n], f"{a}-U@{n}", a, "U", rng, "familiaA", cenario=n) for n in ("E1", "E2") for a in POL], 14)
    B = a13.com_holm([a13.contraste(piv[n], f"U-P5@{n}", "U", "P5_U", rng, "familiaB", cenario=n) for n in ("E1", "E2")], 2)
    C = a13.com_holm([a13.contraste(piv[n], f"{a}-U@{n}", a, "U", rng, "familiaC", cenario=n) for n in ("E1", "E2") for a in FOC], 6)
    E3 = a13.com_holm([a13.contraste(piv["E3"], f"{a}-U@E3", a, "U", rng, "explor_E3", cenario="E3") for a in POL + FOC]
                      + [a13.contraste(piv["E3"], "U-P5@E3", "U", "P5_U", rng, "explor_E3", cenario="E3")], 11)
    rng_x = np.random.default_rng(1)
    X = a13.com_holm([a13.contraste(piv[n], f"{a}-{b}@{n}", a, b, rng_x, "explor_X", cenario=n)
                      for n in ("E1", "E2") for a in ("eff", "ucb1c", "hgucbc") for b in ("FOC_custo", "P5_U", "static")], 18)
    r = pd.concat([A, B, C, E3, X], ignore_index=True)
    pd.set_option("display.width", 240)
    print("\n=== VND: familias A (m=14), B (m=2), C (m=6), E3 (m=11), X (m=18) ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa14_v.csv", index=False)

    # controle positivo
    pc = pd.read_csv(RES / "etapa14_eval_pc.csv")
    print(f"\neval controle positivo: {len(pc)} execucoes, inviaveis {int((~pc.viavel).sum())}")
    limpo = piv["E1"]["U"]
    linhas, rec = [], []
    for tag in ("D_LO", "D_HI"):
        t = a13.pivo(pc[pc.braco.str.startswith(f"s14pc_{tag}_")])
        t.columns = [c.replace(f"s14pc_{tag}_", "") for c in t.columns]
        print(f"\n=== pool {tag}: gap medio ===")
        print(t.mean().round(3).sort_values().to_string())
        perda = t["U"] - limpo.reindex(t.index)
        print(f"perda plantada (U no pool {tag} - U limpo): {perda.mean():.3f} pp")
        for a in POL[:-1]:
            linhas.append(a13.contraste(t, f"{a}-U@{tag}", a, "U", rng, "controle", pool=tag))
            idx = rng.integers(0, len(t), size=(a13.N_BOOT, len(t)))
            num, den = t["U"].values - t[a].values, perda.values
            rr = num[idx].mean(axis=1) / den[idx].mean(axis=1)
            rec.append(dict(pool=tag, politica=a, rec=num.mean() / den.mean(), rec_lo=np.percentile(rr, 2.5), rec_hi=np.percentile(rr, 97.5)))
    cp = a13.com_holm(linhas, 12)
    print("\n=== CONTROLE POSITIVO (Holm m=12) ===")
    print(cp.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    print(pd.DataFrame(rec).round(3).to_string(index=False))
    cp.to_csv(RES / "analise_etapa14_pc.csv", index=False)
    pd.DataFrame(rec).to_csv(RES / "analise_etapa14_recuperacao.csv", index=False)

    # hipoteses
    def sig_b(df):
        return (df.dif_pp < 0) & df.signif_005
    b_ok = (B.dif_pp.abs() < 0.3).all() and not B.signif_005.any()
    print(f"\nH66 (conjunto pequeno: |dif| < 0,3 e n.s. em E1 e E2): {'CONFIRMADA' if b_ok else 'nao confirmada'} ({list(B.dif_pp.round(2))})")
    print(f"H67 (eff significativamente melhor que U em E1): {bool(sig_b(A[(A.a == 'eff') & (A.cenario == 'E1')]).any())}")
    print(f"H68 (eff significativamente pior que U em E2): {bool(((A.a == 'eff') & (A.cenario == 'E2') & (A.dif_pp > 0) & A.signif_005).any())}")
    print(f"H69 (nenhuma de ucb1c, pmc, hgucbc, ts, exp3 melhor que U em E1/E2): "
          f"{int(sig_b(A[A.a.isin(['ucb1c', 'pmc', 'hgucbc', 'ts', 'exp3'])]).sum())} melhores")
    print(f"H70 (static significativamente melhor que U em E1 ou E2): {int(sig_b(A[A.a == 'static']).sum())} de 2")
    print(f"H71 (FOC_custo significativamente melhor que U em E1): {bool(sig_b(C[(C.a == 'FOC_custo') & (C.cenario == 'E1')]).any())}")
    print(f"H72 (>= 2 politicas significativamente melhores que U no pool D_HI): {int(sig_b(cp[cp.pool == 'D_HI']).sum())}")
    # H73: a melhor alternativa NAO adaptativa de cada cenario = a de menor gap medio entre static, P5_U e FOC_custo
    melhores = {n: min(("static", "P5_U", "FOC_custo"), key=lambda b: piv[n][b].mean()) for n in ("E1", "E2")}
    h73 = X[X.apply(lambda r: r.b == melhores[r.cenario], axis=1)]
    print(f"melhor alternativa nao adaptativa: {melhores}")
    print(f"H73 (nenhuma politica significativamente melhor que a MELHOR alternativa nao adaptativa): {int(sig_b(h73).sum())} contrastes melhores "
          f"(de {len(h73)}); contra qualquer das tres: {int(sig_b(X).sum())}")
    if (RES / "etapa14_explor_perm.csv").exists():
        analisa_perm(rng_perm=np.random.default_rng(2))


def analisa_perm(rng_perm):
    """POS-HOC (NAO pre-registrado; Entrada 117): politicas com o pool PERMUTADO ao acaso a cada execucao, contra U e contra a propria politica
    na ordem original, em Solomon E1 e E2 (Holm m=20)."""
    x = pd.read_csv(RES / "etapa14_explor_perm.csv")
    e = pd.read_csv(RES / "etapa14_eval_v.csv")
    d = pd.concat([x, e], ignore_index=True)
    linhas = []
    print("\n=== POS-HOC: politicas com o pool permutado ao acaso ===")
    for nome, nv in (("E1", 1), ("E2", 5)):
        t = a13.pivo(d[d.nivel == nv])
        t.columns = [c.replace("s14v_", "").replace("s14x_", "") for c in t.columns]
        print(f"{nome}: " + ", ".join(f"{k} {t[k].mean():.3f}" for k in ("U", "ucb1c", "ucb1c_perm", "hgucbc", "hgucbc_perm", "pmc", "pmc_perm", "ts", "ts_perm", "eff", "eff_perm")))
        for pol in ("ucb1c", "hgucbc", "pmc", "ts", "eff"):
            for b in ("U", pol):
                linhas.append(a13.contraste(t, f"{pol}_perm-{b}@{nome}", f"{pol}_perm", b, rng_perm, "posthoc_Q", cenario=nome))
    r = a13.com_holm(linhas, len(linhas))
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    r.to_csv(RES / "analise_etapa14_perm.csv", index=False)


if __name__ == "__main__":
    main()
