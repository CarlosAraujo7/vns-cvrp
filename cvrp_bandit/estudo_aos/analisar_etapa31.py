#!/usr/bin/env python3
"""
analisar_etapa31.py
===================
Analise do Estudo 31 (clonagem de operadores: o conjunto de 7 operadores com cada operador triplicado, 12 destruicoes + 9 reparos, no ALNS do CVRP; as 51 instancias de avaliacao do Estudo 21) conforme o
PRE-REGISTRO (diario, Entrada 160). dif = gap(a) - gap(b); negativo = a melhor. Unidade = instancia (media das seeds viaveis); Wilcoxon bilateral pareado; IC95% bootstrap (10.000, semente 0); empate |dif| < 1e-9;
MDE = 2,8 DP / raiz(n); equivalencia = IC90% dentro de +-0,10 pp. Bracos do conjunto clonado ('@C'): U, RP (padrao), S33 (mistura estatica de 21 logits, 33 avaliacoes); referencia do conjunto de 7 operadores
('@P7'): U e RP do Estudo 9 e S33 do Estudo 12 (as mesmas instancias e seeds).
  Confirmatorios (Holm m=4 nos p das diferencas; as hipoteses de equivalencia valem pelo IC90%): H210 S33@C - S33@P7 equivalente (a clonagem NAO piora a mistura ajustada) ; H211 RP@C - RP@P7 equivalente ;
                           H212 S33@C - U@C significativamente NEGATIVO (a mistura ajustada ainda vence a uniforme) ; H213 RP@C - S33@C equivalente (como em P7: +0,01 pp)
  Exploratorios          : U@C - U@P7 (controle), RP@C - U@C, S33@C - RP@P7, as DiD de (S33 - U) e de (RP - U) entre o conjunto clonado e P7, por conjunto de instancias, iteracoes e uso por classe de operador.
Uso: python estudo_aos/analisar_etapa31.py [--ensaio]   (--ensaio fabrica dados para exercitar o codigo; nada e gravado em results/)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo, com_holm  # noqa: E402
import analisar_etapa21 as a21  # noqa: E402

RES = a21.RES
COLS = a21.COLS + ["iteracoes"]


def ler31():
    ref = a21.referencias()
    ref = ref[ref.braco.isin(["U@P7", "RP@P7", "S33@P7"])][a21.COLS].assign(iteracoes=np.nan)
    d = pd.read_csv(RES / "etapa31_eval.csv")
    d["braco"] = d.braco.str.replace("s31_", "", regex=False) + "@C"
    return pd.concat([ref, d[COLS]], ignore_index=True)


def fabricar(rng):
    """ENSAIO: dados fabricados (efeitos conhecidos + ruido) para exercitar o codigo."""
    inst = [("X", f"X-n{101 + 4 * i}-k{8 + i}", 100 + 4 * i, 10) for i in range(21)] + [("SYN_VAL", f"X-syn{i:02d}", 100 + 10 * i, 5) for i in range(30)]
    efeito = {"U@P7": 0.0, "RP@P7": -0.36, "S33@P7": -0.37, "U@C": 0.0, "RP@C": -0.34, "S33@C": -0.30}
    out = []
    for c, nome, n, ns in inst:
        base = 1.0 + 0.004 * n + rng.normal(0, 0.4)
        for a, ef in efeito.items():
            for sd in range(1000, 1000 + ns):
                out.append(dict(braco=a, conjunto=c, instancia=nome, n_customers=n, seed=sd, viavel=True, gap_pct=base * (1 + 0.3 * ef) + rng.normal(0, 0.3), uso_d="|".join(["1"] * 12), uso_r="|".join(["1"] * 9),
                                iteracoes=1000))
    return pd.DataFrame(out)


def por_classe(v, k):
    """Soma o uso das copias de cada classe: o operador j pertence a classe j mod k."""
    v = np.asarray(v, float)
    return [float(v[i::k].sum()) for i in range(k)]


def main(ensaio=False):
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    dados = fabricar(np.random.default_rng(7)) if ensaio else ler31()
    t = pivo(dados).copy()
    print(f"=== {len(t)} instancias; gap medio (%) por braco ===\n{t.mean().round(3).sort_values().to_string()}")
    t["did_S33"] = (t["S33@C"] - t["U@C"]) - (t["S33@P7"] - t["U@P7"])
    t["did_RP"] = (t["RP@C"] - t["U@C"]) - (t["RP@P7"] - t["U@P7"])
    t["zero"] = 0.0
    C = lambda cid, a, b, g, sub=None, **kw: contraste(t if sub is None else t[sub.values], cid, a, b, rng, g, **kw)  # noqa: E731
    prim = com_holm([C("H210 S33@C-S33@P7", "S33@C", "S33@P7", "confirmatorio"), C("H211 RP@C-RP@P7", "RP@C", "RP@P7", "confirmatorio"), C("H212 S33@C-U@C", "S33@C", "U@C", "confirmatorio"),
                     C("H213 RP@C-S33@C", "RP@C", "S33@C", "confirmatorio")], 4)
    expl = [C("U@C-U@P7", "U@C", "U@P7", "explor"), C("RP@C-U@C", "RP@C", "U@C", "explor"), C("S33@C-RP@P7", "S33@C", "RP@P7", "explor"), C("S33@P7-U@P7", "S33@P7", "U@P7", "explor"),
            C("RP@P7-U@P7", "RP@P7", "U@P7", "explor"), C("RP@P7-S33@P7", "RP@P7", "S33@P7", "explor"), C("DiD(S33-U) C-P7", "did_S33", "zero", "explor_did"), C("DiD(RP-U) C-P7", "did_RP", "zero", "explor_did")]
    is_x = pd.Series(t.index.str.startswith("X-n"), index=t.index)
    for nome, sub in (("X", is_x), ("SYN_VAL", ~is_x)):
        for a, b in (("S33@C", "S33@P7"), ("RP@C", "RP@P7"), ("S33@C", "U@C"), ("RP@C", "S33@C")):
            expl.append(C(f"{a}-{b}@{nome}", a, b, "explor_set", sub, conjunto=nome))
    ex = pd.DataFrame(expl)
    ex["p_holm"], ex["signif_005"] = ex.p_bruto, ex.p_bruto < 0.05
    r = pd.concat([prim, ex], ignore_index=True)
    print("\n=== confirmatorios (Holm m=4 nos p) e exploratorios ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    g = lambda cid: r[r.id.str.startswith(cid + " ")].iloc[0]  # noqa: E731
    H = {"H210 S33@C - S33@P7 equivalente (IC90% em +-0,10 pp): a clonagem nao piora a mistura ajustada": bool(g("H210").equiv_010),
         "H211 RP@C - RP@P7 equivalente (IC90% em +-0,10 pp)": bool(g("H211").equiv_010),
         "H212 S33@C - U@C significativamente negativo (Holm m=4)": bool(g("H212").p_holm < 0.05 and g("H212").dif_pp < 0),
         "H213 RP@C - S33@C equivalente (IC90% em +-0,10 pp)": bool(g("H213").equiv_010)}
    print("\n=== hipoteses (pre-registro, Entrada 160) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de 4 (esperado no pre-registro: 1,70)")
    uso = []
    for b in ("U@C", "RP@C", "S33@C"):
        h = dados[dados.braco == b]
        mm = {c: pd.DataFrame(h[c].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean() for c in ("uso_d", "uso_r")}
        ud, ur = mm["uso_d"] / mm["uso_d"].sum(), mm["uso_r"] / mm["uso_r"].sum()
        uso.append(dict(braco=b, iteracoes=float(h.iteracoes.mean()), classe_d="|".join(f"{x:.3f}" for x in por_classe(ud, 4)), classe_r="|".join(f"{x:.3f}" for x in por_classe(ur, 3)),
                        copia_d_max=float(max(ud)), copia_d_min=float(min(ud))))
    uso = pd.DataFrame(uso)
    print("\n=== exploratorio: iteracoes e uso por classe de operador (soma das 3 copias) ===\n" + uso.to_string(index=False))
    if not ensaio:
        f = RES / "etapa31_selecao.json"
        if f.exists():
            esc = json.loads(f.read_text())
            print("\n=== exploratorio: curva do ES (gap de ajuste do incumbente por n. de avaliacoes) ===\n" + ", ".join(f"{x['n_avaliacoes']}:{x['gap_ajuste']:.3f}" for x in esc["trajetoria"]))
        r.to_csv(RES / "analise_etapa31.csv", index=False)
        uso.to_csv(RES / "analise_etapa31_uso.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa31_hipoteses.csv", index=False)


if __name__ == "__main__":
    main(ensaio="--ensaio" in sys.argv)
