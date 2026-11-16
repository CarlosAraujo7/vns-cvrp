#!/usr/bin/env python3
"""
analisar_etapa24.py
===================
Analise do Estudo 24 (quais politicas de selecao aproveitam o conjunto de 21 operadores) conforme o PRE-REGISTRO (diario, Entrada 142). dif = gap(a) - gap(b); negativo = a melhor. Unidade = instancia
(media das seeds viaveis; 51 instancias); Wilcoxon bilateral pareado; IC95% bootstrap (10.000, semente 0); empate |dif| < 1e-9; MDE = 2,8 DP / raiz(n); equivalencia = IC90% em +-0,10 pp.
Bracos novos em P21 (pontuacao fiel, sem ajuste): PM, EXP3, UCB1, TS, EFF; referencias: U, RP, RPh (Estudos 21 e 23), S* = S129 se existe, senao S65, SRP, Uq.
  Primarios (Holm m=5)   : H167-H171  politica - U em P21 (PM, EXP3, UCB1, TS, EFF)            [significativamente NEGATIVAS]
  Secundarios (Holm m=5) : H172-H176  RPh - politica em P21 (PM, EXP3, UCB1, TS, EFF)           [significativamente NEGATIVAS: o RP fiel e melhor que a politica]
  Exploratorios (rotular): cada politica contra S*, SRP e Uq; contra RP; equivalencia; uso de operadores e iteracoes.
Uso: python estudo_aos/analisar_etapa24.py [--ensaio]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo, com_holm  # noqa: E402
import analisar_etapa23 as a23  # noqa: E402
import analisar_etapa21 as a21  # noqa: E402

RES = a21.RES
POL = ("PM", "EXP3", "UCB1", "TS", "EFF")


def ler24():
    d = pd.read_csv(RES / "etapa24_eval_P21.csv")
    d["braco"] = d.braco.str.replace("s24_P21_", "", regex=False) + "@P21"
    return d[a23.COLS]


def main(ensaio=False):
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    if ensaio:
        base = a23.fabricar(np.random.default_rng(7))
        efeito = {"PM": -0.12, "EXP3": -0.08, "UCB1": 0.20, "TS": 0.50, "EFF": 0.05}
        extra = [base[base.braco == "S33@P21"].assign(braco=f"{a}@P21", gap_pct=lambda d, e=e: d.gap_pct + e) for a, e in efeito.items()]
        dados = pd.concat([base] + extra, ignore_index=True)
    else:
        dados = pd.concat([a23.referencias(), a23.ler_novo("p21", a23.NOVOS_P21), ler24()], ignore_index=True)
        if "Uq@P21" not in set(dados.braco):
            dados = pd.concat([dados, dados[dados.braco == "U@P21"].assign(braco="Uq@P21")], ignore_index=True)
    if "S129@P21" not in set(dados.braco):
        dados = pd.concat([dados, dados[dados.braco == "S65@P21"].assign(braco="S129@P21")], ignore_index=True)
    t = pivo(dados[dados.braco.str.endswith("@P21")])
    t["S*@P21"] = t["S129@P21"]
    print(f"=== {len(t)} instancias; gap medio (%) em P21 ===\n{t.mean().round(3).sort_values().to_string()}")
    C = lambda cid, a, b, g: contraste(t, cid, a, b, rng, g)  # noqa: E731
    prim = com_holm([C(f"H{167 + i} {p}-U@P21", f"{p}@P21", "U@P21", "primario") for i, p in enumerate(POL)], 5)
    sec = com_holm([C(f"H{172 + i} RPh-{p}@P21", "RPh@P21", f"{p}@P21", "secundario") for i, p in enumerate(POL)], 5)
    expl = []
    for p in POL:
        for b in ("S*", "SRP", "Uq", "RP"):
            expl.append(C(f"{p}-{b}@P21", f"{p}@P21", f"{b}@P21", "explor"))
    ex = pd.DataFrame(expl)
    ex["p_holm"], ex["signif_005"] = ex.p_bruto, ex.p_bruto < 0.05
    r = pd.concat([prim, sec, ex], ignore_index=True)
    print("\n=== primarios (Holm m=5), secundarios (Holm m=5) e exploratorios ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    g = lambda cid: r[r.id.str.startswith(cid + " ")].iloc[0]  # noqa: E731
    neg = lambda x: bool(x.p_holm < 0.05 and x.dif_pp < 0)  # noqa: E731
    H = {}
    for i, p in enumerate(POL):
        H[f"H{167 + i} {p} - U em P21 significativamente negativo (primario, Holm m=5)"] = neg(g(f"H{167 + i}"))
    for i, p in enumerate(POL):
        H[f"H{172 + i} RPh - {p} em P21 significativamente negativo (secundario, Holm m=5)"] = neg(g(f"H{172 + i}"))
    print("\n=== hipoteses (pre-registro, Entrada 142) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de 10 (esperado no pre-registro: 5,8)")
    rows = []
    print("\n=== exploratorio: iteracoes medias e uso medio (fracao), destruicoes | reparos ===")
    for p in POL:
        h = dados[dados.braco == f"{p}@P21"]
        m = {c: pd.DataFrame(h[c].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean() for c in ("uso_d", "uso_r")}
        ud, ur = (m["uso_d"] / m["uso_d"].sum()).round(3).tolist(), (m["uso_r"] / m["uso_r"].sum()).round(3).tolist()
        print(f"  {p:5s} it {h.iteracoes.mean():8.0f}  d {ud}  r {ur}")
        rows.append(dict(braco=p, iteracoes=float(h.iteracoes.mean()), uso_d="|".join(map(str, ud)), uso_r="|".join(map(str, ur))))
    if not ensaio:
        r.to_csv(RES / "analise_etapa24.csv", index=False)
        pd.DataFrame(rows).to_csv(RES / "analise_etapa24_uso.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa24_hipoteses.csv", index=False)


if __name__ == "__main__":
    main(ensaio="--ensaio" in sys.argv)
