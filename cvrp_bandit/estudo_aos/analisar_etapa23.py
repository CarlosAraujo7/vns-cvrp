#!/usr/bin/env python3
"""
analisar_etapa23.py
===================
Analise do Estudo 23 (fidelidade e mecanismo da vantagem dos pesos adaptativos no conjunto de 21 operadores) conforme o PRE-REGISTRO (diario, Entrada 141).
dif = gap(a) - gap(b); negativo = a melhor. Unidade = instancia (media das seeds viaveis; 51 instancias); Wilcoxon bilateral pareado; IC95% bootstrap (10.000, semente 0); empate
|dif| < 1e-9; MDE = 2,8 DP / raiz(n); equivalencia = IC90% dentro de +-0,10 pp. Bracos de P21: U, RP, S33, S65, S129 (Estudo 21 e sua emenda), SRP (etapa21c), RPh, RPsr, Upr, Uq (novos);
de P7: U, RP, S33, S65 (Estudos 9, 12 e 20) e RPh7 (novo). S* = S129 se existe, senao S65. Se Uq nao foi executado (c* = 1), Uq = U.

  Primarios (Holm m=3)    : H154 RPh - U (P21); H155 RPh - S* (P21); H156 RPh7 - S33 (P7)                              [todas: significativamente NEGATIVAS]
  Secundarios (Holm m=6)  : H157 RPh - RP (P21) e H158 RPh7 - RP (P7) [DIFERENTES de zero, qualquer sinal]; H159 RP - SRP; H160 RP - Upr; H161 RP - Uq (P21) [negativas];
                            H162 RPsr - RP (P21) [diferente de zero]
  Terciarios (Holm m=4)   : H163 SRP - U; H164 Upr - U; H165 Uq - U; H166 RPsr - U (P21) [negativas]
  Exploratorios (rotular) : RPh e as variantes contra S33, S65, S*; Upr, SRP, Uq contra S*; equivalencia; uso de operadores e iteracoes; escolhas da selecao.
Uso: python estudo_aos/analisar_etapa23.py [--ensaio]   (--ensaio fabrica dados para exercitar o codigo; nada e gravado em results/)
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
NOVOS_P21 = ("RPh", "RPsr", "Upr", "Uq")


def ler_novo(reg, arms):
    d = pd.read_csv(RES / f"etapa23_eval_{reg}.csv")
    d["braco"] = d.braco.str.replace(f"s23_{reg}_", "", regex=False)
    d["braco"] = d.braco + ("@P7" if reg == "p7" else "@P21")
    return d[COLS]


def referencias():
    ref = a21.referencias()                                                  # P7: U, RP, HG, S33, RPt, S65
    ref = ref.assign(iteracoes=np.nan)
    e = pd.read_csv(RES / "etapa21_eval_P21.csv")
    out = [e[e.braco == f"s21_P21_{a}"][COLS].assign(braco=f"{a}@P21") for a in ("U", "RP", "S33", "S65") if (e.braco == f"s21_P21_{a}").any()]
    f = RES / "etapa21b_eval_P21.csv"
    if f.exists():
        e2 = pd.read_csv(f)
        out.append(e2[COLS].assign(braco="S129@P21"))
    f = RES / "etapa21c_eval_P21.csv"
    if f.exists():
        e3 = pd.read_csv(f)
        out.append(e3[COLS].assign(braco="SRP@P21"))
    return pd.concat([ref] + out, ignore_index=True)


def fabricar(rng):
    """ENSAIO: bracos novos fabricados a partir de S33@P7 (efeitos conhecidos + ruido)."""
    ref = a21.referencias()
    base = ref[ref.braco == "S33@P7"]
    efeito = {"U@P21": 0.10, "RP@P21": -0.10, "S33@P21": 0.20, "S65@P21": 0.12, "S129@P21": 0.0, "SRP@P21": -0.05, "RPh@P21": -0.08, "RPsr@P21": -0.07, "Upr@P21": -0.02,
              "Uq@P21": 0.0, "RPh7@P7": -0.01}
    out = [ref.assign(iteracoes=np.nan)]
    for a, e in efeito.items():
        d = base.assign(braco=a, gap_pct=base.gap_pct + e + rng.normal(0, 0.3, len(base)), iteracoes=5000)
        d["uso_d"] = "|".join(["1"] * 12)
        d["uso_r"] = "|".join(["1"] * 9)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def main(ensaio=False):
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    if ensaio:
        dados = fabricar(np.random.default_rng(7))
    else:
        esc = json.loads((RES / "etapa23_selecao.json").read_text())
        novos = [ler_novo("p21", NOVOS_P21), ler_novo("p7", ("RPh7",))]
        dados = pd.concat([referencias()] + novos, ignore_index=True)
        if "Uq@P21" not in set(dados.braco):
            print(f"Uq ausente: c* = {esc['c_estrela']} (Uq = U); coluna copiada")
            dados = pd.concat([dados, dados[dados.braco == "U@P21"].assign(braco="Uq@P21")], ignore_index=True)
    for novo, fonte in (("S129@P21", "S65@P21"),):
        if novo not in set(dados.braco):
            print(f"{novo} ausente: copiado de {fonte}")
            dados = pd.concat([dados, dados[dados.braco == fonte].assign(braco=novo)], ignore_index=True)
    t = pivo(dados)
    t["S*@P21"] = t["S129@P21"]
    print(f"=== {len(t)} instancias; gap medio (%) ===\n{t.mean().round(3).sort_values().to_string()}")
    C = lambda cid, a, b, g: contraste(t, cid, a, b, rng, g)  # noqa: E731
    prim = com_holm([C("H154 RPh-U@P21", "RPh@P21", "U@P21", "primario"), C("H155 RPh-S*@P21", "RPh@P21", "S*@P21", "primario"),
                     C("H156 RPh7-S33@P7", "RPh7@P7", "S33@P7", "primario")], 3)
    sec = com_holm([C("H157 RPh-RP@P21", "RPh@P21", "RP@P21", "secundario"), C("H158 RPh7-RP@P7", "RPh7@P7", "RP@P7", "secundario"),
                    C("H159 RP-SRP@P21", "RP@P21", "SRP@P21", "secundario"), C("H160 RP-Upr@P21", "RP@P21", "Upr@P21", "secundario"),
                    C("H161 RP-Uq@P21", "RP@P21", "Uq@P21", "secundario"), C("H162 RPsr-RP@P21", "RPsr@P21", "RP@P21", "secundario")], 6)
    ter = com_holm([C("H163 SRP-U@P21", "SRP@P21", "U@P21", "terciario"), C("H164 Upr-U@P21", "Upr@P21", "U@P21", "terciario"),
                    C("H165 Uq-U@P21", "Uq@P21", "U@P21", "terciario"), C("H166 RPsr-U@P21", "RPsr@P21", "U@P21", "terciario")], 4)
    expl = []
    for a in ("RPh", "RPsr", "Upr", "Uq", "SRP", "RP"):
        for b in ("S33", "S65", "S*"):
            expl.append(C(f"{a}-{b}@P21", f"{a}@P21", f"{b}@P21", "explor"))
    for a, b in (("RPh", "RPsr"), ("Upr", "SRP"), ("Uq", "SRP"), ("Upr", "Uq"), ("RPh", "Upr"), ("RPh", "SRP")):
        expl.append(C(f"{a}-{b}@P21", f"{a}@P21", f"{b}@P21", "explor"))
    expl.append(C("RPh7-U@P7", "RPh7@P7", "U@P7", "explor"))
    expl.append(C("RPh7-S65@P7", "RPh7@P7", "S65@P7", "explor"))
    ex = pd.DataFrame(expl)
    ex["p_holm"], ex["signif_005"] = ex.p_bruto, ex.p_bruto < 0.05
    r = pd.concat([prim, sec, ter, ex], ignore_index=True)
    print("\n=== primarios (Holm m=3), secundarios (Holm m=6), terciarios (Holm m=4) e exploratorios ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    g = lambda cid: r[r.id.str.startswith(cid + " ")].iloc[0]  # noqa: E731
    neg = lambda x: bool(x.p_holm < 0.05 and x.dif_pp < 0)  # noqa: E731
    dif0 = lambda x: bool(x.p_holm < 0.05)  # noqa: E731
    H = {"H154 RPh - U em P21 significativamente negativo (primario, Holm m=3)": neg(g("H154")),
         "H155 RPh - S* em P21 significativamente negativo (primario, Holm m=3)": neg(g("H155")),
         "H156 RPh7 - S33 em P7 significativamente negativo (primario, Holm m=3)": neg(g("H156")),
         "H157 RPh - RP em P21 diferente de zero (secundario, Holm m=6)": dif0(g("H157")),
         "H158 RPh7 - RP em P7 diferente de zero (secundario, Holm m=6)": dif0(g("H158")),
         "H159 RP - SRP em P21 significativamente negativo (secundario, Holm m=6)": neg(g("H159")),
         "H160 RP - Upr em P21 significativamente negativo (secundario, Holm m=6)": neg(g("H160")),
         "H161 RP - Uq em P21 significativamente negativo (secundario, Holm m=6)": neg(g("H161")),
         "H162 RPsr - RP em P21 diferente de zero (secundario, Holm m=6)": dif0(g("H162")),
         "H163 SRP - U em P21 significativamente negativo (terciario, Holm m=4)": neg(g("H163")),
         "H164 Upr - U em P21 significativamente negativo (terciario, Holm m=4)": neg(g("H164")),
         "H165 Uq - U em P21 significativamente negativo (terciario, Holm m=4)": neg(g("H165")),
         "H166 RPsr - U em P21 significativamente negativo (terciario, Holm m=4)": neg(g("H166"))}
    print("\n=== hipoteses (pre-registro, Entrada 141) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de 13 (esperado no pre-registro: 6,6)")
    rows = []
    print("\n=== exploratorio: iteracoes medias por execucao e uso medio (fracao), destruicoes | reparos ===")
    for a in sorted(set(dados.braco)):
        h = dados[dados.braco == a]
        if "uso_d" not in h or h.uso_d.isna().all():
            continue
        m = {c: pd.DataFrame(h[c].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean() for c in ("uso_d", "uso_r")}
        ud, ur = (m["uso_d"] / m["uso_d"].sum()).round(3).tolist(), (m["uso_r"] / m["uso_r"].sum()).round(3).tolist()
        it = h.iteracoes.mean() if "iteracoes" in h else float("nan")
        print(f"  {a:10s} it {it:8.0f}  d {ud}  r {ur}")
        rows.append(dict(braco=a, iteracoes=float(it), uso_d="|".join(map(str, ud)), uso_r="|".join(map(str, ur))))
    if not ensaio:
        print("\nselecao (calibracao de q e conjunto podado):", {k: v for k, v in esc.items() if k in ("c_estrela", "gap_ajuste_Uq", "podado_destruir", "podado_reparar")})
        r.to_csv(RES / "analise_etapa23.csv", index=False)
        pd.DataFrame(rows).to_csv(RES / "analise_etapa23_uso.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa23_hipoteses.csv", index=False)


if __name__ == "__main__":
    main(ensaio="--ensaio" in sys.argv)
