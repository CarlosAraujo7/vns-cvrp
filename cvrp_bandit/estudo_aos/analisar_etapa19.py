#!/usr/bin/env python3
"""
analisar_etapa19.py
===================
Analise do Estudo 19 (replicacao confirmatoria do indicio do Estudo 18: `tilt` = pesos adaptativos de Ropke e Pisinger modulados pela mistura estatica
ajustada, contra a propria mistura `static18`) conforme o PRE-REGISTRO (diario, Entrada 129). dif = gap(a) - gap(b); negativo = a melhor. Unidade =
instancia (media das seeds); Wilcoxon bilateral pareado; IC95% bootstrap (10.000, semente 0); empate |dif| < 1e-9. Conjuntos: SYN (48 sinteticas novas,
seeds 5000-5009) e SOL (56 de Solomon, seeds novas 2000-2009).

  Primario (sem correcao, alfa 0,05) : tilt - static18 em SYN                                                             -> H106
  Secundarios (Holm m=3)             : tilt - static18 em SOL; static18 - RP18 em SYN; static18 - RP18 em SOL              -> H107, H108, H109
  Exploratorios (rotular)            : tilt - U, static18 - U, RP18 - U, tilt - RP18 em cada conjunto; combinado (Solomon com as 20 seeds do Estudo 18 e do
                                       19 + SYN: 104 instancias): tilt - static18                                         -> H110 (IC95% exclui zero); por classe;
                                       uso de operadores por terco do orcamento.
Uso: python estudo_aos/analisar_etapa19.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo, com_holm, classe  # noqa: E402

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"


def main():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    e = pd.read_csv(RES / "etapa19_eval.csv")
    e18 = pd.read_csv(RES / "etapa18_eval.csv")
    print(f"eval Estudo 19: {len(e)} execucoes, inviaveis {int((~e.viavel).sum())}")
    e["braco"] = e.braco.str.replace("s19_", "", regex=False)
    syn = pivo(e[e.conjunto == "SYN_VAL_W"])
    sol = pivo(e[e.conjunto == "SOLOMON"])
    print(f"=== SYN_VAL_W: {len(syn)} instancias; gap medio (%) ===\n{syn.mean().round(3).sort_values().to_string()}")
    print(f"=== SOLOMON (seeds novas): {len(sol)} instancias; gap medio (%) ===\n{sol.mean().round(3).sort_values().to_string()}")
    prim = pd.DataFrame([contraste(syn, "tilt-static18@SYN", "tilt", "static18", rng, "primario")])
    prim["p_holm"], prim["signif_005"] = prim.p_bruto, prim.p_bruto < 0.05
    sec = com_holm([contraste(sol, "tilt-static18@SOL", "tilt", "static18", rng, "secundario"),
                    contraste(syn, "static18-RP18@SYN", "static18", "RP18", rng, "secundario"),
                    contraste(sol, "static18-RP18@SOL", "static18", "RP18", rng, "secundario")], 3)
    expl = []
    for nome, t in (("SYN", syn), ("SOL", sol)):
        for a, b in (("tilt", "U"), ("static18", "U"), ("RP18", "U"), ("tilt", "RP18")):
            expl.append(contraste(t, f"{a}-{b}@{nome}", a, b, rng, "explor"))
    ex = pd.DataFrame(expl)
    ex["p_holm"], ex["signif_005"] = ex.p_bruto, ex.p_bruto < 0.05
    # combinado: Solomon com as seeds do Estudo 18 (1000-1009) e do 19 (2000-2009) + SYN
    s18 = e18[e18.braco.isin(["s18_tilt", "s18_static18"])].copy()
    s18["braco"] = s18.braco.str.replace("s18_", "", regex=False)
    comb_sol = pd.concat([s18, e[(e.conjunto == "SOLOMON") & e.braco.isin(["tilt", "static18"])]])
    comb = pd.concat([pivo(comb_sol), syn[["tilt", "static18"]]])
    cb = pd.DataFrame([contraste(comb, "tilt-static18@combinado", "tilt", "static18", rng, "combinado")])
    cb["p_holm"], cb["signif_005"] = cb.p_bruto, cb.p_bruto < 0.05
    r = pd.concat([prim, sec, ex, cb], ignore_index=True)
    print("\n=== primario, secundarios (Holm m=3), exploratorios e combinado (104 instancias) ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    g = lambda cid: r[r.id == cid].iloc[0]  # noqa: E731
    neg = lambda x: bool(x.p_holm < 0.05 and x.dif_pp < 0)  # noqa: E731
    cbr = g("tilt-static18@combinado")
    H = {
        "H106 tilt - static18 em SYN significativamente negativo (primario)": neg(g("tilt-static18@SYN")),
        "H107 tilt - static18 em SOL (seeds novas) significativamente negativo (Holm)": neg(g("tilt-static18@SOL")),
        "H108 static18 - RP18 em SYN significativamente negativo (Holm)": neg(g("static18-RP18@SYN")),
        "H109 static18 - RP18 em SOL significativamente negativo (Holm)": neg(g("static18-RP18@SOL")),
        "H110 (explor.) tilt - static18 combinado: IC95% exclui zero e e negativo": bool(cbr.ic_hi < 0),
    }
    print("\n=== hipoteses (pre-registro, Entrada 129) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmatorias confirmadas: {sum(list(H.values())[:4])} de 4 (esperado no pre-registro: 2,25); exploratoria H110: {list(H.values())[4]}")

    linhas = []
    for nome, t in (("SOL", sol), ("SYN", syn)):
        c_ = pd.Series({n: (classe(n) if nome == "SOL" else n.split("-")[-1]) for n in t.index})
        for c in sorted(c_.unique()):
            linhas.append(contraste(t[c_ == c], f"tilt-static18@{nome}:{c}", "tilt", "static18", rng, "explor_classe", classe=c, conjunto=nome))
    exc = pd.DataFrame(linhas)
    print("\n=== exploratorio: tilt - static18 por classe (sem correcao; rotular) ===")
    print(exc[["id", "n_inst", "dif_pp", "ic_lo", "ic_hi", "vit_a", "der_a", "p_bruto"]].round(4).to_string(index=False))

    print("\n=== exploratorio: uso por terco do orcamento (destruicoes | reparos), fracao media ===")
    for a in ("static18", "tilt", "RP18"):
        h = e[e.braco == a]
        for col, k in (("uso_d_fase", 4), ("uso_r_fase", 3)):
            m = np.array([[[float(x) for x in p.split("|")] for p in s.split(";")] for s in h[col]]).mean(axis=0)
            print(f"  {a:9s} {col[4]}: {(m / m.sum(axis=1, keepdims=True)).round(2).tolist()}")
    # POS HOC (nao pre-registrado; acrescentado depois de ver os resultados): estimativa combinada SO da replicacao (Solomon com seeds novas e as
    # 48 sinteticas novas, 104 instancias), sem os dados do Estudo 18 que sugeriram o efeito; entra no fim para nao alterar nenhuma linha anterior
    rep = pd.concat([sol[["tilt", "static18"]], syn[["tilt", "static18"]]])
    ph = pd.DataFrame([contraste(rep, "tilt-static18@replicacao", "tilt", "static18", rng, "posthoc")])
    ph["p_holm"], ph["signif_005"] = ph.p_bruto, ph.p_bruto < 0.05
    print("\n=== POS HOC: tilt - static18, so os dados da replicacao (104 instancias) ===")
    print(ph.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    r = pd.concat([r, ph], ignore_index=True)
    r.to_csv(RES / "analise_etapa19.csv", index=False)
    exc.to_csv(RES / "analise_etapa19_classe.csv", index=False)
    pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa19_hipoteses.csv", index=False)


if __name__ == "__main__":
    main()
