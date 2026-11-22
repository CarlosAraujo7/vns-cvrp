#!/usr/bin/env python3
"""
analisar_etapa29.py
===================
Analise do Estudo 29 (os bracos adaptativos AJUSTADOS no ALNS do CVRPTW com 21 operadores: Solomon, 56 instancias de 100 clientes) conforme o PRE-REGISTRO (diario, Entrada 156). dif = gap(a) - gap(b);
negativo = a melhor. Unidade = instancia (media das seeds viaveis); Wilcoxon bilateral pareado; IC95% bootstrap (10.000, semente 0); empate |dif| < 1e-9; MDE = 2,8 DP / raiz(n); equivalencia = IC90% dentro de
+-0,10 pp. Bracos (seeds 1000-1004): do Estudo 28, REUTILIZADOS sem nova execucao: U, RP (rho 0,1; segmento 100), HG (HG-UCB do Estudo 16), S33 e S65 (misturas estaticas ajustadas em 33 e 65 avaliacoes);
do Estudo 29: RPt (RP com rho e segmento ajustados por busca aleatoria de 33 avaliacoes) e HGt (HG-UCB com C e alpha ajustados do mesmo modo). Se a configuracao padrao venceu o ajuste de uma familia, o
braco nao foi executado e a coluna do Estudo 28 e copiada (RPt = RP ou HGt = HG).
  Primarios (Holm m=3)   : H199 RPt - S33 ; H200 RPt - U ; H201 RPt - RP                                                         [significativamente NEGATIVOS]
  Secundarios (Holm m=3) : H202 HGt - U ; H203 HGt - S33 ; H204 HGt - HG                                                         [significativamente NEGATIVOS]
  Exploratorios          : RPt - S65, HGt - S65, RPt - HGt, RPt - HG, HGt - RP, S33 - U (o do Estudo 28); por classe de Solomon; razao de iteracoes; uso de operadores e sua entropia; equivalencia; MDE;
                           curvas do ajuste (melhor ate agora por numero de avaliacoes) e validacao com seeds novas nas instancias de ajuste.
Uso: python estudo_aos/analisar_etapa29.py [--ensaio]   (--ensaio fabrica dados para exercitar o codigo; nada e gravado em results/)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo, com_holm  # noqa: E402
import analisar_etapa21 as a21  # noqa: E402
import analisar_etapa28 as a28  # noqa: E402

RES = a21.RES
COLS = a28.COLS
BRACOS = ("U", "RP", "HG", "S33", "S65", "RPt", "HGt")


def ler29(esc):
    """Bracos de P21 do CVRPTW: Estudo 28 (U, RP, HG, S33, S65) + Estudo 29 (RPt, HGt); copia a coluna quando a configuracao padrao venceu o ajuste."""
    d = a28.ler28()
    if "S65@P21" not in set(d.braco):
        d = pd.concat([d, d[d.braco == "S33@P21"].assign(braco="S65@P21")], ignore_index=True)
    novo = pd.read_csv(RES / "etapa29_eval.csv") if (RES / "etapa29_eval.csv").exists() else pd.DataFrame(columns=COLS)
    if len(novo):
        novo["braco"] = novo.braco.str.replace("s29_", "", regex=False) + "@P21"
        novo = novo[COLS]
    partes = [d, novo]
    for fam, nome, base in (("RP", "RPt", "RP"), ("HG", "HGt", "HG")):
        if not esc[fam]["mudou"]:
            print(f"[{nome}] a configuracao padrao venceu o ajuste: a coluna de {base} do Estudo 28 e copiada")
            partes.append(d[d.braco == f"{base}@P21"].assign(braco=f"{nome}@P21"))
    return pd.concat(partes, ignore_index=True)


def fabricar(rng):
    """ENSAIO: dados fabricados (efeitos conhecidos + ruido) para exercitar o codigo; devolve (dados, escolhidas)."""
    import run_aos_etapa14 as r14
    nomes = sorted(r14.nomes("SOLOMON"))
    efeito = {"U@P21": 0.0, "RP@P21": -0.18, "HG@P21": -0.01, "S33@P21": -0.65, "S65@P21": -0.51, "RPt@P21": -0.30, "HGt@P21": -0.40}
    out = []
    for n in nomes:
        base = 1.0 + (3.0 if "2" in re.match(r"[A-Z]+(\d)", n).group(1) else 0.0) + rng.normal(0, 0.4)
        for b, ef in efeito.items():
            for sd in range(1000, 1005):
                out.append(dict(braco=b, conjunto="SOLOMON", instancia=n, n_customers=100, seed=sd, viavel=True, gap_pct=base * (1 + ef) + rng.normal(0, 0.3), uso_d="|".join(["1"] * 12), uso_r="|".join(["1"] * 9),
                                iteracoes=1000))
    esc = {"RP": {"mudou": True}, "HG": {"mudou": True}}
    return pd.DataFrame(out), esc


def entropia_norm(v):
    v = np.asarray(v, float)
    p = v / v.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(v)))


def curvas_ajuste(rng_dummy=None):
    """Melhor gap de ajuste ate a avaliacao k (na ordem de sorteio) das buscas aleatorias de RP e HG; o ES da mistura (Estudo 28) por numero de avaliacoes; e a dispersao das 33 configuracoes."""
    linhas, disp = [], []
    for fam, nome in (("RP", "tune29_RP"), ("HG", "tune29_HG")):
        p = RES / f"etapa29_ajuste_{fam}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[(d.braco == nome) & d.viavel]
        g = d.groupby("config_id").gap_pct.mean().sort_index()
        run = g.cummin()
        for k, (cid, v) in enumerate(run.items(), start=1):
            linhas.append(dict(metodo=fam, k=k, config_id=cid, gap_ajuste=float(g[cid]), melhor_ate_k=float(v)))
        disp.append(dict(metodo=fam, padrao=float(g["r00"]), minimo=float(g.min()), mediana=float(g.median()), maximo=float(g.max()), dp=float(g.std(ddof=1))))
    esc28 = json.loads((RES / "etapa28_ajuste_escolhidas.json").read_text())
    for t in esc28["trajetoria"]:
        if t["n_avaliacoes"] <= 33:
            linhas.append(dict(metodo="S", k=t["n_avaliacoes"], config_id=f"gen{t['geracao']:02d}", gap_ajuste=float(t["gap_ajuste"]), melhor_ate_k=float(t["gap_ajuste"])))
    return pd.DataFrame(linhas), pd.DataFrame(disp)


def validacao():
    """Gap medio com seeds novas (3003-3005) nas instancias de ajuste: bracos do Estudo 28 + RPt e HGt."""
    v = pd.read_csv(RES / "etapa28_validacao.csv")
    v["braco"] = v.braco.str.replace("s28_", "", regex=False)
    if (RES / "etapa29_validacao.csv").exists():
        w = pd.read_csv(RES / "etapa29_validacao.csv")
        w["braco"] = w.braco.str.replace("s29_", "", regex=False)
        v = pd.concat([v, w], ignore_index=True)
    g = v[v.viavel].groupby("braco").gap_pct.agg(["mean", "count"]).reset_index().rename(columns={"mean": "gap_validacao", "count": "n"})
    return g.sort_values("gap_validacao")


def main(ensaio=False):
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    if ensaio:
        dados, esc = fabricar(np.random.default_rng(7))
    else:
        esc = json.loads((RES / "etapa29_ajuste_escolhidas.json").read_text())
        dados = ler29(esc)
    t = pivo(dados).copy()
    print(f"=== {len(t)} instancias; gap medio (%) por braco ===\n{t.mean().round(3).sort_values().to_string()}")
    classe = pd.Series(t.index.str.extract(r"^([A-Z]+\d)", expand=False).values, index=t.index)
    C = lambda cid, a, b, g, sub=None, **kw: contraste(t if sub is None else t[sub.values], cid, a, b, rng, g, **kw)  # noqa: E731
    prim = com_holm([C("H199 RPt-S33@P21", "RPt@P21", "S33@P21", "primario"), C("H200 RPt-U@P21", "RPt@P21", "U@P21", "primario"), C("H201 RPt-RP@P21", "RPt@P21", "RP@P21", "primario")], 3)
    sec = com_holm([C("H202 HGt-U@P21", "HGt@P21", "U@P21", "secundario"), C("H203 HGt-S33@P21", "HGt@P21", "S33@P21", "secundario"), C("H204 HGt-HG@P21", "HGt@P21", "HG@P21", "secundario")], 3)
    expl = [C("RPt-S65@P21", "RPt@P21", "S65@P21", "explor"), C("HGt-S65@P21", "HGt@P21", "S65@P21", "explor"), C("RPt-HGt@P21", "RPt@P21", "HGt@P21", "explor"), C("RPt-HG@P21", "RPt@P21", "HG@P21", "explor"),
            C("HGt-RP@P21", "HGt@P21", "RP@P21", "explor"), C("S33-U@P21", "S33@P21", "U@P21", "explor"), C("RP-U@P21", "RP@P21", "U@P21", "explor"), C("RP-S33@P21", "RP@P21", "S33@P21", "explor"),
            C("HG-U@P21", "HG@P21", "U@P21", "explor")]
    for cl in sorted(set(classe)):
        sub = classe == cl
        if int(sub.sum()) >= 6 and cl not in ("C1", "C2"):
            expl += [C(f"RPt-U@P21 {cl}", "RPt@P21", "U@P21", "explor_classe", sub, classe=cl), C(f"RPt-S33@P21 {cl}", "RPt@P21", "S33@P21", "explor_classe", sub, classe=cl),
                     C(f"HGt-U@P21 {cl}", "HGt@P21", "U@P21", "explor_classe", sub, classe=cl)]
    ex = pd.DataFrame(expl)
    ex["p_holm"], ex["signif_005"] = ex.p_bruto, ex.p_bruto < 0.05
    r = pd.concat([prim, sec, ex], ignore_index=True)
    print("\n=== primarios (Holm m=3), secundarios (Holm m=3) e exploratorios ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    g = lambda cid: r[r.id.str.startswith(cid + " ")].iloc[0]  # noqa: E731
    neg = lambda x: bool(x.p_holm < 0.05 and x.dif_pp < 0)  # noqa: E731
    H = {"H199 RPt - S33 em P21 do CVRPTW significativamente negativo (primario, Holm m=3)": neg(g("H199")),
         "H200 RPt - U em P21 do CVRPTW significativamente negativo (primario, Holm m=3)": neg(g("H200")),
         "H201 RPt - RP em P21 do CVRPTW significativamente negativo (primario, Holm m=3)": neg(g("H201")),
         "H202 HGt - U em P21 do CVRPTW significativamente negativo (secundario, Holm m=3)": neg(g("H202")),
         "H203 HGt - S33 em P21 do CVRPTW significativamente negativo (secundario, Holm m=3)": neg(g("H203")),
         "H204 HGt - HG em P21 do CVRPTW significativamente negativo (secundario, Holm m=3)": neg(g("H204"))}
    print("\n=== hipoteses (pre-registro, Entrada 156) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de 6 (esperado no pre-registro: 1,30)")
    linhas = []
    it = dados[dados.braco.isin([b + "@P21" for b in BRACOS])].groupby(["instancia", "braco"]).iteracoes.mean().unstack()
    for cl in sorted(set(classe)):
        sub = (classe == cl).values
        linhas.append(dict(classe=cl, n_inst=int(sub.sum()), gap_U=float(t["U@P21"][sub].mean()), RPt_U=float((t["RPt@P21"] - t["U@P21"])[sub].mean()), HGt_U=float((t["HGt@P21"] - t["U@P21"])[sub].mean()),
                           S33_U=float((t["S33@P21"] - t["U@P21"])[sub].mean()), RPt_S33=float((t["RPt@P21"] - t["S33@P21"])[sub].mean()), HGt_S33=float((t["HGt@P21"] - t["S33@P21"])[sub].mean()),
                           razao_it_RPt=float((it["RPt@P21"] / it["U@P21"]).reindex(t.index)[sub].mean())))
    cls = pd.DataFrame(linhas)
    print("\n=== exploratorio: por classe de Solomon (pp; medias) ===\n" + cls.round(3).to_string(index=False))
    uso = []
    for b in ("U@P21", "RP@P21", "RPt@P21", "HG@P21", "HGt@P21", "S33@P21"):
        h = dados[dados.braco == b]
        m = {c: pd.DataFrame(h[c].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean() for c in ("uso_d", "uso_r")}
        ud, ur = (m["uso_d"] / m["uso_d"].sum()).round(3).tolist(), (m["uso_r"] / m["uso_r"].sum()).round(3).tolist()
        uso.append(dict(braco=b.split("@")[0], iteracoes=float(h.iteracoes.mean()), entropia_d=entropia_norm(ud), entropia_r=entropia_norm(ur), uso_d="|".join(map(str, ud)), uso_r="|".join(map(str, ur))))
    print("\n=== exploratorio: iteracoes, entropia normalizada do uso (1 = uniforme) e uso ===\n" + pd.DataFrame(uso).to_string(index=False))
    if not ensaio:
        curva, disp = curvas_ajuste()
        print("\n=== exploratorio: dispersao dos gaps de ajuste das 33 configuracoes de cada busca aleatoria ===\n" + disp.round(4).to_string(index=False))
        print("\n=== exploratorio: melhor gap de ajuste ate k avaliacoes ===")
        for m in ("RP", "HG", "S"):
            c = curva[curva.metodo == m]
            print(f"  {m}: " + ", ".join(f"k{int(k)}:{v:.3f}" for k, v in zip(c.k, c.melhor_ate_k) if int(k) in (1, 2, 4, 5, 8, 9, 16, 17, 25, 29, 33)))
        val = validacao()
        print("\n=== exploratorio: validacao (seeds 3003-3005 nas instancias de ajuste; gap medio %) ===\n" + val.round(4).to_string(index=False))
        r.to_csv(RES / "analise_etapa29.csv", index=False)
        cls.to_csv(RES / "analise_etapa29_classes.csv", index=False)
        pd.DataFrame(uso).to_csv(RES / "analise_etapa29_uso.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa29_hipoteses.csv", index=False)
        curva.to_csv(RES / "analise_etapa29_curva_ajuste.csv", index=False)
        disp.to_csv(RES / "analise_etapa29_dispersao_ajuste.csv", index=False)
        val.to_csv(RES / "analise_etapa29_validacao.csv", index=False)


if __name__ == "__main__":
    main(ensaio="--ensaio" in sys.argv)
