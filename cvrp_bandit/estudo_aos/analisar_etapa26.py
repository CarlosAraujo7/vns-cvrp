#!/usr/bin/env python3
"""
analisar_etapa26.py
===================
Analise do Estudo 26 (replica das conclusoes de P21 em 98 instancias NOVAS: X_FRESH, 38 instancias X de 200 a 401 clientes, e SYN_REP, 60 sinteticas) conforme o PRE-REGISTRO (diario, Entrada 146).
dif = gap(a) - gap(b); negativo = a melhor. Unidade = instancia (media das 2 seeds viaveis); Wilcoxon bilateral pareado; IC95% bootstrap (10.000, semente 0); empate |dif| < 1e-9;
MDE = 2,8 DP / raiz(n); equivalencia = IC90% dentro de +-0,10 pp. Bracos congelados do Estudo 21 (e do Estudo 23): U, Uq (uniforme com a faixa de remocao escalada por c*), RP, S65, HG.
  Primarios (Holm m=3)   : H181 RP - U ; H182 RP - S65 ; H183 HG - U                       (todas as 98 instancias; H181 e H182 "significativamente NEGATIVAS", H183 tambem)
  Secundarios (Holm m=5) : H184 RP - U em X_FRESH ; H185 RP - U em SYN_REP ; H186 Uq - U ; H187 Spearman(RP - U, K) ; H188 Spearman parcial(RP - U, K | n)   (todas "significativamente NEGATIVAS";
                           K = numero de rotas da solucao de referencia do HGS)
  Exploratorios          : demais pares de bracos; por conjunto e por classe de tamanho; reducao relativa (RP - U)/gap da uniforme; correlacao de RP - U com a folga medida SEM as execucoes da uniforme
                           (gap medio de HG e S65) e com o tamanho; metanalise de efeito fixo de RP - U com as 51 instancias do Estudo 21; equivalencia; uso de operadores e iteracoes.
Uso: python estudo_aos/analisar_etapa26.py [--ensaio]   (--ensaio fabrica dados para exercitar o codigo; nada e gravado em results/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo, com_holm  # noqa: E402
import analisar_etapa21 as a21  # noqa: E402

RES = a21.RES
COLS = a21.COLS + ["iteracoes"]
ARMS = ("U", "Uq", "RP", "S65", "HG")


def ler26():
    d = pd.read_csv(RES / "etapa26_eval.csv")
    d["braco"] = d.braco.str.replace("s26_", "", regex=False)
    return d[COLS]


def ler_k():
    """Numero de rotas da solucao de referencia (coluna rotas_ref de instances/<conjunto>/metadata.csv) de cada instancia nova."""
    out = {}
    for c in ("X_FRESH", "SYN_REP"):
        m = pd.read_csv(RES.parent.parent / "instances" / c / "metadata.csv")
        out.update(dict(zip(m.nome, m.rotas_ref)))
    return pd.Series(out, dtype=float)


def fabricar(rng):
    """ENSAIO: dados fabricados (efeitos conhecidos + ruido) para exercitar o codigo; devolve (dados, K por instancia)."""
    inst = [("X_FRESH", f"X-n{200 + 5 * i}-k{10 + i}", 199 + 5 * i, 10 + 2 * i) for i in range(38)] + [("SYN_REP", f"X-synr{k + 1:02d}-n{n + 1}", n, int(n / (4 + k))) for n in (100, 200, 300) for k in range(20)]
    efeito = {"U": 0.0, "Uq": -0.10, "RP": -0.25, "S65": 0.02, "HG": -0.03}
    out = []
    for c, nome, n, kk in inst:
        base = 1.0 + 0.004 * n + rng.normal(0, 0.5)
        for a in ARMS:
            for sd in (1000, 1001):
                ef = efeito[a] - (0.004 * kk if a == "RP" else 0.0)
                out.append(dict(braco=a, conjunto=c, instancia=nome, n_customers=n, seed=sd, viavel=True, gap_pct=base * (1 + ef) + rng.normal(0, 0.3), uso_d="|".join(["1"] * 12),
                                uso_r="|".join(["1"] * 9), iteracoes=1000))
    return pd.DataFrame(out), pd.Series({nome: float(kk) for _c, nome, _n, kk in inst})


def parcial(x, y, z):
    """Correlacao parcial de postos de x e y dado z e o p bilateral (t com n - 3 graus de liberdade)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    r = lambda a, b: np.corrcoef(a, b)[0, 1]  # noqa: E731
    rxy, rxz, ryz = r(rx, ry), r(rx, rz), r(ry, rz)
    rp = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    n = len(x)
    t = rp * np.sqrt((n - 3) / (1 - rp ** 2))
    return float(rp), float(2 * stats.t.sf(abs(t), n - 3))


def linha_corr(cid, rho, p, n):
    """Linha no formato de `contraste` para um teste de correlacao (dif_pp guarda o rho)."""
    return dict(grupo="secundario", id=cid, a="RP-U", b="K", n_inst=n, dif_pp=rho, ic_lo=float("nan"), ic_hi=float("nan"), vit_a=0, der_a=0, emp=0, p_bruto=p, equiv_010=False, mde_pp=float("nan"))


def meta_efeito_fixo(difs):
    """Metanalise de efeito fixo (inverso da variancia) das medias de cada conjunto de instancias; (estimativa, IC95% lo, hi, Q, p de Q, I2)."""
    est = np.array([d.mean() for d in difs])
    var = np.array([d.var(ddof=1) / len(d) for d in difs])
    w = 1 / var
    pooled = (w * est).sum() / w.sum()
    se = np.sqrt(1 / w.sum())
    q = float((w * (est - pooled) ** 2).sum())
    df = len(difs) - 1
    return float(pooled), float(pooled - 1.96 * se), float(pooled + 1.96 * se), q, float(stats.chi2.sf(q, df)), float(max(0.0, (q - df) / q)) if q > 0 else 0.0


def main(ensaio=False):
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    dados, kser = fabricar(np.random.default_rng(7)) if ensaio else (ler26(), ler_k())
    t = pivo(dados).copy()
    K = kser.reindex(t.index)
    n_cl = dados.groupby("instancia").n_customers.first().reindex(t.index)
    is_x = pd.Series(t.index.str.startswith("X-n"), index=t.index)
    nx, ns = int(is_x.sum()), int((~is_x).sum())
    print(f"=== {len(t)} instancias ({nx} de X_FRESH, {ns} de SYN_REP); gap medio (%) por braco ===\n{t.mean().round(3).sort_values().to_string()}")
    t["zero"] = 0.0
    C = lambda cid, a, b, g, sub=None, **kw: contraste(t if sub is None else t[sub.values], cid, a, b, rng, g, **kw)  # noqa: E731
    prim = com_holm([C("H181 RP-U", "RP", "U", "primario"), C("H182 RP-S65", "RP", "S65", "primario"), C("H183 HG-U", "HG", "U", "primario")], 3)
    dif0 = t["RP"] - t["U"]
    rho_k, p_k = stats.spearmanr(K, dif0)
    rho_kn, p_kn = parcial(K.values, dif0.values, n_cl.values.astype(float))
    sec = com_holm([C("H184 RP-U@X_FRESH", "RP", "U", "secundario", is_x), C("H185 RP-U@SYN_REP", "RP", "U", "secundario", ~is_x), C("H186 Uq-U", "Uq", "U", "secundario"),
                    linha_corr("H187 Spearman(RP-U, K) [rho]", float(rho_k), float(p_k), len(t)), linha_corr("H188 Spearman parcial(RP-U, K | n) [rho]", rho_kn, p_kn, len(t))], 5)
    expl = []
    for a, b in (("S65", "U"), ("Uq", "RP"), ("HG", "RP"), ("Uq", "S65"), ("S65", "HG")):
        expl.append(C(f"{a}-{b}", a, b, "explor"))
    for nome, sub in (("X_FRESH", is_x), ("SYN_REP", ~is_x)):
        for a, b in (("RP", "S65"), ("HG", "U"), ("Uq", "U"), ("S65", "U"), ("Uq", "RP")):
            expl.append(C(f"{a}-{b}@{nome}", a, b, "explor_set", sub, conjunto=nome))
    classes = {"SYN_REP n=100": (~is_x) & (n_cl == 100), "SYN_REP n=200": (~is_x) & (n_cl == 200), "SYN_REP n=300": (~is_x) & (n_cl == 300)}
    xq = pd.qcut(n_cl[is_x], 3, labels=["X_FRESH menores", "X_FRESH medias", "X_FRESH maiores"])
    for lab in ("X_FRESH menores", "X_FRESH medias", "X_FRESH maiores"):
        classes[lab] = pd.Series(False, index=t.index) | (xq == lab).reindex(t.index).fillna(False)
    for lab, sub in classes.items():
        if int(sub.sum()) >= 6:
            expl.append(C(f"RP-U@{lab}", "RP", "U", "explor_classe", sub, classe=lab))
    ex = pd.DataFrame(expl)
    ex["p_holm"], ex["signif_005"] = ex.p_bruto, ex.p_bruto < 0.05
    r = pd.concat([prim, sec, ex], ignore_index=True)
    print("\n=== primarios (Holm m=3), secundarios (Holm m=5) e exploratorios ===")
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    g = lambda cid: r[r.id.str.startswith(cid + " ")].iloc[0]  # noqa: E731
    neg = lambda x: bool(x.p_holm < 0.05 and x.dif_pp < 0)  # noqa: E731
    H = {"H181 RP - U significativamente negativo nas 98 instancias novas (primario, Holm m=3)": neg(g("H181")),
         "H182 RP - S65 significativamente negativo nas 98 instancias novas (primario, Holm m=3)": neg(g("H182")),
         "H183 HG - U significativamente negativo nas 98 instancias novas (primario, Holm m=3)": neg(g("H183")),
         "H184 RP - U significativamente negativo em X_FRESH (secundario, Holm m=5)": neg(g("H184")),
         "H185 RP - U significativamente negativo em SYN_REP (secundario, Holm m=5)": neg(g("H185")),
         "H186 Uq - U significativamente negativo nas 98 instancias novas (secundario, Holm m=5)": neg(g("H186")),
         "H187 correlacao de Spearman de RP - U com o numero de rotas K significativamente negativa (secundario, Holm m=5)": neg(g("H187")),
         "H188 correlacao parcial de RP - U com K dado o tamanho n significativamente negativa (secundario, Holm m=5)": neg(g("H188"))}
    print("\n=== hipoteses (pre-registro, Entrada 146) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de 8 (esperado no pre-registro: 4,8)")

    # reducao relativa e folga SEM as execucoes da uniforme (gap medio de HG e S65, bracos independentes de U e de RP)
    dif = t["RP"] - t["U"]
    rel = dif / t["U"]
    x_ind = (t["HG"] + t["S65"]) / 2
    linhas = []
    for nome, sub in (("todas", pd.Series(True, index=t.index)), ("X_FRESH", is_x), ("SYN_REP", ~is_x)):
        d, rl, xi, xu, nn = dif[sub], rel[sub], x_ind[sub], t["U"][sub], n_cl[sub]
        rho_i, p_i = stats.spearmanr(xi, d)
        rho_u, p_u = stats.spearmanr(xu, d)
        rho_n, p_n = stats.spearmanr(nn, d)
        rho_r, p_r = stats.spearmanr(xi, rl)
        linhas.append(dict(conjunto=nome, n_inst=int(sub.sum()), gap_U=float(xu.mean()), dif=float(d.mean()), rel_media=float(rl.mean()), rel_mediana=float(rl.median()),
                           rho_folga_indep=float(rho_i), p_folga_indep=float(p_i), rho_folga_mesmas=float(rho_u), p_folga_mesmas=float(p_u), rho_tamanho=float(rho_n), p_tamanho=float(p_n),
                           rho_rel_folga=float(rho_r), p_rel_folga=float(p_r)))
    folga = pd.DataFrame(linhas)
    print("\n=== exploratorio: reducao relativa (RP - U)/gap U e correlacoes (folga independente = gap medio de HG e S65) ===")
    print(folga.round(4).to_string(index=False))

    # moderadores: K, n, clientes por rota e a razao de iteracoes RP/U (mediador), por conjunto
    it_ratio = np.log(dados[dados.braco == "RP"].groupby("instancia").iteracoes.mean() / dados[dados.braco == "U"].groupby("instancia").iteracoes.mean()).reindex(t.index)
    mods = []
    for nome, sub in (("todas", pd.Series(True, index=t.index)), ("X_FRESH", is_x), ("SYN_REP", ~is_x)):
        d_, kk, nn = dif[sub], K[sub], n_cl[sub].astype(float)
        for col, v in (("K", kk), ("n", nn), ("clientes por rota", nn / kk), ("log razao de iteracoes RP/U", it_ratio[sub])):
            rho, p = stats.spearmanr(v, d_)
            mods.append(dict(conjunto=nome, n_inst=int(sub.sum()), moderador=col, rho=float(rho), p=float(p)))
        for a, b, rot in ((kk, nn, "K | n"), (nn, kk, "n | K")):
            rp, pp = parcial(a.values, d_.values, b.values)
            mods.append(dict(conjunto=nome, n_inst=int(sub.sum()), moderador=rot, rho=rp, p=pp))
        rho, p = stats.spearmanr(kk, nn)
        mods.append(dict(conjunto=nome, n_inst=int(sub.sum()), moderador="corr(K, n)", rho=float(rho), p=float(p)))
    mods = pd.DataFrame(mods)
    print("\n=== exploratorio: correlacoes de RP - U com moderadores (K, n, clientes por rota) e com a razao de iteracoes ===")
    print(mods.round(4).to_string(index=False))

    # metanalise de efeito fixo de RP - U: Estudo 21 (51 instancias de avaliacao) e as 98 novas
    e21 = pivo(pd.concat([a21.referencias(), a21.ler_novos("P21")], ignore_index=True)) if not ensaio else None
    meta = []
    if e21 is not None:
        d21 = (e21["RP@P21"] - e21["U@P21"]).values
        for nome, difs in (("RP-U: Estudo 21 + 98 novas", [d21, dif.values]), ("RP-U: Estudo 21 + X_FRESH + SYN_REP", [d21, dif[is_x].values, dif[~is_x].values])):
            est, lo, hi, q, pq, i2 = meta_efeito_fixo(difs)
            meta.append(dict(analise=nome, estimativa=est, ic_lo=lo, ic_hi=hi, Q=q, p_Q=pq, I2=i2))
        d21b = (e21["RP@P21"] - e21["S65@P21"]).values
        est, lo, hi, q, pq, i2 = meta_efeito_fixo([d21b, (t["RP"] - t["S65"]).values])
        meta.append(dict(analise="RP-S65: Estudo 21 + 98 novas", estimativa=est, ic_lo=lo, ic_hi=hi, Q=q, p_Q=pq, I2=i2))
        # POS HOC (acrescentado depois de ver os resultados do Estudo 26; exploratorio, rotulado no diario): HG - U e S65 - U com o Estudo 21, e Uq - U com o Estudo 23 (as mesmas 51 instancias)
        for nome, a, b in (("HG-U: Estudo 21 + 98 novas", "HG", "U"), ("S65-U: Estudo 21 + 98 novas", "S65", "U")):
            est, lo, hi, q, pq, i2 = meta_efeito_fixo([(e21[f"{a}@P21"] - e21[f"{b}@P21"]).values, (t[a] - t[b]).values])
            meta.append(dict(analise=nome, estimativa=est, ic_lo=lo, ic_hi=hi, Q=q, p_Q=pq, I2=i2))
        e23 = pd.read_csv(RES / "etapa23_eval_p21.csv")
        uq = e23[(e23.braco == "s23_p21_Uq") & e23.viavel].groupby("instancia").gap_pct.mean().reindex(e21.index)
        est, lo, hi, q, pq, i2 = meta_efeito_fixo([(uq - e21["U@P21"]).values, (t["Uq"] - t["U"]).values])
        meta.append(dict(analise="Uq-U: Estudo 23 + 98 novas", estimativa=est, ic_lo=lo, ic_hi=hi, Q=q, p_Q=pq, I2=i2))
    meta = pd.DataFrame(meta)
    if len(meta):
        print("\n=== exploratorio: metanalise de efeito fixo (as duas amostras de instancias sao independentes) ===")
        print(meta.round(4).to_string(index=False))

    rows = []
    print("\n=== exploratorio: iteracoes medias e uso medio (fracao) por braco e conjunto, destruicoes | reparos ===")
    for nome, sub in (("X_FRESH", dados.instancia.str.startswith("X-n")), ("SYN_REP", ~dados.instancia.str.startswith("X-n"))):
        for a in ARMS:
            h = dados[(dados.braco == a) & sub]
            m = {c: pd.DataFrame(h[c].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean() for c in ("uso_d", "uso_r")}
            ud, ur = (m["uso_d"] / m["uso_d"].sum()).round(3).tolist(), (m["uso_r"] / m["uso_r"].sum()).round(3).tolist()
            print(f"  {nome:8s} {a:3s} it {h.iteracoes.mean():8.0f}  d {ud}  r {ur}")
            rows.append(dict(conjunto=nome, braco=a, iteracoes=float(h.iteracoes.mean()), uso_d="|".join(map(str, ud)), uso_r="|".join(map(str, ur))))
    if not ensaio:
        r.to_csv(RES / "analise_etapa26.csv", index=False)
        folga.to_csv(RES / "analise_etapa26_folga.csv", index=False)
        meta.to_csv(RES / "analise_etapa26_meta.csv", index=False)
        mods.to_csv(RES / "analise_etapa26_moderadores.csv", index=False)
        pd.DataFrame(rows).to_csv(RES / "analise_etapa26_uso.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa26_hipoteses.csv", index=False)


if __name__ == "__main__":
    main(ensaio="--ensaio" in sys.argv)
