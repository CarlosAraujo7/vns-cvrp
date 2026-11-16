#!/usr/bin/env python3
"""
analisar_moderadores.py
=======================
Analise EXPLORATORIA (declarada depois de ver que a correlacao de folga estava inflada; diario, Entrada 145): onde a vantagem dos pesos adaptativos sobre a uniforme (RP - U, pp de gap, P21) aparece?
Moderadores: numero de rotas K (rotas da solucao de referencia: nao e medido com as execucoes comparadas), tamanho n, clientes por rota, e a razao de iteracoes RP/U (um MEDIADOR: depende do
comportamento do RP, nao da instancia). Conjuntos: as 51 instancias de avaliacao (X e SYN_VAL, Estudo 21), A, B e CMT (Estudo 22, regime cls) e as 104. Correlacoes de Spearman e, para K e n, as
correlacoes parciais (ranks; a outra variavel controlada). Tambem o uso medio do RP em rota_pequena e a taxa de candidatos inviaveis por conjunto. Grava `analise_moderadores.csv`.
Uso: python estudo_aos/analisar_moderadores.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
RES = ROOT / "results" / "aos_study"
sys.path.insert(0, str(ROOT / "src"))
import alns_ext as ax  # noqa: E402

ND = [n for n, _ in ax.POOLS["P21"][0]]


def k_referencia(inst, conj):
    """Numero de rotas da solucao de referencia (arquivo .sol da instancia)."""
    p = ROOT / "instances" / conj / f"{inst}.sol"
    return sum(1 for linha in p.read_text().splitlines() if linha.startswith("Route"))


def parcial(x, y, z):
    """Correlacao parcial de postos de x e y dado z, e o p (t com n - 3 graus de liberdade)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    r = lambda a, b: np.corrcoef(a, b)[0, 1]  # noqa: E731
    rxy, rxz, ryz = r(rx, ry), r(rx, rz), r(ry, rz)
    rp = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    n = len(x)
    t = rp * np.sqrt((n - 3) / (1 - rp ** 2))
    return float(rp), float(2 * stats.t.sf(abs(t), n - 3))


def main():
    e21 = pd.read_csv(RES / "etapa21_eval_P21.csv")
    e22 = pd.read_csv(RES / "etapa22_eval_cls.csv")
    e22["braco"] = e22.braco.str.replace("s22_cls_", "s21_P21_", regex=False)
    d = pd.concat([e21[e21.braco.isin(["s21_P21_U", "s21_P21_RP"])], e22[e22.braco.isin(["s21_P21_U", "s21_P21_RP"])]], ignore_index=True)
    d = d[d.viavel].copy()
    d["frac_inv"] = 100 * d.inviaveis / d.iteracoes
    g = d.groupby(["instancia", "braco"]).agg(gap=("gap_pct", "mean"), it=("iteracoes", "mean"), n=("n_customers", "first"), conj=("conjunto", "first"), inv=("frac_inv", "mean")).reset_index()
    u, r = g[g.braco == "s21_P21_U"].set_index("instancia"), g[g.braco == "s21_P21_RP"].set_index("instancia")
    t = pd.DataFrame({"conj": u.conj, "n": u.n, "gap_u": u.gap, "dif": r.gap - u.gap, "lr": np.log(r.it / u.it), "inv_u": u.inv, "inv_rp": r.inv})
    t["K"] = [k_referencia(i, c) for i, c in zip(t.index, t.conj)]
    t["cpr"] = t.n / t.K
    t["rel"] = t.dif / t.gap_u
    linhas = []
    for nome, sub in (("51 de avaliacao (X e SYN_VAL)", t[t.conj.isin(["X", "SYN_VAL"])]), ("A, B e CMT (cls)", t[t.conj.isin(["A", "B", "CMT"])]), ("todas as 104", t)):
        for col in ("K", "n", "cpr", "lr"):
            rho, p = stats.spearmanr(sub[col], sub.dif)
            linhas.append(dict(conjunto=nome, n_inst=len(sub), moderador=col, rho=float(rho), p=float(p), media_moderador=float(sub[col].mean())))
        for a, b, rot in (("K", "n", "K | n"), ("n", "K", "n | K")):
            rp, pp = parcial(sub[a], sub.dif, sub[b])
            linhas.append(dict(conjunto=nome, n_inst=len(sub), moderador=rot, rho=rp, p=pp, media_moderador=float("nan")))
        rho, p = stats.spearmanr(sub.K, sub.n)
        linhas.append(dict(conjunto=nome, n_inst=len(sub), moderador="corr(K, n)", rho=float(rho), p=float(p), media_moderador=float("nan")))
    out = pd.DataFrame(linhas)
    print(out.round(4).to_string(index=False))
    por_conj = t.groupby("conj").agg(n_inst=("n", "size"), K=("K", "mean"), n=("n", "mean"), gap_u=("gap_u", "mean"), dif=("dif", "mean"), razao_it=("lr", lambda v: float(np.exp(v).mean())),
                                    inv_u=("inv_u", "mean"), inv_rp=("inv_rp", "mean")).round(3)
    print("\n" + por_conj.to_string())
    # uso do RP em rota_pequena por conjunto
    uso = []
    for c in ("A", "B", "CMT", "X", "SYN_VAL"):
        h = d[(d.conjunto == c) & (d.braco == "s21_P21_RP")]
        m = pd.DataFrame(h.uso_d.str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).mean()
        m = 100 * m / m.sum()
        uso.append(dict(conjunto=c, rota_pequena=float(m[ND.index("rota_pequena")]), rota=float(m[ND.index("rota")])))
    print("\nuso medio (%) do RP em rota_pequena e rota:\n" + pd.DataFrame(uso).round(1).to_string(index=False))
    out.to_csv(RES / "analise_moderadores.csv", index=False)
    por_conj.reset_index().to_csv(RES / "analise_moderadores_conjuntos.csv", index=False)
    pd.DataFrame(uso).to_csv(RES / "analise_moderadores_uso.csv", index=False)


if __name__ == "__main__":
    main()
