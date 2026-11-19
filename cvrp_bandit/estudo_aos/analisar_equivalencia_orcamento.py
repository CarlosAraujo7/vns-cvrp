"""Equivalencia em orcamento (EXPLORATORIO, pos hoc): quanto mais orcamento a selecao uniforme precisaria para igualar o gap de um braco adaptativo no orcamento de referencia.

Para cada grupo de instancias com a mesma selecao uniforme em dois orcamentos (referencia e alternativo), a curva do gap da uniforme contra o logaritmo do orcamento e
aproximada pela reta que passa pelos dois pontos (gap medio das instancias); o fator de equivalencia de um braco A e o orcamento relativo B* em que a reta da uniforme
cai ao gap de A no orcamento de referencia: ln B* = (gap_A - gap_U) * ln(r) / (gap_U_alt - gap_U), com r = orcamento alternativo / orcamento de referencia.
IC95% bootstrap de 10.000 reamostragens das instancias (semente 0). Fator 1 = sem vantagem; 1,3 = a vantagem vale 30% mais orcamento para a uniforme.
Limitacao declarada: a reta de dois pontos supoe o gap linear em log do orcamento entre os dois orcamentos (a curva e convexa: a interpolacao subestima o fator se r > 1 e o
extrapola se A estiver fora do intervalo); a media por instancia usa o numero de sementes de cada CSV (10 e 3 nas X; 5 e 3 nas outras).

Uso: python estudo_aos/analisar_equivalencia_orcamento.py   (le os CSV de results/aos_study; escreve analise_equivalencia_orcamento.csv e imprime a tabela).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results" / "aos_study"
RAIZ = RES
NBOOT = 10_000


def medias(arq, conjuntos=None):
    """gap medio por instancia e braco (sufixo do nome do braco apos o ultimo '_')."""
    p = RES / arq
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d.viavel == True]  # noqa: E712
    if conjuntos is not None:
        d = d[d.conjunto.isin(conjuntos)]
    d = d.assign(arm=d.braco.str.split("_").str[-1])
    return d.groupby(["arm", "instancia"]).gap_pct.mean().unstack(0)


def fator(gu, ga, gu_alt, r):
    """ln B* = (ga - gu) ln r / (gu_alt - gu); gu, ga, gu_alt: medias das instancias."""
    den = gu_alt - gu
    if abs(den) < 1e-12:
        return float("nan")
    return math.exp((ga - gu) * math.log(r) / den)


def linha(grupo, ref, alt, r, bracos=("RP", "S65")):
    ids = ref.index.intersection(alt.index)
    ref, alt = ref.loc[ids], alt.loc[ids]
    rng = np.random.default_rng(0)
    n = len(ids)
    out = []
    for a in bracos:
        if a not in ref.columns or "U" not in alt.columns:
            continue
        gu, ga, gua = ref["U"].mean(), ref[a].mean(), alt["U"].mean()
        f = fator(gu, ga, gua, r)
        S = rng.integers(0, n, (NBOOT, n))
        bu, ba, bua = ref["U"].to_numpy()[S].mean(axis=1), ref[a].to_numpy()[S].mean(axis=1), alt["U"].to_numpy()[S].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            bs = np.exp((ba - bu) * math.log(r) / (bua - bu))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        out.append(dict(grupo=grupo, braco=a, razao_orcamento=r, n_inst=n, gap_U_ref=gu, gap_A_ref=ga, gap_U_alt=gua,
                        dif_ref=ga - gu, inclinacao_por_quadruplo=(gua - gu) / math.log(r) * math.log(4), fator=f, ic_lo=lo, ic_hi=hi))
    return out


def main():
    linhas = []
    ref_x = medias("etapa21_eval_P21.csv", ["X"])
    alt_x = medias("etapa22_eval_longx.csv")
    if ref_x is not None and alt_x is not None:
        linhas += linha("X (21 instancias): 1x contra 4x do orcamento", ref_x, alt_x, 4.0)
    ref_abc = medias("etapa22_eval_cls.csv")
    for nome, arq, r in [("A, B e CMT (53): 1x contra 0,25x", "etapa22_eval_short.csv", 0.25), ("A, B e CMT (53): 1x contra 20x", "etapa22_eval_longabc.csv", 20.0)]:
        alt = medias(arq)
        if ref_abc is not None and alt is not None:
            linhas += linha(nome, ref_abc, alt, r)
    ref_51 = medias("etapa21_eval_P21.csv", ["X", "SYN_VAL"])
    alt_51 = medias("etapa25_eval_short.csv")
    if ref_51 is not None and alt_51 is not None:
        linhas += linha("51 instancias de avaliacao: 1x contra 0,25x", ref_51, alt_51, 0.25)
    if not linhas:
        print("nenhum grupo com dados")
        return
    t = pd.DataFrame(linhas)
    t.to_csv(RES / "analise_equivalencia_orcamento.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
