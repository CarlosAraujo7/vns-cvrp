#!/usr/bin/env python3
"""
analisar_etapa21b.py
====================
Analise da EMENDA 1 do Estudo 21 (diario, Entrada 139): o braco S129, a mistura estatica de P21 com 129 avaliacoes de ajuste (quatro vezes o esforco de S33).
Mesma unidade, estatisticas e convencoes de analisar_etapa21.py (dif = a - b em pp; negativo favorece a; instancia = media das seeds viaveis; Wilcoxon bilateral
pareado; IC95% bootstrap 10.000, semente 0; empate |dif| < 1e-9; MDE = 2,8 DP / raiz(n)); 51 instancias; sem correcao (cada afirmacao fica presa a UM teste):
  H139  S129 - RP  em P21 (negativo: a mistura com 129 avaliacoes supera os pesos adaptativos sem ajuste)
  H140  S129 - S65 em P21 (negativo: dobrar de novo o ajuste ajuda)
Exploratorios (rotular): S129 - U, S129 - S33, S129 - RPt, S129 - HG em P21; equivalencia (IC90% em +-0,10 pp); a curva de aprendizado estendida (validacao, seeds 3003-3005).
Se a continuacao do ES nao achou melhora, S129 = S65 (coluna copiada).
Uso: python estudo_aos/analisar_etapa21b.py [--ensaio]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisar_etapa16 import contraste, pivo  # noqa: E402
import analisar_etapa21 as a21  # noqa: E402

RES = a21.RES
COLS = a21.COLS


def main(ensaio=False):
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    ref = a21.referencias()
    if ensaio:
        novos = a21.fabricar(ref, np.random.default_rng(7))
        s129 = novos[novos.braco == "S33@P21"].assign(braco="S129@P21", gap_pct=lambda d: d.gap_pct - 0.12)
    else:
        novos = pd.concat([a21.ler_novos(p) for p in a21.POOLS], ignore_index=True)
        f = RES / "etapa21b_eval_P21.csv"
        if f.exists():
            s129 = pd.read_csv(f)
            s129["braco"] = s129.braco.str.replace("s21_P21_", "", regex=False) + "@P21"
            s129 = s129[COLS]
        else:
            s129 = None
    dados = a21.completar(pd.concat([ref, novos], ignore_index=True))
    if s129 is None:
        print("S129 ausente: a continuacao do ES nao achou melhora (S129 = S65); coluna copiada")
        s129 = dados[dados.braco == "S65@P21"].assign(braco="S129@P21")
    dados = pd.concat([dados, s129], ignore_index=True)
    t = pivo(dados)
    print(f"=== {len(t)} instancias; gap medio (%) em P21 ===")
    print(t[[c for c in t.columns if c.endswith("@P21")]].mean().round(3).sort_values().to_string())
    C = lambda cid, a, b, g: contraste(t, cid, a, b, rng, g)  # noqa: E731
    prim = [C("H139 S129-RP@P21", "S129@P21", "RP@P21", "emenda1"), C("H140 S129-S65@P21", "S129@P21", "S65@P21", "emenda1")]
    expl = [C(f"S129-{b}@P21", "S129@P21", f"{b}@P21", "explor") for b in ("U", "S33", "RPt", "HG")]
    r = pd.DataFrame(prim + expl)
    r["p_holm"], r["signif_005"] = r.p_bruto, r.p_bruto < 0.05
    print(r.drop(columns=["grupo", "a", "b"]).round(4).to_string(index=False))
    H = {"H139 S129 - RP em P21 significativamente negativo (emenda 1, sem correcao)": bool(r.iloc[0].p_bruto < 0.05 and r.iloc[0].dif_pp < 0),
         "H140 S129 - S65 em P21 significativamente negativo (emenda 1, sem correcao)": bool(r.iloc[1].p_bruto < 0.05 and r.iloc[1].dif_pp < 0)}
    print("\n=== hipoteses da emenda 1 (Entrada 139) ===")
    for k, ok in H.items():
        print(f"  {'CONFIRMADA' if ok else 'REFUTADA  '}  {k}")
    print(f"  confirmadas: {sum(H.values())} de 2 (esperado: 0,55)")
    if not ensaio:
        v = pd.read_csv(RES / "etapa21b_validacao_P21.csv")
        v = v[v.viavel]
        v0 = pd.read_csv(RES / "etapa21_validacao_P21.csv")
        v0 = v0[v0.viavel]
        esc = json.loads((RES / "etapa21b_ajuste_escolhidas_P21.json").read_text())
        curva = []
        for tr in esc["trajetoria"]:
            g = tr["geracao"]
            nome = f"val21_P21_gen{g:02d}"
            src = v if (v.braco == nome).any() else v0
            if (src.braco == nome).any():
                curva.append(dict(pool="P21", geracao=g, n_avaliacoes=tr["n_avaliacoes"], gap_ajuste=tr["gap_ajuste"], gap_validacao=float(src[src.braco == nome].gap_pct.mean())))
        curva = pd.DataFrame(curva)
        print("\n=== exploratorio: curva de aprendizado do ajuste de P21 (validacao: seeds 3003-3005) ===")
        print(curva.round(4).to_string(index=False))
        print("validacao de S129 e de RP (padrao):", {k: round(float(x), 4) for k, x in {"S129": v[v.braco == "s21_P21_S129"].gap_pct.mean(),
                                                                                         "RP": v0[v0.braco == "s21_P21_RP"].gap_pct.mean()}.items()})
        r.to_csv(RES / "analise_etapa21b.csv", index=False)
        curva.to_csv(RES / "analise_etapa21b_curva.csv", index=False)
        pd.DataFrame([dict(hipotese=k, confirmada=bool(ok)) for k, ok in H.items()]).to_csv(RES / "analise_etapa21b_hipoteses.csv", index=False)


if __name__ == "__main__":
    main(ensaio="--ensaio" in sys.argv)
