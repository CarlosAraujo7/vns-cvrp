#!/usr/bin/env python3
"""
analisar_folga_metades.py
=========================
Checagem EXPLORATORIA (declarada depois de ver a analise de folga de `analisar_etapa21_expl.py`) de um possivel artefato na correlacao entre a folga da instancia e a vantagem dos pesos adaptativos:
a folga (o gap medio da uniforme, x) e a vantagem (RP - U, y) usam as MESMAS execucoes da uniforme, e o ruido da uniforme entra com sinais opostos em x e em y, o que produz uma correlacao negativa
ESPURIA mesmo se a vantagem verdadeira fosse constante (regressao a media). Aqui as seeds de cada instancia sao divididas ao acaso em duas metades: x vem de uma metade (gap medio da uniforme nas
seeds dessa metade) e y da outra (RP - U nas seeds da outra), e a correlacao de Spearman e a media de DUAS direcoes sobre 500 divisoes aleatorias (semente 0). Tambem se calcula a correlacao
de y com o gap de uma terceira medida independente do ruido da uniforme (o gap medio de TODOS os bracos exceto U e RP, nas mesmas instancias; usa execucoes distintas) e com a razao relativa
(RP - U) / gap da uniforme. Usos: pools P7 (Estudos 9 e 12) e P21 (Estudo 21), as 51 instancias de avaliacao. Nada e gravado alem de `analise_folga_metades.csv`.
Uso: python estudo_aos/analisar_folga_metades.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
sys.path.insert(0, str(AQUI))
import analisar_etapa21 as a21  # noqa: E402


def por_instancia_seed(d, braco):
    h = d[(d.braco == braco) & d.viavel]
    return h.pivot_table(index="instancia", columns="seed", values="gap_pct", aggfunc="mean")


def metades(m_u, m_rp, rng, reps=500):
    """m_u, m_rp: instancias x seeds (mesmas seeds). Devolve a media das duas direcoes de Spearman(x, y) sobre `reps` divisoes aleatorias das seeds de cada instancia."""
    inst = m_u.index.intersection(m_rp.index)
    rhos, ps = [], []
    for _ in range(reps):
        xa, ya, xb, yb = [], [], [], []
        for i in inst:
            s = [c for c in m_u.columns if pd.notna(m_u.at[i, c]) and c in m_rp.columns and pd.notna(m_rp.at[i, c])]
            s = list(rng.permutation(s))
            k = len(s) // 2
            a, b = s[:k], s[k:]
            if not a or not b:
                continue
            xa.append(m_u.loc[i, a].mean())
            ya.append(m_rp.loc[i, a].mean() - m_u.loc[i, a].mean())
            xb.append(m_u.loc[i, b].mean())
            yb.append(m_rp.loc[i, b].mean() - m_u.loc[i, b].mean())
        r1, p1 = stats.spearmanr(xa, yb)                                           # folga medida na metade A, vantagem medida na metade B
        r2, p2 = stats.spearmanr(xb, ya)
        rhos.append((r1 + r2) / 2)
        ps.append(max(p1, p2))
    return float(np.mean(rhos)), float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5)), float(np.median(ps)), len(xa)


def main():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 220)
    ref = a21.referencias()
    novo = a21.ler_novos("P21")
    saida = []
    for pool, d in (("P7", ref), ("P21", novo)):
        tag = f"@{pool}"
        dU, dRP = d[d.braco == f"U{tag}"], d[d.braco == f"RP{tag}"]
        outros = d[d.braco.isin([f"HG{tag}", f"S33{tag}", f"S65{tag}"])]                    # bracos que nao sao variantes do RP nem a uniforme
        mu = dU[dU.viavel].pivot_table(index="instancia", columns="seed", values="gap_pct", aggfunc="mean")
        mr = dRP[dRP.viavel].pivot_table(index="instancia", columns="seed", values="gap_pct", aggfunc="mean")
        x_all = mu.mean(axis=1)
        y_all = mr.mean(axis=1).reindex(mu.index) - x_all
        rho0, p0 = stats.spearmanr(x_all, y_all)
        rho_m, lo, hi, p_m, n = metades(mu, mr, rng)
        # medida independente da folga: gap medio dos OUTROS bracos (S33, S65, HG, ...) nas mesmas instancias
        x_out = outros[outros.viavel].groupby("instancia").gap_pct.mean().reindex(mu.index)
        rho_o, p_o = stats.spearmanr(x_out, y_all)
        rel = y_all / x_all
        rho_r, p_r = stats.spearmanr(x_all, rel)
        linha = dict(pool=pool, n_inst=len(mu), rho_completo=float(rho0), p_completo=float(p0), rho_metades=rho_m, rho_metades_lo=lo, rho_metades_hi=hi, p_metades_mediana=p_m,
                     rho_outros=float(rho_o), p_outros=float(p_o), rel_media=float(rel.mean()), rel_mediana=float(rel.median()), rho_rel=float(rho_r), p_rel=float(p_r))
        saida.append(linha)
        print(f"{pool}: n {len(mu)}; Spearman(gap U, RP - U) com as mesmas execucoes {rho0:+.3f} (p {p0:.4f}); em metades das seeds {rho_m:+.3f} [{lo:+.3f}; {hi:+.3f}] (p mediano {p_m:.4f}); "
              f"com o gap dos outros bracos {rho_o:+.3f} (p {p_o:.4f}); razao (RP - U)/gap U: media {rel.mean():+.3f}, mediana {rel.median():+.3f}, Spearman com o gap {rho_r:+.3f} (p {p_r:.3f})")
    pd.DataFrame(saida).to_csv(RES / "analise_folga_metades.csv", index=False)


if __name__ == "__main__":
    main()
