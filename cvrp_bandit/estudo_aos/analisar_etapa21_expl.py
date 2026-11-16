#!/usr/bin/env python3
"""
analisar_etapa21_expl.py
========================
Analises EXPLORATORIAS do Estudo 21 (rotular como tal; nenhuma hipotese confirmatoria): o perfil de custo e de rendimento de cada operador de P21 sob a selecao uniforme e o que o RP
faz com ele. Para cada operador (destruicao e reparo): avaliacoes de distancia por aplicacao (calls/uso), escore de sucesso por aplicacao (score/uso, sigma1 33, sigma2 9, sigma3 13), escore por
1.000 avaliacoes; participacao de uso sob U e sob RP (em %); correlacoes de Spearman entre a participacao do RP e o custo, o escore por aplicacao e o escore por avaliacao; uso do RP por terco do
orcamento; correlacao da diferenca RP - U com o tamanho. Fonte: etapa21_eval_P21.csv (colunas uso_d, uso_r, score_d, score_r, calls_d, calls_r, uso_d_fase, uso_r_fase).
Uso: python estudo_aos/analisar_etapa21_expl.py            (grava analise_etapa21_expl_operadores.csv e analise_etapa21_expl_fases.csv)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import alns_ext as ax  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results" / "aos_study"
ND = [n for n, _ in ax.DESTRUIR_P21]
NR = [n for n, _ in ax.REPARAR_P21]


def soma(h, col):
    return pd.DataFrame(h[col].str.split("|").apply(lambda v: [float(x) for x in v]).tolist()).sum().values


def main():
    e = pd.read_csv(RES / "etapa21_eval_P21.csv")
    e = e[e.viavel]
    linhas = []
    for comp, nomes, ku, ks, kc in (("destruir", ND, "uso_d", "score_d", "calls_d"), ("reparar", NR, "uso_r", "score_r", "calls_r")):
        u = e[e.braco == "s21_P21_U"]
        uso_u, sc_u, ca_u = soma(u, ku), soma(u, ks), soma(u, kc)
        rp = e[e.braco == "s21_P21_RP"]
        uso_rp = soma(rp, ku)
        custo = ca_u / np.maximum(uso_u, 1)                      # avaliacoes de distancia por aplicacao
        esc_ap = sc_u / np.maximum(uso_u, 1)                     # escore de sucesso por aplicacao
        esc_ev = 1000 * sc_u / np.maximum(ca_u, 1)               # escore por 1.000 avaliacoes de distancia
        part_u, part_rp = 100 * uso_u / uso_u.sum(), 100 * uso_rp / uso_rp.sum()
        for j, n in enumerate(nomes):
            linhas.append(dict(componente=comp, operador=n, custo_por_aplicacao=custo[j], escore_por_aplicacao=esc_ap[j], escore_por_mil_avaliacoes=esc_ev[j],
                               uso_U_pct=part_u[j], uso_RP_pct=part_rp[j]))
        for nome_c, v in (("custo", custo), ("escore por aplicacao", esc_ap), ("escore por 1.000 avaliacoes", esc_ev)):
            rho, p = stats.spearmanr(part_rp, v)
            print(f"[{comp}] Spearman(participacao do RP, {nome_c}) = {rho:+.2f} (p {p:.3f}, n {len(v)})")
    ops = pd.DataFrame(linhas)
    print(ops.round(2).to_string(index=False))
    # uso do RP por terco do orcamento (so os operadores com participacao media > 5%)
    rp = e[e.braco == "s21_P21_RP"]
    fases = []
    for comp, nomes, kf in (("destruir", ND, "uso_d_fase"), ("reparar", NR, "uso_r_fase")):
        m = np.zeros((3, len(nomes)))
        for s in rp[kf]:
            m += np.array([[float(x) for x in t.split("|")] for t in s.split(";")])
        m = 100 * m / m.sum(axis=1, keepdims=True)
        for j, n in enumerate(nomes):
            fases.append(dict(componente=comp, operador=n, terco1=m[0, j], terco2=m[1, j], terco3=m[2, j]))
    fases = pd.DataFrame(fases)
    print("\nuso do RP por terco do orcamento (%):")
    print(fases.round(1).to_string(index=False))
    # iteracoes e tempo
    for a in ("U", "RP", "S33", "S65", "HG"):
        h = e[e.braco == f"s21_P21_{a}"]
        print(f"{a:3s}: iteracoes medias {h.iteracoes.mean():8.0f}; avaliacoes por iteracao {h.calls_usadas.mean() / h.iteracoes.mean():8.0f}; tempo medio {h.tempo_s.mean():5.1f} s")
    # correlacao de RP - U com o tamanho
    t = e.groupby(["braco", "instancia"]).agg(gap=("gap_pct", "mean"), n=("n_customers", "first")).reset_index()
    p = t.pivot(index="instancia", columns="braco", values="gap")
    n = t.groupby("instancia").n.first()
    d = p["s21_P21_RP"] - p["s21_P21_U"]
    rho, pv = stats.spearmanr(n.reindex(d.index), d)
    print(f"\nSpearman(RP - U, tamanho) = {rho:+.2f} (p {pv:.3f}, n {len(d)}); X: {d[d.index.str.startswith('X-n')].mean():+.3f}  SYN_VAL: {d[d.index.str.startswith('X-synv')].mean():+.3f}")
    # folga da instancia: a vantagem do RP sobre a uniforme cresce com o gap da uniforme? (instancias de avaliacao X e SYN_VAL; P7 do Estudo 9 e P21)
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analisar_etapa21 as a21
    from analisar_etapa16 import pivo
    dd = pd.concat([a21.referencias(), a21.ler_novos("P21")], ignore_index=True)
    tt = pivo(dd)
    folga = []
    for pool in ("P7", "P21"):
        dif, gu = tt[f"RP@{pool}"] - tt[f"U@{pool}"], tt[f"U@{pool}"]
        rho_, p_ = stats.spearmanr(gu, dif)
        q = pd.qcut(gu, 3, labels=["baixo", "medio", "alto"])
        linha = dict(pool=pool, rho_gap=float(rho_), p_gap=float(p_), media=float(dif.mean()))
        for lab in ("baixo", "medio", "alto"):
            linha[f"dif_{lab}"], linha[f"gapU_{lab}"], linha[f"n_{lab}"] = float(dif[q == lab].mean()), float(gu[q == lab].mean()), int((q == lab).sum())
        folga.append(linha)
    folga = pd.DataFrame(folga)
    print("\nfolga (RP - U por terco do gap da uniforme):\n" + folga.round(3).to_string(index=False))
    folga.to_csv(RES / "analise_etapa21_expl_folga.csv", index=False)
    ops.to_csv(RES / "analise_etapa21_expl_operadores.csv", index=False)
    fases.to_csv(RES / "analise_etapa21_expl_fases.csv", index=False)


if __name__ == "__main__":
    main()
