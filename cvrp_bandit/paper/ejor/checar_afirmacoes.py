#!/usr/bin/env python3
"""
checar_afirmacoes.py
====================
Confere, direto das tabelas de testes confirmatorios, as afirmacoes UNIVERSAIS do manuscrito (as que dizem "nenhuma",
"todas", "em nenhum estudo"), porque um unico contraste em contrario as derruba:

  U1  nos Estudos do CVRP (S1-S11) nenhuma politica adaptativa (nem ajustada, nem de outras familias) supera a selecao
      uniforme (ou a ordem aleatoria dentro do VND) com significancia apos Holm: nenhum contraste confirmatorio
      "politica - baseline" com dif < 0 e p ajustado < 0,05. No TSP (Estudo 13, VND) essa afirmacao e FALSA e o script lista
      as excecoes (U1t); a afirmacao que vale no TSP e U5;
  U2  o HG-UCB (literal ou sensivel a custo) nunca e significativamente melhor que a baseline nao adaptativa nos estudos do CVRP;
  U5  no TSP (Estudo 13), toda politica que supera a uniforme com significancia e, no mesmo cenario, significativamente PIOR que a
      ordem fixa de custo crescente sem aprendizado (analise pos-hoc, Holm sobre a familia pos-hoc);
  U3  a variante literal do HG-UCB e pior que a normalizada nos dois orcamentos (74/74);
  U4  os efeitos estruturais (termino, ordem aleatoria, conjunto de operadores) sao significativos e maiores que 0,2pp.
Uso: python checar_afirmacoes.py     (sai com codigo 1 se alguma afirmacao falhar)
"""
import sys
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent.parent / "results" / "aos_study"
falhas = []


def ler(nome):
    p = RES / nome
    return pd.read_csv(p) if p.exists() else None


def aos_vs_baseline():
    """(estudo, id/contraste, a, b, dif, p_holm) para os contrastes 'politica adaptativa - baseline nao adaptativa'."""
    linhas = []
    d1 = ler("analise_confirmatoria.csv")
    for _, r in d1.iterrows():
        if r.b == "uniforme" and r.a != "uniforme":
            linhas.append(("S1", f"B={r.B}", r.a, r.b, r.dif_media_pp, r.p_holm))
    d2 = ler("analise_etapa2_confirmatoria.csv")
    for _, r in d2[d2.id.isin(["D1", "D2", "D3", "D4"])].iterrows():
        linhas.append(("S2", f"{r.id} B={r.B}", r.a, r.b, r.dif_pp, r.p_holm))
    d3 = ler("analise_etapa3_confirmatoria.csv")
    for _, r in d3[d3.id.str.contains("vsU") & ~d3.id.str.contains("FOC")].iterrows():
        linhas.append(("S3", f"{r.id} E{int(r.E)}", r.a, r.b, r.dif_pp, r.p_holm))
    d4 = ler("analise_etapa4_confirmatoria.csv")
    for _, r in d4[d4.id.isin(["V1", "V2", "V3", "V4"])].iterrows():
        linhas.append(("S4", f"{r.id} E{int(r.nivel)}", r.a, r.b, r.dif_pp, r.p_holm))
    for _, r in d4[d4.id.isin(["W3", "W4", "W5", "W6"])].iterrows():
        linhas.append(("S4val", r.id, r.a, r.b, r.dif_pp, r.p_holm))
    d6 = ler("analise_etapa6_confirmatoria.csv")
    if d6 is not None:
        for _, r in d6[d6.grupo == "confirmatorio"].iterrows():
            linhas.append(("S6", f"{r.id} E{int(r.nivel)}", r.a, r.b, r.dif_pp, r.p_holm))
    d7 = ler("analise_etapa7_confirmatoria.csv")
    if d7 is not None:
        # Estudo 7 (HGS-CVRP): 4 politicas adaptativas x 2 pontos de controle x 2 populacoes contra a ordem aleatoria
        for _, r in d7[(d7.grupo == "confirmatorio") & (d7.a != "fixed")].iterrows():
            linhas.append(("S7", f"{r.a} {r['pop']} g{int(r.ponto)}", r.a, r.b, r.dif_pp, r.p_holm))
    # Estudo 10 (instancias grandes) e Estudo 11 (SWAP* eficiente): mesmos operadores deterministas, so mudam o tamanho e o custo do SWAP*
    d10 = ler("analise_etapa10_confirmatoria.csv")
    if d10 is not None:
        for _, r in d10[d10.grupo == "confirmatorio"].iterrows():
            linhas.append(("S10", r.id, r.a, r.b, r.dif_pp, r.p_holm))
    d11 = ler("analise_etapa11_confirmatoria.csv")
    if d11 is not None:
        for _, r in d11[d11.grupo == "familiaA"].iterrows():
            linhas.append(("S11", f"{r.a} {r.cenario}", r.a, r.b, r.dif_pp, r.p_holm))
    d13 = ler("analise_etapa13_v.csv")
    if d13 is not None:
        # Estudo 13 (TSP, VND): 7 politicas x 3 cenarios contra a uniforme (familia A pre-registrada, Holm m=21)
        for _, r in d13[d13.grupo == "familiaA"].iterrows():
            linhas.append(("S13", f"{r.a} {r.cenario}", r.a, r.b, r.dif_pp, r.p_holm))
    return pd.DataFrame(linhas, columns=["estudo", "contraste", "a", "b", "dif_pp", "p_holm"])


def main():
    todos = aos_vs_baseline()
    t = todos[todos.estudo != "S13"]
    tsp = todos[todos.estudo == "S13"]
    print(f"{len(t)} contrastes 'politica adaptativa (ou mistura estatica) - baseline nao adaptativa' no CVRP (S1-S4, S6, S7, S10, S11)")
    melhores = t[(t.dif_pp < 0) & (t.p_holm < 0.05)]
    print("U1: contrastes do CVRP com a politica significativamente MELHOR:", len(melhores))
    if len(melhores):
        falhas.append("U1")
        print(melhores.to_string(index=False))
    piores = t[(t.dif_pp > 0) & (t.p_holm < 0.05)]
    print(f"    contrastes do CVRP com a politica significativamente PIOR: {len(piores)}")
    print(piores.to_string(index=False))
    hg = t[t.a.str.contains("hgucb") | (t.a == "hg")]   # "hg" = HG-UCB-lite do Estudo 7
    mh = hg[(hg.dif_pp < 0) & (hg.p_holm < 0.05)]
    print(f"U2: HG-UCB significativamente melhor (CVRP): {len(mh)} de {len(hg)} contrastes; "
          f"significativamente pior: {int(((hg.dif_pp > 0) & (hg.p_holm < 0.05)).sum())}")
    if len(mh):
        falhas.append("U2")
    if len(tsp):
        mt = tsp[(tsp.dif_pp < 0) & (tsp.p_holm < 0.05)]
        pt = tsp[(tsp.dif_pp > 0) & (tsp.p_holm < 0.05)]
        print(f"U1t (TSP, Estudo 13, {len(tsp)} contrastes): politica significativamente melhor que a uniforme em {len(mt)}, pior em {len(pt)}")
        print(mt.to_string(index=False))
        d13o = ler("analise_etapa13_ordem.csv")
        if d13o is None:
            falhas.append("U5 (sem analise_etapa13_ordem.csv)")
        else:
            ok5, n5 = True, 0
            for _, r in mt.iterrows():
                pol, cen = r.a, r.contraste.split(" ", 1)[1]
                rr = d13o[(d13o.a == "FOC_asc") & (d13o.b == pol) & (d13o.cenario == cen)]
                if rr.empty:
                    print(f"    sem contraste de ordem fixa para {pol} em {cen}")
                    ok5 = False
                    continue
                rr = rr.iloc[0]
                dom = bool(rr.dif_pp < 0 and rr.p_holm < 0.05)
                print(f"    {pol} em {cen}: ordem fixa crescente - politica = {rr.dif_pp:+.2f} pp (p Holm {rr.p_holm:.4f}) -> {'domina' if dom else 'NAO domina'}")
                ok5 &= dom
                n5 += 1
            print(f"U5: toda politica que supera a uniforme no TSP e dominada pela ordem fixa de custo crescente: {ok5} ({n5} contrastes)")
            if not ok5:
                falhas.append("U5")
    d1 = ler("analise_confirmatoria.csv")
    lit = d1[d1.a == "hgucb_literal"]
    print(f"U3: literal - normalizado: {[round(x, 2) for x in lit.dif_media_pp]} pp, derrotas do literal {list(lit.der_a)} de 74")
    if not (len(lit) == 2 and (lit.dif_media_pp > 0.5).all() and (lit.der_a == 74).all()):
        falhas.append("U3")
    d2, d3 = ler("analise_etapa2_confirmatoria.csv"), ler("analise_etapa3_confirmatoria.csv")
    estr = pd.concat([d2[d2.id.isin(["C1", "C2"])].dif_pp.abs(), d3[d3.id == "T1"].dif_pp.abs()])
    ok4 = bool((estr > 0.2).all()) and bool(d2[d2.id.isin(["C1", "C2", "C5", "C6"])].signif_005.all()) and bool(d3[d3.id == "T1"].signif_005.all())
    print(f"U4: efeitos estruturais (termino, ordem, conjunto) todos significativos e > 0,2pp: {ok4} (min |dif| termino/conjunto = {estr.min():.2f})")
    if not ok4:
        falhas.append("U4")
    print("AFIRMACOES UNIVERSAIS:", "OK" if not falhas else f"FALHAS {falhas}")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
