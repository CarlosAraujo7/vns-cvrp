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
  U7  no teste prospectivo (Estudo 15; oito pools aninhados do TSP e do CVRPTW) nenhuma politica e significativamente melhor que a
      descida de ordem fixa do MESMO pool (familia exploratoria de 6 contrastes por pool);
  U8  no ALNS sob janelas de tempo (Estudo 16) nenhuma politica adaptativa e significativamente melhor que a mistura estatica
      ajustada (0 de 7); em relacao ao reparo unico fixo a afirmacao e FALSA para exatamente uma politica, o HG-UCB (H83 refutada,
      familia pre-registrada de 14 contrastes politica - {static, fixrep}, Holm); o script lista a excecao e falha se o conjunto de
      excecoes mudar;
  U6  no CVRPTW (Estudo 14) politicas superam a uniforme (U1w lista), mas NENHUMA politica e significativamente melhor que a melhor
      alternativa nao adaptativa (a ordem fixa de custo crescente, a de menor gap medio em E1 e E2): nenhum contraste
      politica - FOC_custo com dif < 0 e p ajustado < 0,05 (familia exploratoria de 18 testes);
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
    d14 = ler("analise_etapa14_v.csv")
    if d14 is not None:
        for _, r in d14[d14.grupo == "familiaA"].iterrows():
            linhas.append(("S14", f"{r.a} {r.cenario}", r.a, r.b, r.dif_pp, r.p_holm))
    d13 = ler("analise_etapa13_v.csv")
    if d13 is not None:
        # Estudo 13 (TSP, VND): 7 politicas x 3 cenarios contra a uniforme (familia A pre-registrada, Holm m=21)
        for _, r in d13[d13.grupo == "familiaA"].iterrows():
            linhas.append(("S13", f"{r.a} {r.cenario}", r.a, r.b, r.dif_pp, r.p_holm))
    return pd.DataFrame(linhas, columns=["estudo", "contraste", "a", "b", "dif_pp", "p_holm"])


def so_cvrp(todos):
    """os contrastes do CVRP (S1-S4, S6, S7, S10, S11): as afirmacoes U1/U2 e as macros u_* do texto valem SO para eles"""
    return todos[~todos.estudo.isin(["S13", "S14"])]


def main():
    todos = aos_vs_baseline()
    t = so_cvrp(todos)
    tsp = todos[todos.estudo == "S13"]
    tw = todos[todos.estudo == "S14"]
    print(f"{len(t)} contrastes 'politica adaptativa (ou mistura estatica) - baseline nao adaptativa' no CVRP (S1-S4, S6, S7, S10, S11)")
    melhores = t[(t.dif_pp < 0) & (t.p_holm < 0.05)]
    print("U1: contrastes do CVRP com a politica significativamente MELHOR:", len(melhores))
    if len(melhores):
        falhas.append("U1")
        print(melhores.to_string(index=False))
    piores = t[(t.dif_pp > 0) & (t.p_holm < 0.05)]
    print(f"    contrastes do CVRP com a politica significativamente PIOR: {len(piores)}")
    print(piores.to_string(index=False))
    # as macros u_* do texto (numeros.json, geradas por gerar_tabelas.py) devem ser EXATAMENTE estas contagens: o texto afirma "nenhum de u_n contrastes"
    # para o CVRP, e uma versao antiga as calculava tambem com o TSP e o CVRPTW (129 contrastes, 13 deles melhores)
    import json
    nj = Path(__file__).resolve().parent / "numeros.json"
    if nj.exists():
        num = json.loads(nj.read_text(encoding="utf-8"))
        esperado = {"u_n": len(t), "u_better": len(melhores), "u_worse": len(piores)}
        for k, v in esperado.items():
            if str(num.get(k)) != str(v):
                print(f"    MACRO {k} = {num.get(k)} no texto, mas o checador conta {v}")
                falhas.append(f"macro {k}")
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
    if len(tw):
        mw = tw[(tw.dif_pp < 0) & (tw.p_holm < 0.05)]
        pw = tw[(tw.dif_pp > 0) & (tw.p_holm < 0.05)]
        print(f"U1w (CVRPTW, Estudo 14, {len(tw)} contrastes): politica significativamente melhor que a uniforme em {len(mw)}, pior em {len(pw)}")
        print(mw.to_string(index=False))
        d14 = ler("analise_etapa14_v.csv")
        foc = d14[(d14.grupo == "explor_X") & (d14.b == "FOC_custo")]
        n6 = int(((foc.dif_pp < 0) & (foc.p_holm < 0.05)).sum())
        print(f"U6: politicas significativamente melhores que a ordem fixa de custo crescente: {n6} de {len(foc)} contrastes")
        if n6 > 0 or len(foc) == 0:
            falhas.append("U6")
    d15 = ler("analise_etapa15_foc.csv")
    if d15 is not None:
        n7 = int(((d15.dif_pp < 0) & (d15.p_holm < 0.05)).sum())
        print(f"U7 (Estudo 15, 8 pools): politicas significativamente melhores que o FOC do mesmo pool: {n7} de {len(d15)} contrastes")
        if n7 > 0:
            falhas.append("U7")
    d16 = ler("analise_etapa16_a.csv")
    if d16 is not None:
        m16 = d16[d16.grupo == "melhor_nao_adaptativo"]
        m8 = m16[(m16.dif_pp < 0) & (m16.p_holm < 0.05)]
        print(f"U8 (Estudo 16, ALNS no CVRPTW): politicas significativamente melhores que a mistura estatica ou o reparo fixo: {len(m8)} de {len(m16)} contrastes")
        for _, r in m8.iterrows():
            print(f"    excecao: {r.a} - {r.b} = {r.dif_pp:.2f} pp (p Holm {r.p_holm:.4f})")
        if len(m16) != 14 or sorted(zip(m8.a, m8.b)) != [("hgucb", "fixrep")]:
            falhas.append("U8")
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
