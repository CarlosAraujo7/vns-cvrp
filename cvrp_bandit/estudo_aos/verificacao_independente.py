#!/usr/bin/env python3
"""
verificacao_independente.py  (v3)
=================================
Controle de verificacao independente (GUIA_PESQUISA_GERAL.md, Secao 3.6).
Recalcula, DIRETO dos custos brutos e dos valores de referencia e usando apenas
a biblioteca padrao (csv, json, statistics) -- sem pandas/scipy, sem codigo
compartilhado com os scripts de analise --, os numeros que o manuscrito usa:

  (1) as medias de gap de todos os bracos (macros `s1_gap_*`, `s2a_gap_*`,
      `s3_gap_*`, `s4g_*`, `s4v_gap_*`, `s5_gap_*` em paper/ejor/numeros.json,
      geradas por gerar_tabelas.py), as medias da tabela de reproducao
      (`rep_*`), os tetos externos (`hgs_*`, `kool_*`) e as medias do ajuste
      (`s4_tune_*`);
  (2) para TODOS os contrastes confirmatorios (analise_*_confirmatoria.csv,
      Estudos 1-5): a diferenca media pareada, o numero de vitorias e de
      derrotas (com tolerancia de empate de 1e-9);
  (3) que a coluna gap gravada em cada CSV bruto e igual a
      100*(custo-ref)/ref, que nao ha execucao inviavel e que nao ha chave
      (braco, nivel, instancia, seed) duplicada.
(Os valores-p vem do scipy e nao sao reimplementados aqui; ver ressalva no texto.)

Sai com codigo 1 se algo divergir. Uso: python verificacao_independente.py
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RES = AQUI.parent / "results" / "aos_study"
NUMJSON = AQUI.parent / "paper" / "ejor" / "numeros.json"

# (arquivo, coluna do braco, coluna do nivel, prefixo de grupo). O conjunto de validacao sintetico fica em grupo
# proprio ("VAL:") para nao misturar as 30 instancias sinteticas com as 74 da CVRPLIB.
FONTES = [("full.csv", "politica", "B", ""), ("etapa2_full.csv", "braco", "B", ""),
          ("etapa3_full.csv", "braco", "nivel", ""), ("etapa4_eval74.csv", "braco", "nivel", ""),
          ("etapa5_full.csv", "braco", "nivel", ""), ("etapa6_full.csv", "braco", "nivel", ""),
          ("etapa4_val.csv", "braco", "nivel", "VAL:"), ("etapa6_val.csv", "braco", "nivel", "VAL:")]
problemas = []
# macros s4_gap_<familia>_<nivel> -> braco bruto (t: ajustada no Estudo 4; d: configuracao padrao do Estudo 3)
ALIAS_S4 = {"ucb1_t": "s4_R5_ucb1t", "ucb1_d": "s3_R5_ucb1c", "pm_t": "s4_R5_pmt", "pm_d": "s3_R5_pmc",
            "hgucb_t": "s4_R5_hgucbt", "hgucb_d": "s3_R5_hgucbc", "static": "s4_R5_static", "U": "s3_R5_U"}


def melhor_config(arq, braco=None):
    """menor media de gap entre as configuracoes de um CSV de ajuste (regra de selecao do pre-registro)."""
    por = defaultdict(list)
    with open(RES / arq, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if braco is None or r["braco"] == braco:
                por[r["config_id"]].append(float(r["gap_pct"]))
    return min(statistics.fmean(v) for v in por.values())


def lerr(arq, col_braco, col_nivel, grupo):
    """dict (grupo+braco, nivel) -> dict instancia -> lista de gaps, e contagens de integridade."""
    d = defaultdict(lambda: defaultdict(list))
    n = inv = dv = dup = 0
    chaves = set()
    p = RES / arq
    if not p.exists():
        return d, 0, 0, 0, 0
    with open(p, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            n += 1
            k = (r[col_braco], r[col_nivel], r["instancia"], r["seed"])
            dup += k in chaves
            chaves.add(k)
            if r["viavel"] != "True":
                inv += 1
                continue
            ref = float(r["otimo"] if "otimo" in r else r["ref"])
            gap = 100.0 * (float(r["custo"]) - ref) / ref
            if abs(gap - float(r["gap_pct"])) > 1e-6:
                dv += 1
            d[(grupo + r[col_braco], int(r[col_nivel]))][r["instancia"]].append(gap)
    return d, n, inv, dv, dup


def media_csv(arq, col, filtro=None):
    """media (sobre instancias) das medias por instancia de `col`, so com a biblioteca padrao."""
    por = defaultdict(list)
    with open(RES.parent / arq if not (RES / arq).exists() else RES / arq, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if filtro and not filtro(r):
                continue
            por[r["instancia"]].append(float(r[col]))
    return statistics.fmean(statistics.fmean(v) for v in por.values())


def confere(chave, valor, m, casas=2):
    if f"{m:.{casas}f}" != valor:
        problemas.append(f"macro {chave}: texto {valor} vs. recalculado {m:.5f}")
        return 0
    return 1


def main():
    dados = {}
    tot = 0
    for arq, cb, cn, grupo in FONTES:
        d, n, inv, dv, dup = lerr(arq, cb, cn, grupo)
        for k, v in d.items():
            dados.setdefault(k, {}).update({i: statistics.fmean(g) for i, g in v.items()})
        tot += n
        if inv or dv or dup:
            problemas.append(f"{arq}: {inv} inviaveis, {dv} gaps gravados divergentes, {dup} chaves duplicadas")
        print(f"  {arq}: {n} execucoes lidas")
    num = json.loads(NUMJSON.read_text(encoding="utf-8"))
    # --- (1a) medias dos macros de bracos
    conferidos = 0
    pref_cvrplib = ("s1_gap_", "s2a_gap_", "s3_gap_", "s4g_", "s5_gap_", "s6_gap_")
    for chave, valor in num.items():
        pref = next((p for p in pref_cvrplib if chave.startswith(p)), None)
        if pref:
            arm, nivel = chave[len(pref):].rsplit("_", 1)
            k = (arm, int(nivel))
        elif chave.startswith(("s4v_gap_", "s6v_gap_")):
            k = ("VAL:" + chave[len("s4v_gap_"):], 1)           # os dois prefixos tem o mesmo tamanho
        elif chave.startswith("s4_gap_"):
            fam, nv = chave[len("s4_gap_"):].rsplit("_", 1)
            k = (ALIAS_S4[fam], int(nv))
        else:
            continue
        if k not in dados:
            problemas.append(f"macro {chave}: braco/nivel {k} ausente nos dados brutos")
            continue
        conferidos += confere(chave, valor, statistics.fmean(dados[k].values()))
    print(f"(1a) medias de bracos: {conferidos} macros conferidos")
    # --- (1b) reproducao, tetos externos e ajuste
    outros = 0
    rep = {"gvns": "gvns_baseline_{s}.csv", "lit_fixo": "bandit_vns_validation_{s}.csv",
           "lit_prog": "bandit_vns_validation_progressivo_{s}.csv",
           "n5": "bandit_vns_validation_progressivo_normalizado_alpha5.0_{s}.csv",
           "n05": "bandit_vns_validation_progressivo_normalizado_alpha0.5_{s}.csv"}
    for ch, arq in rep.items():
        for s in "ABX":
            k = f"rep_{ch}_{s}"
            if k in num:
                outros += confere(k, num[k], media_csv(arq.format(s=s), "erro_pct"))
    for ch, arq in (("hgs_matched", "hgs_74_matched.csv"), ("hgs_n01", "hgs_74_n01.csv")):
        if ch in num:
            outros += confere(ch, num[ch], media_csv(arq, "gap_pct", lambda r: r["viavel"] == "True"), 3)
    if "kool_gap" in num:
        outros += confere("kool_gap", num["kool_gap"], media_csv("kool_zeroshot_74.csv", "gap_pct",
                                                                  lambda r: r["viavel_sem_restricao_k"] == "True"))
        outros += confere("kool_gap_strict", num["kool_gap_strict"],
                          media_csv("kool_zeroshot_74.csv", "gap_pct", lambda r: r["viavel_estrito"] == "True"))
    for ch, br in (("s4_tune_U", "tune_U"),):
        if ch in num:
            outros += confere(ch, num[ch], media_csv("aos_study/etapa4_tuning_grid.csv", "gap_pct", lambda r, b=br: r["braco"] == b))
    if "s4_tune_static0" in num:
        outros += confere("s4_tune_static0", num["s4_tune_static0"],
                          media_csv("aos_study/etapa4_tuning_static.csv", "gap_pct", lambda r: r["config_id"] == "gen00"))
    for fam, br in (("ucb1", "tune_ucb1t"), ("pm", "tune_pmt"), ("hgucb", "tune_hgucbt")):
        k = f"s4_tune_{fam}_t"
        if k in num:
            outros += confere(k, num[k], melhor_config("etapa4_tuning_grid.csv", br))
    if "s4_tune_static_best" in num:
        outros += confere("s4_tune_static_best", num["s4_tune_static_best"], melhor_config("etapa4_tuning_static.csv"))
    print(f"(1b) reproducao, tetos externos e ajuste: {outros} macros conferidos")
    # --- (2) contrastes
    cont = 0
    for arq, niv_col in (("analise_confirmatoria.csv", "B"), ("analise_etapa2_confirmatoria.csv", "B"),
                         ("analise_etapa3_confirmatoria.csv", "E"), ("analise_etapa4_confirmatoria.csv", "nivel"),
                         ("analise_etapa5_confirmatoria.csv", "nivel"), ("analise_etapa6_confirmatoria.csv", "nivel"),
                         ("analise_etapa6_val.csv", "nivel")):
        p = RES / arq
        if not p.exists():
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            val = r.get(niv_col, "") == ""                       # linha de validacao sintetica (sem nivel)
            nv = 1 if val else int(float(r[niv_col]))
            grupo, n_esp = ("VAL:", 30) if val else ("", 74)
            a, b = r["a"], r["b"]
            if (grupo + a, nv) not in dados or (grupo + b, nv) not in dados:
                problemas.append(f"{arq} {r.get('id', '')}: ({a}|{b}, {nv}) ausente nos dados brutos")
                continue
            ia, ib = dados[(grupo + a, nv)], dados[(grupo + b, nv)]
            inst = sorted(set(ia) & set(ib))
            dif = [ia[i] - ib[i] for i in inst]
            dif = [0.0 if abs(x) < 1e-9 else x for x in dif]    # empate: residuo de ponto flutuante nao conta
            dm = statistics.fmean(dif)
            ref_dm = float(r.get("dif_pp", r.get("dif_media_pp")))
            vit, der = sum(x < 0 for x in dif), sum(x > 0 for x in dif)
            if len(inst) != n_esp or abs(dm - ref_dm) > 1e-6 or vit != int(r["vit_a"]) or der != int(r["der_a"]):
                problemas.append(f"{arq} {r.get('id', '')} {a} vs {b} ({nv}): n={len(inst)} dif {dm:.5f} vs {ref_dm:.5f}; "
                                 f"vit/der {vit}/{der} vs {r['vit_a']}/{r['der_a']}")
            cont += 1
    print(f"(2) contrastes: {cont} conferidos (diferenca media, vitorias, derrotas)")
    print(f"execucoes lidas no total: {tot}")
    (RES / "verificacao_independente.json").write_text(json.dumps(
        {"medias": conferidos + outros, "contrastes": cont, "execucoes": tot, "ok": not problemas}), encoding="utf-8")
    if problemas:
        print("\nPROBLEMAS:")
        for p in problemas[:40]:
            print("  -", p)
    print("VERIFICACAO INDEPENDENTE:", "OK" if not problemas else f"{len(problemas)} PROBLEMA(S)")
    sys.exit(1 if problemas else 0)


if __name__ == "__main__":
    main()
