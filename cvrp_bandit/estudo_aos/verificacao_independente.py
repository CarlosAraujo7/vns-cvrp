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


def verifica_s7(num):
    """Estudo 7 (HGS): recomputa as medias de gap por braco/populacao/ponto de controle e todos os contrastes do CSV de analise
    (diferenca media, vitorias, derrotas) direto de etapa7_full.csv, so com a biblioteca padrao."""
    arq = RES / "etapa7_full.csv"
    if not arq.exists() or not (RES / "analise_etapa7_confirmatoria.csv").exists():
        return 0, 0
    por = defaultdict(list)                       # (braco, instancia, ponto) -> lista de gaps
    pop_de = {}
    inv7 = []
    with open(arq, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["viavel"] != "True":                  # excluidas pela regra pre-registrada; a contagem e conferida abaixo
                inv7.append((r["instancia"], r["braco"], r["seed"]))
                continue
            ref = float(r["ref"])
            if abs(100.0 * (float(r["custo_final"]) - ref) / ref - float(r["gap_final"])) > 1e-6:
                problemas.append(f"etapa7: gap_final divergente {r['instancia']} {r['braco']} {r['seed']}")
            pop_de[r["instancia"]] = "P2" if r["conjunto"] == "SYN_VAL" else "P1"
            for ck in (250, 500, 1000, 2000, 4000):
                v = r[f"gap_{ck}"]
                if v not in ("", "nan"):
                    por[(r["braco"], r["instancia"], ck)].append(float(v))
    medias = {k: statistics.fmean(v) for k, v in por.items()}
    if str(len(inv7)) != num.get("s7_ninv"):
        problemas.append(f"etapa7: {len(inv7)} execucoes inviaveis no CSV vs. macro s7_ninv = {num.get('s7_ninv')}")
    print(f"     etapa7: {len(inv7)} execucoes inviaveis pela convencao (instancias: {sorted({i for i, _, _ in inv7})}), excluidas")
    nm = 0
    for chave, valor in num.items():
        if chave.startswith("s7_gap_"):
            braco, pop, ck = chave[len("s7_gap_"):].split("_")
            vals = [m for (b, i, c), m in medias.items() if b == braco and c == int(ck) and pop_de[i] == pop]
            nm += confere(chave, valor, statistics.fmean(vals), 3)
    nc = 0
    for r in csv.DictReader(open(RES / "analise_etapa7_confirmatoria.csv", encoding="utf-8")):
        pops = {"P1": ["P1"], "P2": ["P2"], "P1+P2": ["P1", "P2"]}[r["pop"]]
        ck = int(float(r["ponto"]))
        inst = sorted({i for (b, i, c) in medias if c == ck and pop_de[i] in pops})
        dif = [medias[(r["a"], i, ck)] - medias[(r["b"], i, ck)] for i in inst]
        dif = [0.0 if abs(x) < 1e-9 else x for x in dif]
        dm, vit, der = statistics.fmean(dif), sum(x < 0 for x in dif), sum(x > 0 for x in dif)
        if abs(dm - float(r["dif_pp"])) > 1e-6 or vit != int(r["vit_a"]) or der != int(r["der_a"]) or len(inst) != int(r["n_inst"]):
            problemas.append(f"etapa7 {r['id']}: dif {dm:.5f} vs {float(r['dif_pp']):.5f}; vit/der {vit}/{der} vs {r['vit_a']}/{r['der_a']}")
        nc += 1
    return nm, nc


def verifica_s11(num):
    """Estudo 11: recomputa as medias de gap (macros `s11_gap_<braco>_<E1|E2|L>`) e a diferenca media, vitorias, derrotas e
    n de todos os contrastes de analise_etapa11_confirmatoria.csv direto dos CSV brutos (eval74: nivel 1 = E1, 5 = E2;
    evalL = SYN_VAL_L em E1), so com a biblioteca padrao."""
    a74, aL, an = RES / "etapa11_eval74.csv", RES / "etapa11_evalL.csv", RES / "analise_etapa11_confirmatoria.csv"
    if not (a74.exists() and aL.exists() and an.exists()):
        return 0, 0
    por = defaultdict(list)
    for arq, cenario_de in ((a74, lambda r: {"1": "E1", "5": "E2"}[r["nivel"]]), (aL, lambda r: "L")):
        with open(arq, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["viavel"] != "True":
                    problemas.append(f"{arq.name}: execucao inviavel {r['braco']} {r['instancia']} {r['seed']}")
                    continue
                ref = float(r["ref"])
                gap = 100.0 * (float(r["custo"]) - ref) / ref
                if abs(gap - float(r["gap_pct"])) > 1e-6:
                    problemas.append(f"{arq.name}: gap gravado divergente {r['braco']} {r['instancia']} {r['seed']}")
                por[(cenario_de(r), r["braco"], r["instancia"])].append(gap)
    medias = {k: statistics.fmean(v) for k, v in por.items()}
    nm = 0
    for chave, valor in num.items():
        if chave.startswith("s11_gap_"):
            resto = chave[len("s11_gap_"):]
            arm, cen = resto.rsplit("_", 1)
            vals = [m for (c_, b, _i), m in medias.items() if c_ == cen and b == "s11_" + arm]
            if not vals:
                problemas.append(f"macro {chave}: braco/cenario ausente nos CSV do Estudo 11")
                continue
            nm += confere(chave, valor, statistics.fmean(vals), 3)
    cod = {"CVRPLIB E1": "E1", "CVRPLIB E2": "E2", "SYN_VAL_L E1": "L"}
    nc = 0
    for r in csv.DictReader(open(an, encoding="utf-8")):
        cen = cod[r["cenario"]]
        na, nb = "s11_" + r["a"], "s11_" + r["b"]
        ia = {i: m for (c_, b, i), m in medias.items() if c_ == cen and b == na}
        ib = {i: m for (c_, b, i), m in medias.items() if c_ == cen and b == nb}
        inst = sorted(set(ia) & set(ib))
        dif = [ia[i] - ib[i] for i in inst]
        dif = [0.0 if abs(x) < 1e-9 else x for x in dif]
        if not dif:
            problemas.append(f"{an.name} {r['id']}: sem instancias comuns")
            continue
        dm, vit, der = statistics.fmean(dif), sum(x < 0 for x in dif), sum(x > 0 for x in dif)
        if abs(dm - float(r["dif_pp"])) > 1e-6 or vit != int(r["vit_a"]) or der != int(r["der_a"]) or len(inst) != int(r["n_inst"]):
            problemas.append(f"{an.name} {r['id']}: dif {dm:.5f} vs {float(r['dif_pp']):.5f}; vit/der {vit}/{der} vs "
                             f"{r['vit_a']}/{r['der_a']}; n {len(inst)} vs {r['n_inst']}")
        nc += 1
    return nm, nc


def verifica_s13(num):
    """Estudo 13 (TSP): recomputa, so com a biblioteca padrao, as medias de gap dos macros `s13_gap_*`, `s13_pcgap_*`, `s13a_gap_*` e
    `s13r_gap_*` e a diferenca media, vitorias, derrotas e n de todos os contrastes dos CSV de analise do Estudo 13 (VND, pos-hoc P5,
    pos-hoc de ordem fixa, controle positivo, ALNS e pos-hoc ALNS), direto dos CSV brutos (media das seeds por instancia)."""
    brutos = RES / "etapa13_eval_v.csv"
    if not brutos.exists() or not (RES / "analise_etapa13_v.csv").exists():
        return 0, 0
    por = defaultdict(list)
    inst_conj = {}
    nivel_rot = {"1": "E1", "5": "E2", "25": "E3"}

    def ler13(nome_csv, cen_de, braco_de):
        arq = RES / nome_csv
        if not arq.exists():
            return
        with open(arq, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["viavel"] != "True":
                    problemas.append(f"{nome_csv}: execucao inviavel {r['braco']} {r['instancia']} {r['seed']}")
                    continue
                ref = float(r["ref"])
                gap = 100.0 * (float(r["custo"]) - ref) / ref
                if abs(gap - float(r["gap_pct"])) > 1e-6:
                    problemas.append(f"{nome_csv}: gap gravado divergente {r['braco']} {r['instancia']} {r['seed']}")
                inst_conj[r["instancia"]] = r["conjunto"]
                por[(cen_de(r), braco_de(r["braco"]), r["instancia"])].append(gap)

    def vnd_cen(r):
        return f"{r['conjunto']} {nivel_rot[str(int(float(r['nivel'])))]}"

    strip_v = lambda b: b.replace("s13v_", "").replace("s13x_", "")    # noqa: E731
    ler13("etapa13_eval_v.csv", vnd_cen, strip_v)
    ler13("etapa13_explor_ordem.csv", vnd_cen, strip_v)
    ler13("etapa13_explor_perm.csv", vnd_cen, strip_v)
    ler13("etapa13_eval_pc.csv", lambda r: r["braco"].split("_")[1], lambda b: b.split("_", 2)[2])
    ler13("etapa13_eval_a.csv", lambda r: "ALNS", lambda b: b.replace("s13a_", ""))
    ler13("etapa13_eval_ref.csv", lambda r: "ALNS", lambda b: "vnd_" + b.replace("s13r_", ""))
    medias = {k: statistics.fmean(v) for k, v in por.items()}
    nm = 0
    cenmac = {"T1": "TSPLIB E1", "T2": "TSPLIB E2", "S1": "SYN_VAL_T E1", "T3": "TSPLIB E3"}
    for chave, valor in num.items():
        vals = None
        if chave.startswith("s13_gap_"):
            arm, k = chave[len("s13_gap_"):].rsplit("_", 1)
            vals = [m for (c_, b, _i), m in medias.items() if c_ == cenmac[k] and b == arm]
        elif chave.startswith("s13_pcgap_"):
            arm, tag = chave[len("s13_pcgap_"):].rsplit("_", 1)
            vals = [m for (c_, b, _i), m in medias.items() if c_ == tag and b == arm]
        elif chave.startswith("s13a_gap_"):
            arm, k = chave[len("s13a_gap_"):].rsplit("_", 1)
            sel = {"all": None, "T": "TSPLIB", "S": "SYN_VAL_T"}[k]
            vals = [m for (c_, b, i), m in medias.items() if c_ == "ALNS" and b == arm and (sel is None or inst_conj[i] == sel)]
        elif chave.startswith("s13r_gap_"):
            arm = "vnd_" + chave[len("s13r_gap_"):]
            vals = [m for (c_, b, _i), m in medias.items() if c_ == "ALNS" and b == arm]
        if vals is None:
            continue
        if not vals:
            problemas.append(f"macro {chave}: braco/cenario ausente nos CSV do Estudo 13")
            continue
        nm += confere(chave, valor, statistics.fmean(vals), 3)

    def confere_contrastes(nome_an, resolve):
        arq = RES / nome_an
        if not arq.exists():
            return 0
        n = 0
        for r in csv.DictReader(open(arq, encoding="utf-8")):
            cen, na, nb, conj = resolve(r)
            ia = {i: m for (c_, b, i), m in medias.items() if c_ == cen and b == na and (conj is None or inst_conj[i] == conj)}
            ib = {i: m for (c_, b, i), m in medias.items() if c_ == cen and b == nb and (conj is None or inst_conj[i] == conj)}
            inst = sorted(set(ia) & set(ib))
            dif = [ia[i] - ib[i] for i in inst]
            dif = [0.0 if abs(x) < 1e-9 else x for x in dif]
            if not dif:
                problemas.append(f"{nome_an} {r['id']}: sem instancias comuns")
                continue
            dm, vit, der = statistics.fmean(dif), sum(x < 0 for x in dif), sum(x > 0 for x in dif)
            if abs(dm - float(r["dif_pp"])) > 1e-6 or vit != int(r["vit_a"]) or der != int(r["der_a"]) or len(inst) != int(r["n_inst"]):
                problemas.append(f"{nome_an} {r['id']}: dif {dm:.5f} vs {float(r['dif_pp']):.5f}; vit/der {vit}/{der} vs "
                                 f"{r['vit_a']}/{r['der_a']}; n {len(inst)} vs {r['n_inst']}")
            n += 1
        return n

    por_cenario = lambda r: (r["cenario"], r["a"], r["b"], None)    # noqa: E731
    nc = 0
    for nome_an in ("analise_etapa13_v.csv", "analise_etapa13_v_posthoc.csv", "analise_etapa13_ordem.csv", "analise_etapa13_perm.csv"):
        nc += confere_contrastes(nome_an, por_cenario)
    nc += confere_contrastes("analise_etapa13_pc.csv", lambda r: (r["pool"], r["a"], r["b"], None))

    def res_a(r):
        conj = r.get("conjunto") or None
        return ("ALNS", r["a"].replace("alns_", ""), r["b"], conj)
    nc += confere_contrastes("analise_etapa13_a.csv", res_a)
    nc += confere_contrastes("analise_etapa13_a_posthoc.csv", res_a)
    return nm, nc


def verifica_s14(num):
    """Estudo 14 (CVRPTW): recomputa, so com a biblioteca padrao, as medias de gap dos macros `s14_gap_*` e `s14_pcgap_*` e a diferenca media,
    vitorias, derrotas e n de todos os contrastes de analise_etapa14_v.csv e analise_etapa14_pc.csv, direto dos CSV brutos (media das
    seeds por instancia; nivel 1 = E1, 5 = E2, 25 = E3; controle positivo em E1)."""
    ev, pcf = RES / "etapa14_eval_v.csv", RES / "etapa14_eval_pc.csv"
    if not (ev.exists() and pcf.exists() and (RES / "analise_etapa14_v.csv").exists()):
        return 0, 0
    por = defaultdict(list)
    rot = {"1": "E1", "5": "E2", "25": "E3"}

    def ler14(arq, cen_de, braco_de):
        with open(arq, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["viavel"] != "True":
                    problemas.append(f"{arq.name}: execucao inviavel {r['braco']} {r['instancia']} {r['seed']}")
                    continue
                ref = float(r["ref"])
                gap = 100.0 * (float(r["custo"]) - ref) / ref
                if abs(gap - float(r["gap_pct"])) > 1e-6:
                    problemas.append(f"{arq.name}: gap gravado divergente {r['braco']} {r['instancia']} {r['seed']}")
                por[(cen_de(r), braco_de(r["braco"]), r["instancia"])].append(gap)

    ler14(ev, lambda r: rot[str(int(float(r["nivel"])))], lambda b: b.replace("s14v_", ""))
    ler14(pcf, lambda r: r["braco"].split("_")[1] + "_" + r["braco"].split("_")[2], lambda b: b.split("_", 3)[3])
    if (RES / "etapa14_explor_perm.csv").exists():
        ler14(RES / "etapa14_explor_perm.csv", lambda r: rot[str(int(float(r["nivel"])))], lambda b: b.replace("s14x_", ""))
    medias = {k: statistics.fmean(v) for k, v in por.items()}
    nm = 0
    for chave, valor in num.items():
        vals = None
        if chave.startswith("s14_gap_"):
            arm, k = chave[len("s14_gap_"):].rsplit("_", 1)
            vals = [m for (c_, b, _i), m in medias.items() if c_ == k and b == arm]
        elif chave.startswith("s14_pcgap_"):
            resto = chave[len("s14_pcgap_"):]
            arm, tag = resto.split("_D_")
            vals = [m for (c_, b, _i), m in medias.items() if c_ == "D_" + tag and b == arm]
        if vals is None:
            continue
        if not vals:
            problemas.append(f"macro {chave}: braco/cenario ausente nos CSV do Estudo 14")
            continue
        nm += confere(chave, valor, statistics.fmean(vals), 3)
    nc = 0
    for nome_an, resolve in (("analise_etapa14_v.csv", lambda r: (r["cenario"], r["a"], r["b"])),
                             ("analise_etapa14_pc.csv", lambda r: (r["pool"], r["a"], r["b"])),
                             ("analise_etapa14_perm.csv", lambda r: (r["cenario"], r["a"], r["b"]))):
        for r in csv.DictReader(open(RES / nome_an, encoding="utf-8")):
            cen, na, nb = resolve(r)
            ia = {i: m for (c_, b, i), m in medias.items() if c_ == cen and b == na}
            ib = {i: m for (c_, b, i), m in medias.items() if c_ == cen and b == nb}
            inst = sorted(set(ia) & set(ib))
            dif = [ia[i] - ib[i] for i in inst]
            dif = [0.0 if abs(x) < 1e-9 else x for x in dif]
            if not dif:
                problemas.append(f"{nome_an} {r['id']}: sem instancias comuns")
                continue
            dm, vit, der = statistics.fmean(dif), sum(x < 0 for x in dif), sum(x > 0 for x in dif)
            if abs(dm - float(r["dif_pp"])) > 1e-6 or vit != int(r["vit_a"]) or der != int(r["der_a"]) or len(inst) != int(r["n_inst"]):
                problemas.append(f"{nome_an} {r['id']}: dif {dm:.5f} vs {float(r['dif_pp']):.5f}; vit/der {vit}/{der} vs "
                                 f"{r['vit_a']}/{r['der_a']}; n {len(inst)} vs {r['n_inst']}")
            nc += 1
    return nm, nc


def verifica_s15(num):
    """Estudo 15: recomputa, so com a biblioteca padrao, as medias de gap dos macros `s15_gap_*` (conjunto primario, E1) e a diferenca media,
    vitorias, derrotas e n dos contrastes de analise_etapa15_prim.csv e analise_etapa15_foc.csv, direto dos CSV brutos."""
    arqs = {"t": (RES / "etapa15_eval_t.csv", "TSPLIB"), "w": (RES / "etapa15_eval_w.csv", "SOLOMON")}
    if not all(a.exists() for a, _ in arqs.values()) or not (RES / "analise_etapa15_prim.csv").exists():
        return 0, 0
    por = defaultdict(list)
    for prob, (arq, conj) in arqs.items():
        with open(arq, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["viavel"] != "True":
                    problemas.append(f"{arq.name}: execucao inviavel {r['braco']} {r['instancia']} {r['seed']}")
                    continue
                if r["conjunto"] != conj:
                    continue
                ref = float(r["ref"])
                gap = 100.0 * (float(r["custo"]) - ref) / ref
                if abs(gap - float(r["gap_pct"])) > 1e-6:
                    problemas.append(f"{arq.name}: gap gravado divergente {r['braco']} {r['instancia']} {r['seed']}")
                _, pool, arm = r["braco"].split("_", 2)
                por[(prob, pool, arm, r["instancia"])].append(gap)
    medias = {k: statistics.fmean(v) for k, v in por.items()}
    nm = 0
    for chave, valor in num.items():
        if chave.startswith("s15_gap_"):
            kk, arm = chave[len("s15_gap_"):].split("_", 1)
            vals = [m for (p_, pool, a, _i), m in medias.items() if p_ == kk[0] and pool == "P" + kk[1:] and a == arm]
            if not vals:
                problemas.append(f"macro {chave}: ausente nos CSV do Estudo 15")
                continue
            nm += confere(chave, valor, statistics.fmean(vals), 3)
    nc = 0
    for nome_an in ("analise_etapa15_prim.csv", "analise_etapa15_foc.csv"):
        for r in csv.DictReader(open(RES / nome_an, encoding="utf-8")):
            ia = {i: m for (p_, pool, a, i), m in medias.items() if p_ == r["prob"] and pool == r["pool"] and a == r["a"]}
            ib = {i: m for (p_, pool, a, i), m in medias.items() if p_ == r["prob"] and pool == r["pool"] and a == r["b"]}
            inst = sorted(set(ia) & set(ib))
            dif = [ia[i] - ib[i] for i in inst]
            dif = [0.0 if abs(x) < 1e-9 else x for x in dif]
            if not dif:
                problemas.append(f"{nome_an} {r['id']}: sem instancias comuns")
                continue
            dm, vit, der = statistics.fmean(dif), sum(x < 0 for x in dif), sum(x > 0 for x in dif)
            if abs(dm - float(r["dif_pp"])) > 1e-6 or vit != int(r["vit_a"]) or der != int(r["der_a"]) or len(inst) != int(r["n_inst"]):
                problemas.append(f"{nome_an} {r['id']}: dif {dm:.5f} vs {float(r['dif_pp']):.5f}; vit/der {vit}/{der} vs {r['vit_a']}/{r['der_a']}")
            nc += 1
    return nm, nc


def verifica_s16(num):
    """Estudo 16 (ALNS no CVRPTW): recomputa, so com a biblioteca padrao, as medias de gap dos macros `s16a_gap_*` e `s16r_gap_*` e a
    diferenca media, vitorias, derrotas e n de todos os contrastes de analise_etapa16_a.csv e analise_etapa16_classe.csv, direto dos CSV
    brutos (media das seeds por instancia; braco do ALNS `alns_X` = `s16a_X`; referencias VND `vnd_Y` = `s16r_Y`)."""
    import re
    ea, er, an = RES / "etapa16_eval_a.csv", RES / "etapa16_eval_ref.csv", RES / "analise_etapa16_a.csv"
    if not (ea.exists() and er.exists() and an.exists()):
        return 0, 0
    por = defaultdict(list)
    for arq, pref, nova in ((ea, "s16a_", ""), (er, "s16r_", "vnd_")):
        with open(arq, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["viavel"] != "True":
                    problemas.append(f"{arq.name}: execucao inviavel {r['braco']} {r['instancia']} {r['seed']}")
                    continue
                ref = float(r["ref"])
                gap = 100.0 * (float(r["custo"]) - ref) / ref
                if abs(gap - float(r["gap_pct"])) > 1e-6:
                    problemas.append(f"{arq.name}: gap gravado divergente {r['braco']} {r['instancia']} {r['seed']}")
                por[(nova + r["braco"][len(pref):], r["instancia"])].append(gap)
    medias = {k: statistics.fmean(v) for k, v in por.items()}
    nm = 0
    for chave, valor in num.items():
        if chave.startswith("s16a_gap_"):
            arm = chave[len("s16a_gap_"):]
        elif chave.startswith("s16r_gap_"):
            arm = "vnd_" + chave[len("s16r_gap_"):]
        else:
            continue
        vals = [m for (b, _i), m in medias.items() if b == arm]
        if not vals:
            problemas.append(f"macro {chave}: braco ausente nos CSV do Estudo 16")
            continue
        nm += confere(chave, valor, statistics.fmean(vals), 3)

    def nome(b):
        return b[len("alns_"):] if b.startswith("alns_") else b

    def classe(inst):
        m = re.match(r"([A-Z]+)(\d)", inst)
        return m.group(1) + m.group(2)

    nc = 0
    for nome_an in ("analise_etapa16_a.csv", "analise_etapa16_classe.csv"):
        for r in csv.DictReader(open(RES / nome_an, encoding="utf-8")):
            ia = {i: m for (b, i), m in medias.items() if b == nome(r["a"])}
            ib = {i: m for (b, i), m in medias.items() if b == nome(r["b"])}
            inst = sorted(set(ia) & set(ib))
            if nome_an.endswith("classe.csv"):
                inst = [i for i in inst if classe(i) == r["classe"]]
            dif = [ia[i] - ib[i] for i in inst]
            dif = [0.0 if abs(x) < 1e-9 else x for x in dif]
            if not dif:
                problemas.append(f"{nome_an} {r['id']}: sem instancias comuns")
                continue
            dm, vit, der = statistics.fmean(dif), sum(x < 0 for x in dif), sum(x > 0 for x in dif)
            if abs(dm - float(r["dif_pp"])) > 1e-6 or vit != int(r["vit_a"]) or der != int(r["der_a"]) or len(inst) != int(r["n_inst"]):
                problemas.append(f"{nome_an} {r['id']}: dif {dm:.5f} vs {float(r['dif_pp']):.5f}; vit/der {vit}/{der} vs "
                                 f"{r['vit_a']}/{r['der_a']}; n {len(inst)} vs {r['n_inst']}")
            nc += 1
    return nm, nc


def verifica_s17(num):
    """Estudo 17 (PyVRP): recomputa, so com a biblioteca padrao, as medias de gap dos macros `s17_gap_<braco>_<nivel>` e a diferenca media,
    vitorias, derrotas e n de todos os contrastes de analise_etapa17_a.csv e analise_etapa17_conjunto.csv, direto de etapa17_eval.csv
    (media das seeds VIAVEIS por instancia; execucoes inviaveis sao excluidas pela regra pre-registrada e contadas, nao sao erro)."""
    ev, an = RES / "etapa17_eval.csv", RES / "analise_etapa17_a.csv"
    if not (ev.exists() and an.exists()):
        return 0, 0
    por, niveis = defaultdict(list), set()
    with open(ev, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            niveis.add(float(r["nivel"]))
            if r["viavel"] != "True":
                continue
            ref = float(r["ref"])
            gap = 100.0 * (float(r["custo"]) - ref) / ref
            if abs(gap - float(r["gap_pct"])) > 1e-6:
                problemas.append(f"etapa17_eval.csv: gap gravado divergente {r['braco']} {r['instancia']} {r['seed']}")
            por[(float(r["nivel"]), r["braco"][len("s17_"):], r["instancia"])].append(gap)
    ordem = {nv: k for k, nv in enumerate(sorted(niveis), 1)}
    medias = {(ordem[nv], b, i): statistics.fmean(vs) for (nv, b, i), vs in por.items()}
    nm = 0
    for chave, valor in num.items():
        if not chave.startswith("s17_gap_"):
            continue
        resto = chave[len("s17_gap_"):]
        arm, k = resto.rsplit("_", 1)
        # so as instancias com TODOS os bracos (como `pivo` da analise): intersecao
        por_braco = defaultdict(dict)
        for (kk, b, i), m in medias.items():
            if kk == int(k):
                por_braco[b][i] = m
        comuns = set.intersection(*[set(d) for d in por_braco.values()])
        vals = [por_braco[arm][i] for i in comuns]
        if not vals:
            problemas.append(f"macro {chave}: braco ausente nos CSV do Estudo 17")
            continue
        nm += confere(chave, valor, statistics.fmean(vals), 3)

    def familia(inst):
        return "SYN_VAL" if inst.startswith("X-syn") else ("CMT" if inst.startswith("CMT") else inst[0])

    nc = 0
    for nome_an in ("analise_etapa17_a.csv", "analise_etapa17_conjunto.csv"):
        for r in csv.DictReader(open(RES / nome_an, encoding="utf-8")):
            k = int(r["nivel"])
            ia = {i: m for (kk, b, i), m in medias.items() if kk == k and b == r["a"]}
            ib = {i: m for (kk, b, i), m in medias.items() if kk == k and b == r["b"]}
            comuns = set(ia) & set(ib)
            # a analise usa so as instancias com todos os bracos do nivel (dropna): replica a intersecao completa
            todos = defaultdict(set)
            for (kk, b, i), m in medias.items():
                if kk == k:
                    todos[b].add(i)
            comuns &= set.intersection(*todos.values())
            if nome_an.endswith("conjunto.csv"):
                cj = r["conjunto"]
                if cj == "CVRPLIB":
                    comuns = {i for i in comuns if familia(i) != "SYN_VAL"}
                elif cj == "SYN_VAL":
                    comuns = {i for i in comuns if familia(i) == "SYN_VAL"}
                else:
                    comuns = {i for i in comuns if familia(i) == cj}
            inst = sorted(comuns)
            dif = [ia[i] - ib[i] for i in inst]
            dif = [0.0 if abs(x) < 1e-9 else x for x in dif]
            if not dif:
                problemas.append(f"{nome_an} {r['id']}: sem instancias comuns")
                continue
            dm, vit, der = statistics.fmean(dif), sum(x < 0 for x in dif), sum(x > 0 for x in dif)
            if abs(dm - float(r["dif_pp"])) > 1e-6 or vit != int(r["vit_a"]) or der != int(r["der_a"]) or len(inst) != int(r["n_inst"]):
                problemas.append(f"{nome_an} {r['id']}: dif {dm:.5f} vs {float(r['dif_pp']):.5f}; vit/der {vit}/{der} vs "
                                 f"{r['vit_a']}/{r['der_a']}; n {len(inst)} vs {r['n_inst']}")
            nc += 1
    return nm, nc


def verifica_s18(num):
    """Estudo 18 (politicas com estado, esforco de ajuste igual): recomputa, so com a biblioteca padrao, as medias de gap dos macros `s18_gap_<braco>`
    e a diferenca media, vitorias, derrotas e n de todos os contrastes de analise_etapa18.csv e analise_etapa18_classe.csv, direto dos CSV brutos
    (media das seeds por instancia; bracos novos `s18_X` = X; referencias do Estudo 16 `s16a_X` = X, exceto RP -> RP16 e static -> static16)."""
    import re
    e18, e16, an = RES / "etapa18_eval.csv", RES / "etapa16_eval_a.csv", RES / "analise_etapa18.csv"
    if not (e18.exists() and e16.exists() and an.exists()):
        return 0, 0
    por = defaultdict(list)
    for arq, pref in ((e18, "s18_"), (e16, "s16a_")):
        with open(arq, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["viavel"] != "True":
                    problemas.append(f"{arq.name}: execucao inviavel {r['braco']} {r['instancia']} {r['seed']}")
                    continue
                ref = float(r["ref"])
                gap = 100.0 * (float(r["custo"]) - ref) / ref
                if abs(gap - float(r["gap_pct"])) > 1e-6:
                    problemas.append(f"{arq.name}: gap gravado divergente {r['braco']} {r['instancia']} {r['seed']}")
                nome = r["braco"][len(pref):]
                if pref == "s16a_":
                    nome = {"RP": "RP16", "static": "static16"}.get(nome, nome)
                por[(nome, r["instancia"])].append(gap)
    medias = {k: statistics.fmean(v) for k, v in por.items()}
    nm = 0
    for chave, valor in num.items():
        if not chave.startswith("s18_gap_"):
            continue
        arm = chave[len("s18_gap_"):]
        vals = [m for (b, _i), m in medias.items() if b == arm]
        if not vals:
            problemas.append(f"macro {chave}: braco ausente nos CSV do Estudo 18")
            continue
        nm += confere(chave, valor, statistics.fmean(vals), 3)

    def classe(inst):
        m = re.match(r"([A-Z]+)(\d)", inst)
        return m.group(1) + m.group(2)

    nc = 0
    for nome_an in ("analise_etapa18.csv", "analise_etapa18_classe.csv"):
        for r in csv.DictReader(open(RES / nome_an, encoding="utf-8")):
            ia = {i: m for (b, i), m in medias.items() if b == r["a"]}
            ib = {i: m for (b, i), m in medias.items() if b == r["b"]}
            inst = sorted(set(ia) & set(ib))
            if nome_an.endswith("classe.csv"):
                inst = [i for i in inst if classe(i) == r["classe"]]
            dif = [ia[i] - ib[i] for i in inst]
            dif = [0.0 if abs(x) < 1e-9 else x for x in dif]
            if not dif:
                problemas.append(f"{nome_an} {r['id']}: sem instancias comuns")
                continue
            dm, vit, der = statistics.fmean(dif), sum(x < 0 for x in dif), sum(x > 0 for x in dif)
            if abs(dm - float(r["dif_pp"])) > 1e-6 or vit != int(r["vit_a"]) or der != int(r["der_a"]) or len(inst) != int(r["n_inst"]):
                problemas.append(f"{nome_an} {r['id']}: dif {dm:.5f} vs {float(r['dif_pp']):.5f}; vit/der {vit}/{der} vs "
                                 f"{r['vit_a']}/{r['der_a']}; n {len(inst)} vs {r['n_inst']}")
            nc += 1
    return nm, nc


def verifica_gaps(eval_csv, analise_csv, num, prefixo, braco_de_chave, extra_csvs=()):
    """Estudos 8-10: recomputa, so com a biblioteca padrao, a media de gap de cada braco (macros `<prefixo>_gap_*`) e a
    diferenca media, vitorias e derrotas de todos os contrastes do CSV de analise, direto do CSV bruto (media das seeds por
    instancia). Colunas esperadas no bruto: braco, instancia, custo, ref, gap_pct, viavel."""
    arq = RES / eval_csv
    if not arq.exists() or not (RES / analise_csv).exists():
        return 0, 0
    por = defaultdict(list)
    for nome_csv in (eval_csv,) + tuple(c for c in extra_csvs if (RES / c).exists()):
      with open(RES / nome_csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["viavel"] != "True":
                problemas.append(f"{nome_csv}: execucao inviavel {r['braco']} {r['instancia']} {r['seed']}")
                continue
            ref = float(r["ref"])
            gap = 100.0 * (float(r["custo"]) - ref) / ref
            if abs(gap - float(r["gap_pct"])) > 1e-6:
                problemas.append(f"{nome_csv}: gap gravado divergente {r['braco']} {r['instancia']} {r['seed']}")
            por[(r["braco"], r["instancia"])].append(gap)
    medias = {k: statistics.fmean(v) for k, v in por.items()}
    nm = 0
    for chave, valor in num.items():
        if chave.startswith(prefixo + "_gap_"):
            braco = braco_de_chave(chave)
            vals = [m for (b, i), m in medias.items() if b == braco]
            if not vals:
                problemas.append(f"macro {chave}: braco {braco} ausente em {eval_csv}")
                continue
            nm += confere(chave, valor, statistics.fmean(vals), 3 if prefixo in ("s9", "s10", "s12") else 2)
    nc = 0
    bracos = {b for (b, _i) in medias}
    for r in csv.DictReader(open(RES / analise_csv, encoding="utf-8")):
        def _nome(x):
            return next((cand for cand in (x, prefixo + "_" + x, "s9_" + x, "s12_" + x) if cand in bracos), x)
        na, nb = _nome(r["a"]), _nome(r["b"])
        ia = {i: m for (b, i), m in medias.items() if b == na}
        ib = {i: m for (b, i), m in medias.items() if b == nb}
        inst = sorted(set(ia) & set(ib))
        if r["id"].endswith("@X"):                        # contrastes por conjunto (Estudo 9): X = nomes sem "syn"
            inst = [i for i in inst if "syn" not in i]
        elif r["id"].endswith("@SYN_VAL"):
            inst = [i for i in inst if "synv" in i]
        dif = [ia[i] - ib[i] for i in inst]
        dif = [0.0 if abs(x) < 1e-9 else x for x in dif]
        if not dif:
            problemas.append(f"{analise_csv} {r['id']}: sem instancias comuns")
            continue
        dm, vit, der = statistics.fmean(dif), sum(x < 0 for x in dif), sum(x > 0 for x in dif)
        if abs(dm - float(r["dif_pp"])) > 1e-6 or vit != int(r["vit_a"]) or der != int(r["der_a"]) or len(inst) != int(r["n_inst"]):
            problemas.append(f"{analise_csv} {r['id']}: dif {dm:.5f} vs {float(r['dif_pp']):.5f}; vit/der {vit}/{der} vs "
                             f"{r['vit_a']}/{r['der_a']}; n {len(inst)} vs {r['n_inst']}")
        nc += 1
    return nm, nc


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
    n7m, n7c = verifica_s7(num)
    outros += n7m
    cont += n7c
    print(f"(2b) Estudo 7 (HGS): {n7m} medias e {n7c} contrastes conferidos")
    for rot, ev, an, pref, bk, *ex in (("Estudo 8 (controle positivo)", "etapa8_full.csv", "analise_etapa8_confirmatoria.csv", "s8",
                                   lambda k: "s8_" + ("P10+" if k[len("s8_gap_"):].startswith("D4") else "") + k[len("s8_gap_"):]),
                                  ("Estudo 9 (ALNS)", "etapa9_eval.csv", "analise_etapa9_confirmatoria.csv", "s9",
                                   lambda k: "s9_" + k[len("s9_gap_"):], ("etapa9_vns.csv",)),
                                  ("Estudo 10 (tamanho)", "etapa10_eval.csv", "analise_etapa10_confirmatoria.csv", "s10",
                                   lambda k: "s10_" + k[len("s10_gap_"):]),
                                  ("Estudo 12 (controle estatico no ALNS)", "etapa12_eval.csv", "analise_etapa12_confirmatoria.csv", "s12",
                                   lambda k: ("s12_" if k[len("s12_gap_"):] in ("static", "reg2") else "s9_") + k[len("s12_gap_"):],
                                   ("etapa9_eval.csv",))):
        nm_, nc_ = verifica_gaps(ev, an, num, pref, bk, tuple(ex[0]) if ex else ())
        outros += nm_
        cont += nc_
        if nm_ or nc_:
            print(f"(2c) {rot}: {nm_} medias e {nc_} contrastes conferidos")
    n11m, n11c = verifica_s11(num)
    outros += n11m
    cont += n11c
    if n11m or n11c:
        print(f"(2c) Estudo 11 (SWAP* eficiente): {n11m} medias e {n11c} contrastes conferidos")
    n13m, n13c = verifica_s13(num)
    outros += n13m
    cont += n13c
    if n13m or n13c:
        print(f"(2c) Estudo 13 (TSP): {n13m} medias e {n13c} contrastes conferidos")
    n14m, n14c = verifica_s14(num)
    outros += n14m
    cont += n14c
    if n14m or n14c:
        print(f"(2c) Estudo 14 (CVRPTW): {n14m} medias e {n14c} contrastes conferidos")
    n15m, n15c = verifica_s15(num)
    outros += n15m
    cont += n15c
    if n15m or n15c:
        print(f"(2c) Estudo 15 (teste prospectivo): {n15m} medias e {n15c} contrastes conferidos")
    n16m, n16c = verifica_s16(num)
    outros += n16m
    cont += n16c
    if n16m or n16c:
        print(f"(2c) Estudo 16 (ALNS no CVRPTW): {n16m} medias e {n16c} contrastes conferidos")
    n17m, n17c = verifica_s17(num)
    outros += n17m
    cont += n17c
    if n17m or n17c:
        print(f"(2c) Estudo 17 (PyVRP): {n17m} medias e {n17c} contrastes conferidos")
    n18m, n18c = verifica_s18(num)
    outros += n18m
    cont += n18c
    if n18m or n18c:
        print(f"(2c) Estudo 18 (politicas com estado, ajuste igual): {n18m} medias e {n18c} contrastes conferidos")
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
