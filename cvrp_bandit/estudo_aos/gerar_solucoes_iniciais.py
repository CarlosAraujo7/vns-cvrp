#!/usr/bin/env python3
"""
gerar_solucoes_iniciais.py
==========================
Calcula UMA vez, sequencialmente e sem concorrencia, a solucao inicial
(Clarke-Wright + reparo de rotas) de todas as instancias dos experimentos
(74 CVRPLIB + SYN_TUNE + SYN_VAL) e grava em instances/_init/<nome>.json
(ver src/solucao_inicial.py e diario, Entrada 88). Nao sobrescreve arquivos
existentes (use --forcar). Roda no WSL (usa ortools no fallback).

Para as instancias cuja solucao inicial for NAO deterministica (fallback com
CP-SAT), a solucao gravada e a de ESTA execucao; o motivo e documentado no
JSON (`origem`). O arquivo `instances/_init/_relatorio.csv` lista, por
instancia, o numero de rotas, o custo inicial e se o reparo usou o fallback.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import construction  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from instance import load_all  # noqa: E402
from solucao_inicial import gravar_inicial, caminho, INIT_DIR  # noqa: E402

forcar = "--forcar" in sys.argv
usou_fallback = {"v": False}
_orig = construction._pack_exact_k_feasible


def _marcado(*a, **k):
    usou_fallback["v"] = True
    return _orig(*a, **k)


construction._pack_exact_k_feasible = _marcado

linhas = []
for conj in ("A", "B", "CMT", "X", "SYN_TUNE", "SYN_VAL"):
    for inst in load_all(ROOT / "instances" / conj):
        if caminho(inst.name).exists() and not forcar:
            print(f"  {inst.name}: ja existe, mantido")
            continue
        usou_fallback["v"] = False
        rotas = repair_route_count(inst, clarke_wright_savings(inst))
        viavel, motivo = inst.is_feasible(rotas)
        assert viavel, (inst.name, motivo)
        origem = "cws+reparo (determinístico)" if not usou_fallback["v"] else \
            "cws+reparo com FALLBACK CP-SAT (nao deterministico; esta e a solucao fixada)"
        gravar_inicial(inst, rotas, origem)
        linhas.append((inst.name, conj, len(rotas), inst.solution_cost(rotas), usou_fallback["v"]))
        print(f"  {inst.name:16s} K={len(rotas):3d} custo={inst.solution_cost(rotas):10.1f} fallback={usou_fallback['v']}", flush=True)

if linhas:
    rel = INIT_DIR / "_relatorio.csv"
    novo = not rel.exists()
    with open(rel, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["instancia", "conjunto", "n_rotas", "custo_inicial", "usou_fallback_cpsat"])
        w.writerows(linhas)
    print(f"\n{len(linhas)} solucoes gravadas; com fallback: {[l[0] for l in linhas if l[4]]}")
