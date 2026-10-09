#!/usr/bin/env python3
"""
diag_complementaridade.py
=========================
Diagnostico EXPLORATORIO (sem inferencia confirmatoria; diario, Entrada 96) da explicacao proposta para o resultado
da Etapa 5: os operadores adicionados -- sobretudo os caros, CROSS e SWAP* -- acham melhoras em solucoes onde os cinco
operadores originais (P5) ja falharam ("complementaridade").

Procedimento (por instancia e seed; solucao inicial fixada em disco): repete R vezes
  shake -> VND de ordem fixa sobre P5 ate o otimo local de P5 -> em CADA operador adicionado
  (2-opt*, Or-opt, Swap-intra, CROSS, SWAP*) aplica UMA vez o melhor movimento na solucao em otimo local de P5 e
  registra se houve melhora estrita e o ganho relativo. A solucao seguinte do laco e o otimo local de P5 (aceitacao
  'best': so substitui a corrente se melhora), de modo que os otimos locais amostrados sao os que uma busca em P5 visita.
Saida: results/aos_study/diag_complementaridade.csv (uma linha por instancia x seed x repeticao).

Uso (WSL): python3 -u diag_complementaridade.py --workers 8
"""
import argparse
import csv
import multiprocessing as mp
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_aos_study import _instancia, todas_instancias  # noqa: E402
from run_aos_etapa3 import POOL_P10, POOL_P5  # noqa: E402
from vns import shake  # noqa: E402

ADICIONADOS = [5, 6, 7, 8, 9]            # indices em POOL_P10: 2-opt*, Or-opt, Swap-intra, CROSS, SWAP*
NOMES = {5: "2-opt*", 6: "Or-opt", 7: "Swap-intra", 8: "CROSS", 9: "SWAP*"}
R = 8


def vnd_p5(inst, routes):
    """VND de ordem fixa sobre os 5 operadores originais ate o otimo local de P5."""
    l = 0
    while l < len(POOL_P5):
        melhorou, novo = POOL_P5[l][1](inst, routes)
        if melhorou and inst.solution_cost(novo) < inst.solution_cost(routes) - 1e-9:
            routes, l = novo, 0
        else:
            l += 1
    return routes


def executar(args):
    set_name, nome, seed = args
    inst, cws = _instancia(set_name, nome)
    rng = random.Random(seed)
    x = [list(r) for r in cws]
    custo_x = inst.solution_cost(x)
    linhas = []
    for rep in range(R):
        xl = shake(inst, x, rng, POOL_P5)
        r = vnd_p5(inst, xl)
        c = inst.solution_cost(r)
        linha = dict(conjunto=set_name, instancia=inst.name, seed=seed, rep=rep)
        for j in ADICIONADOS:
            melhorou, novo = POOL_P10[j][1](inst, r)
            cn = inst.solution_cost(novo) if melhorou else c
            ok = melhorou and cn < c - 1e-9
            linha[f"ok_{NOMES[j]}"] = int(ok)
            linha[f"ganho_{NOMES[j]}"] = (c - cn) / c if ok else 0.0
        linha["ok_algum"] = int(any(linha[f"ok_{NOMES[j]}"] for j in ADICIONADOS))
        linhas.append(linha)
        if c < custo_x - 1e-9:
            x, custo_x = r, c
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    tarefas = [(s, n, 1000 + sd) for (s, n, _) in todas_instancias() for sd in range(a.seeds)]
    out = ROOT / "results" / "aos_study" / "diag_complementaridade.csv"
    campos = ["conjunto", "instancia", "seed", "rep"] + [f"{p}_{NOMES[j]}" for j in ADICIONADOS for p in ("ok", "ganho")] + ["ok_algum"]
    with mp.Pool(a.workers) as pool, open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for i, linhas in enumerate(pool.imap_unordered(executar, tarefas, chunksize=1), 1):
            w.writerows(linhas)
            if i % 20 == 0:
                f.flush()
                print(f"  {i}/{len(tarefas)}", flush=True)
    print("gravado", out)


if __name__ == "__main__":
    main()
