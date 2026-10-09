#!/usr/bin/env python3
"""
diag_estrutura.py
=================
Diagnostico reprodutivel (EXPLORATORIO) da ineficiencia do laco original
(Algoritmo 3: selecao uniforme, aceitacao 'shaken', 40 aplicacoes por passo,
B=800, 5 operadores originais). Substitui o diagnostico de 3 instancias da
Entrada 70 por todas as 74 instancias e 10 seeds, e grava o resultado em
results/aos_study/diag_estrutura.csv (uma linha por instancia x seed).

Para cada passo de 40 aplicacoes mede (i) a fracao das aplicacoes que ocorre
DEPOIS da ultima melhora do passo e (ii) a fracao que ocorre depois do
primeiro instante em que os cinco operadores ja falharam desde a ultima
melhora -- isto e, aplicacoes que, sendo as vizinhancas deterministicas,
nao podem alterar a solucao (otimo local ja atingido). Passos incompletos
(fim do orcamento) sao descartados.

Uso (WSL, venv de direcao1_local): python3 -u diag_estrutura.py --workers 8
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
from run_aos_study import _instancia, _contar, todas_instancias  # noqa: E402
from neighborhoods import NEIGHBORHOODS  # noqa: E402
from aos_vnd import vnd_generico, construir_politica  # noqa: E402

B = 800
PASSO = 40
K = len(NEIGHBORHOODS)


def executar(args):
    set_name, nome, seed = args
    inst, cws = _instancia(set_name, nome)
    rng = random.Random(seed)
    contador, uso = [0], [0] * K
    nbhs = _contar(contador, uso)
    politica, modo = construir_politica("U", K, rng)
    log = []                                    # (operador, melhorou) na ordem de aplicacao
    vnd_generico(inst, [list(r) for r in cws], rng, B, nbhs, politica, "shaken", "F40", modo_recompensa=modo,
                 on_aplicacao=lambda j, ok, g: log.append((j, ok)))
    apos_ultima, apos_otimo, passos = 0, 0, 0
    for s in range(len(log) // PASSO):
        passo = log[s * PASSO:(s + 1) * PASSO]
        ultima = max((i for i, (_, ok) in enumerate(passo) if ok), default=-1)
        apos_ultima += PASSO - 1 - ultima
        falhos, otimo_em = set(), None
        for i, (j, ok) in enumerate(passo):
            if ok:
                falhos = set()
            else:
                falhos.add(j)
                if len(falhos) == K and otimo_em is None:
                    otimo_em = i
            if ok:
                otimo_em = None
        apos_otimo += (PASSO - 1 - otimo_em) if otimo_em is not None else 0
        passos += 1
    return dict(conjunto=set_name, instancia=inst.name, seed=seed, passos=passos,
                apos_ultima_melhora=apos_ultima, apos_otimo_local=apos_otimo, aplicacoes=passos * PASSO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    tarefas = [(s, n, 1000 + sd) for (s, n, _) in todas_instancias() for sd in range(a.seeds)]
    out = ROOT / "results" / "aos_study" / "diag_estrutura.csv"
    with mp.Pool(a.workers) as pool, open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["conjunto", "instancia", "seed", "passos", "aplicacoes",
                                          "apos_ultima_melhora", "apos_otimo_local"])
        w.writeheader()
        for i, linha in enumerate(pool.imap_unordered(executar, tarefas, chunksize=1), 1):
            w.writerow(linha)
            if i % 100 == 0:
                print(f"  {i}/{len(tarefas)}", flush=True)
    print("gravado", out)


if __name__ == "__main__":
    main()
