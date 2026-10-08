#!/usr/bin/env python3
"""
medir_tempo_hgucb.py
=======================
Mede o tempo de PAREDE do nosso próprio HG-UCB (configuração de produção:
steps_max=20, max_inner_iter=40, alpha=0,5, reward_positive_normalized,
progressive_x_linha=True -- mesma configuração de `diag16_ablacao_74completas.py`,
braço `hgucb_completo`), por instância, nas 74 instâncias completas
(A=26/B=23/X=21/CMT=4), 5 seeds cada (SEED_BASE=1000, mesma convenção de
sempre).

Este é o PRIMEIRO passo do protocolo de tempo igualado aprovado (diário,
Entrada 34, decisão 3): o orçamento de tempo dado a OR-Tools/LKH3 em
`rodar_ortools_74.py`/`rodar_lkh3_74.py` é a média (sobre os 5 seeds) do
tempo medido AQUI, por instância -- por isso este script precisa rodar
PRIMEIRO, na MESMA máquina que vai rodar os outros dois (senão a
comparação de tempo perde o sentido -- é exatamente a falha que a Entrada
2 do diário documentou e que este protocolo existe para evitar).

IMPORTANTE: a MEDIDA em si é tempo de parede (`time.perf_counter`), então
rode isto sem outras cargas pesadas competindo por CPU na mesma máquina
(feche outros programas pesados) -- caso contrário o tempo medido fica
inflado e os solvers de comparação ganham um orçamento maior do que
deveriam.

Uso: python medir_tempo_hgucb.py
"""
import csv
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from instance import load_all  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from vns import bandit_vns  # noqa: E402
from bandit import HGUCBBandit, reward_positive_normalized  # noqa: E402

STEPS_MAX = 20
MAX_INNER_ITER = 40
ALPHA = 0.5
N_SEEDS = 5
SEED_BASE = 1000
CONJUNTOS = ("A", "B", "X", "CMT")


def rodar_uma_vez(inst, cws, seed):
    rng = random.Random(seed)
    bandit_obj = HGUCBBandit(n_operators=5, rng=rng, p_hat_mode="q_values", alpha=ALPHA)
    t0 = time.perf_counter()
    resultado = bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
        bandit=bandit_obj, max_inner_iter=MAX_INNER_ITER, progressive_x_linha=True,
        reward_positive_fn=reward_positive_normalized,
    )
    dt = time.perf_counter() - t0
    return resultado, dt


def main():
    print("=== Medindo tempo de parede do HG-UCB (configuração de produção) nas 74 instâncias ===\n")
    t_inicio_total = time.perf_counter()
    linhas_csv = []

    for set_name in CONJUNTOS:
        instancias = load_all(ROOT / "instances" / set_name)
        for inst in instancias:
            cws = repair_route_count(inst, clarke_wright_savings(inst))
            tempos, gaps = [], []
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                resultado, dt = rodar_uma_vez(inst, cws, seed)
                viavel, motivo = inst.is_feasible(resultado)
                if not viavel:
                    print(f"  ERRO: {inst.name} seed={seed} inviável: {motivo}")
                    sys.exit(1)
                custo = inst.solution_cost(resultado)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                tempos.append(dt)
                gaps.append(gap)
                linhas_csv.append({
                    "conjunto": set_name, "instancia": inst.name, "n_customers": inst.n_customers,
                    "seed": seed, "tempo_s": dt, "gap_pct": gap,
                })
            media_t = sum(tempos) / len(tempos)
            media_g = sum(gaps) / len(gaps)
            print(f"  {inst.name:16s} n={inst.n_customers:4d}  "
                  f"tempo_medio={media_t:7.3f}s (min={min(tempos):.3f} max={max(tempos):.3f})  "
                  f"gap_medio={media_g:6.2f}%")

    dt_total = time.perf_counter() - t_inicio_total
    print(f"\nTempo total de execução deste script: {dt_total:.1f}s ({dt_total/60:.1f}min)\n")

    out_csv = ROOT / "results" / "tempos_hgucb_74.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas_csv[0].keys()))
        w.writeheader()
        w.writerows(linhas_csv)
    print(f"CSV salvo em: {out_csv}")
    print("\nEste CSV é lido por rodar_ortools_74.py e rodar_lkh3_74.py para o orçamento de "
          "tempo por instância (média dos 5 seeds) -- rode-os depois deste.")


if __name__ == "__main__":
    main()
