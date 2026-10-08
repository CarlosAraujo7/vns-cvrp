#!/usr/bin/env python3
"""
diag17_coldstart_maisseeds.py
================================
Complemento ao Diagnóstico 16 (Entrada 31): dos 4 mecanismos testados na
configuração de produção (alpha=0,5) nas 74 instâncias completas, só o
cold-start manteve um efeito consistente em direção e magnitude entre
n=16 (Entrada 29) e n=74 (Entrada 31), chegando perto do limiar de
significância sem cruzá-lo (teste t p=0,0708, Wilcoxon p=0,1004). Decisão
do pesquisador (07/09/2026, via `AskUserQuestion`, respondendo à Entrada
31): rodar mais seeds SÓ para este par (`hgucb_completo` vs.
`sem_cold_start`), em vez de mais instâncias (já são as 74 completas) ou
de encerrar a Direção 2 aqui.

Roda 10 seeds NOVOS (1005-1014, continuando a sequência SEED_BASE=1000 já
usada em todo este projeto -- nunca repete um seed já rodado) para os 2
braços em questão, nas 74 instâncias completas -- protocolo idêntico ao
Diagnóstico 16 em tudo mais (steps_max=20, max_inner_iter=40,
reward_positive_normalized, progressive_x_linha=True, alpha=0,5).
Combinados com os 5 seeds já existentes no CSV do Diagnóstico 16
(results/diag16_ablacao_74completas.csv), totalizam 15 seeds por
instância -- 3x o N original, reduzindo o ruído de cada média-por-
instância antes de repetir o teste de significância.

Uso: python diag17_coldstart_maisseeds.py
"""
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from vns import bandit_vns  # noqa: E402
from bandit import HGUCBBandit, reward_positive_normalized  # noqa: E402
from ablation_randomness_bandits import NoColdStartBandit  # noqa: E402
from run_bandit_vns_validation import PAPER_HGUCB_POR_INSTANCIA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STEPS_MAX = 20
SEED_BASE_NOVO = 1005  # continua a sequencia 1000-1004 ja usada no diag16
N_SEEDS_NOVOS = 10      # 1005..1014 -- total combinado = 5+10 = 15
CONJUNTOS = ("A", "B", "X", "CMT")

ARMS = [
    ("hgucb_completo", HGUCBBandit),
    ("sem_cold_start", NoColdStartBandit),
]


def rodar(inst, cws, bandit_class, seed):
    rng = random.Random(seed)
    bandit_obj = bandit_class(n_operators=5, rng=rng, p_hat_mode="q_values", alpha=0.5)
    resultado = bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
        bandit=bandit_obj, max_inner_iter=40, progressive_x_linha=True,
        reward_positive_fn=reward_positive_normalized,
    )
    return resultado


def main():
    print("=== Diagnóstico 17: 10 seeds novos (1005-1014) para hgucb_completo vs. sem_cold_start, 74 instâncias ===\n")
    t_inicio = time.perf_counter()

    linhas_csv = []
    gap_por_instancia = {rotulo: {} for rotulo, _ in ARMS}

    for set_name in CONJUNTOS:
        arquivos = sorted((ROOT / "instances" / set_name).glob("*.vrp"))
        for vrp_path in arquivos:
            inst = load_instance(vrp_path)
            cws = repair_route_count(inst, clarke_wright_savings(inst))
            alvo_artigo = PAPER_HGUCB_POR_INSTANCIA.get(inst.name)
            alvo_str = f", artigo HG-UCB={alvo_artigo:.2f}%" if alvo_artigo is not None else ""

            print(f"--- {inst.name} (conjunto {set_name}{alvo_str}) ---")
            for rotulo, bandit_class in ARMS:
                gaps = []
                for s in range(N_SEEDS_NOVOS):
                    seed = SEED_BASE_NOVO + s
                    resultado = rodar(inst, cws, bandit_class, seed)
                    viavel, motivo = inst.is_feasible(resultado)
                    if not viavel:
                        print(f"    ERRO: {rotulo} {inst.name} seed={seed} inviável: {motivo}")
                        sys.exit(1)
                    custo = inst.solution_cost(resultado)
                    gap = 100 * (custo - inst.optimum) / inst.optimum
                    gaps.append(gap)
                    linhas_csv.append({
                        "braco": rotulo, "conjunto": set_name,
                        "instancia": inst.name, "seed": seed, "gap_pct": gap,
                    })
                media = sum(gaps) / len(gaps)
                gap_por_instancia[rotulo][inst.name] = media
                print(f"  [{rotulo:15s}] gap_medio_10_seeds_novos={media:6.2f}%")
            print()

    dt_total = time.perf_counter() - t_inicio
    print(f"Tempo total de execução: {dt_total:.1f}s ({dt_total/60:.1f}min)\n")

    out_csv = ROOT / "results" / "diag17_coldstart_maisseeds.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas_csv[0].keys()))
        w.writeheader()
        w.writerows(linhas_csv)
    print(f"CSV salvo em: {out_csv}")
    print("(a análise combinando com os 5 seeds do Diagnóstico 16 -- total 15 -- é feita à parte, lendo os dois CSVs)")


if __name__ == "__main__":
    main()
