#!/usr/bin/env python3
"""
diag18_alpha_transicao.py
===========================
Diagnóstico #18 -- follow-up da Direção 2 (Tarefa #179), decidido pelo
pesquisador via `AskUserQuestion` respondendo à Entrada 38 (07/09/2026):
com o cold-start/ε-greedy/drift-reset todos nulos em alpha=0,5 (produção)
e o drift-reset robusto em alpha=5,0 (stress-test, fora de produção),
mapear a TRANSIÇÃO entre os dois -- em vez de encerrar a Direção 2 aqui
(opção recomendada) ou investigar interação entre pares de mecanismos.

Reaproveita os DOIS PONTOS JÁ MEDIDOS como as extremidades da curva --
não precisam ser re-rodados:
  alpha=0,5 -> hgucb_completo vs. sem_drift_reset: diff=+0,008pp,
               NÃO significativo (Diagnóstico 16, Entrada 31).
  alpha=5,0 -> hgucb_completo_a5 vs. sem_drift_reset_a5: diff=+0,431pp,
               p<0,0001 nos dois testes (Diagnóstico 16, Entrada 31).

Roda 5 valores de alpha NOVOS, intermediários (1,0 / 1,5 / 2,0 / 3,0 /
4,0), só os 2 braços relevantes para o drift-reset (hgucb_completo,
sem_drift_reset -- não repete cold-start/ε-greedy, já encerrados), na
MESMA amostra de 16 instâncias e mesmo protocolo do Diagnóstico 15
(steps_max=20, mii=40, 5 seeds, SEED_BASE=1000) -- fase exploratória,
então só médias brutas + taxa de disparo do ADWIN por alpha (sem teste de
significância formal ainda -- só faz sentido depois de decidir, com a
curva em mãos, se/onde vale a pena aprofundar).

Uso: python diag18_alpha_transicao.py
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

ROOT = Path(__file__).resolve().parent.parent
STEPS_MAX = 20
N_SEEDS = 5
SEED_BASE = 1000

# Mesma amostra de 16 instâncias dos Diagnósticos 12/15 (Entradas 21/29).
CASOS = [
    ("A-n32-k5", "A", 0.1892),
    ("A-n34-k5", "A", 0.1404),
    ("A-n38-k5", "A", 0.6835),
    ("A-n55-k9", "A", 1.2720),
    ("A-n64-k9", "A", 4.0859),
    ("B-n41-k6", "B", 0.9000),
    ("B-n63-k10", "B", 4.0804),
    ("X-n101-k25", "X", 3.5419),
    ("A-n69-k9", "A", 3.4536),
    ("B-n31-k5", "B", 0.2592),
    ("B-n50-k8", "B", 1.1188),
    ("B-n78-k10", "B", 6.0025),
    ("CMT1", "CMT", 0.0),
    ("X-n106-k14", "X", 2.5488),
    ("X-n134-k13", "X", 7.3767),
    ("X-n157-k13", "X", 3.6606),
]

ALPHAS_NOVOS = [1.0, 1.5, 2.0, 3.0, 4.0]

# (rotulo, alpha, drift_reset_enabled)
ARMS = []
for a in ALPHAS_NOVOS:
    ARMS.append((f"completo_a{a}", a, True))
    ARMS.append((f"sem_drift_a{a}", a, False))


def caminho_instancia(name, set_name):
    if set_name == "CMT":
        return ROOT / "instances" / "CMT" / f"{name}.vrp"
    return ROOT / "instances" / set_name / f"{name}.vrp"


def rodar(inst, cws, alpha, drift_reset_enabled, seed):
    rng = random.Random(seed)
    bandit_obj = HGUCBBandit(n_operators=5, rng=rng, p_hat_mode="q_values", alpha=alpha)
    contador = {"drift_sinalizado": 0, "total": 0}

    def on_inner(step, inner_iter, escolhido, melhorou, reward, drift, _c=contador):
        _c["total"] += 1
        if drift:
            _c["drift_sinalizado"] += 1  # bruto do ADWIN, vale mesmo com drift_reset_enabled=False

    resultado = bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
        bandit=bandit_obj, max_inner_iter=40, progressive_x_linha=True,
        reward_positive_fn=reward_positive_normalized, on_inner=on_inner,
        drift_reset_enabled=drift_reset_enabled,
    )
    contador["resets_reais"] = contador["drift_sinalizado"] if drift_reset_enabled else 0
    return resultado, contador


def main():
    print("=== Diagnóstico 18: mapeando a transição alpha=0,5 -> 5,0 do efeito do drift-reset ===\n")
    t_inicio = time.perf_counter()

    linhas_csv = []
    resumo_extra = {rotulo: {"drift_sinalizado": 0, "total": 0} for rotulo, *_ in ARMS}
    gap_por_instancia = {rotulo: {} for rotulo, *_ in ARMS}

    for name, set_name, alvo_artigo in CASOS:
        path = caminho_instancia(name, set_name)
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        print(f"--- {name} (conjunto {set_name}) ---")
        for rotulo, alpha, drift_reset_enabled in ARMS:
            gaps = []
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                resultado, contador = rodar(inst, cws, alpha, drift_reset_enabled, seed)
                viavel, motivo = inst.is_feasible(resultado)
                if not viavel:
                    print(f"    ERRO: {rotulo} seed={seed} inviável: {motivo}")
                    sys.exit(1)
                custo = inst.solution_cost(resultado)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                gaps.append(gap)
                linhas_csv.append({
                    "braco": rotulo, "alpha": alpha, "drift_reset_enabled": drift_reset_enabled,
                    "conjunto": set_name, "instancia": name, "seed": seed, "gap_pct": gap,
                })
                e = resumo_extra[rotulo]
                e["drift_sinalizado"] += contador["drift_sinalizado"]
                e["total"] += contador["total"]
            media = sum(gaps) / len(gaps)
            gap_por_instancia[rotulo][name] = media
        print(f"  (16 instancias x {len(ARMS)} configs x {N_SEEDS} seeds processadas)")

    dt_total = time.perf_counter() - t_inicio
    print(f"\nTempo total de execução: {dt_total:.1f}s ({dt_total/60:.1f}min)\n")

    out_csv = ROOT / "results" / "diag18_alpha_transicao.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas_csv[0].keys()))
        w.writeheader()
        w.writerows(linhas_csv)
    print(f"CSV salvo em: {out_csv}\n")

    print("=== Curva alpha -> efeito do drift-reset (diff = completo - sem_drift, média bruta sobre 16 instâncias) ===")
    print("  alpha=0.5 (Diagnostico 16, ja medido): diff=+0.008pp, NAO significativo, drift_sinalizado~0.05%")
    for alpha in ALPHAS_NOVOS:
        rot_completo = f"completo_a{alpha}"
        rot_sem = f"sem_drift_a{alpha}"
        medias_completo = list(gap_por_instancia[rot_completo].values())
        medias_sem = list(gap_por_instancia[rot_sem].values())
        media_completo = sum(medias_completo) / len(medias_completo)
        media_sem = sum(medias_sem) / len(medias_sem)
        diff = media_completo - media_sem
        e_completo = resumo_extra[rot_completo]
        pct_drift = 100 * e_completo["drift_sinalizado"] / e_completo["total"] if e_completo["total"] else 0.0
        vitorias_sem = sum(1 for n in gap_por_instancia[rot_completo] if gap_por_instancia[rot_sem][n] < gap_por_instancia[rot_completo][n])
        print(f"  alpha={alpha:4.1f}  gap_completo={media_completo:6.3f}%  gap_sem_drift={media_sem:6.3f}%  "
              f"diff={diff:+.4f}pp  drift_sinalizado={pct_drift:6.3f}%  vitorias_sem_drift={vitorias_sem}/16")
    print("  alpha=5.0 (Diagnostico 16, ja medido): diff=+0.431pp, p<0.0001, drift_sinalizado~0.69%")


if __name__ == "__main__":
    main()
