#!/usr/bin/env python3
"""
diag16_ablacao_74completas.py
================================
Fase 2 da Direção 2 (Tarefa #179, diário Entradas 28-30): repete as 8
configurações do Diagnóstico 15 (Entrada 29) nas 74 instâncias COMPLETAS
do benchmark (A=26/B=23/X=21/CMT=4), em vez da amostra de 16 -- decisão
do pesquisador (06/09/2026, via `AskUserQuestion`, respondendo à Entrada
30): "ampliar todas as 5 comparações", não só a do drift-reset (que já
tinha saído estatisticamente robusta em n=16), exatamente o mesmo padrão
de "ir para a validação completa" já usado para o par alpha/max_inner_iter
(Entradas 21→24).

Protocolo IDÊNTICO ao Diagnóstico 15 e ao fechamento da Tarefa #178:
steps_max=20, 5 seeds, SEED_BASE=1000, max_inner_iter=40,
reward_positive_normalized, progressive_x_linha=True -- só o TAMANHO da
amostra de instâncias muda (16 -> 74). As 74 instâncias são obtidas por
glob de instances/{A,B,X,CMT}/*.vrp (mesmo padrão de
tests/run_bandit_vns_validation.py) -- não por uma lista digitada à mão,
para não arriscar erro de transcrição numa lista deste tamanho. Valores
de referência do artigo (coluna "HG-UCB mean error") importados de
`tests/run_bandit_vns_validation.py` (`PAPER_HGUCB_POR_INSTANCIA`) --
mesma fonte já validada usada em todos os diagnósticos anteriores, não
retranscritos à mão.

Mesmas 8 configurações do Diagnóstico 15 -- ver docstring de lá para a
justificativa completa de cada uma (filosofia de ablação B do Gemini +
refinamento próprio, piso `max(n[j],1)`, `drift_reset_enabled`, a nota
mecânica sobre `sem_nenhum`):

  PASSADA PRINCIPAL (alpha=0,5): gvns_classico, hgucb_completo,
    sem_cold_start, sem_epsilon_greedy, sem_drift_reset, sem_nenhum.
  PASSADA SECUNDÁRIA (alpha=5,0, só drift-reset): hgucb_completo_a5,
    sem_drift_reset_a5.

NOVIDADE em relação ao Diagnóstico 15: grava um CSV por-execução
(results/diag16_ablacao_74completas.csv, uma linha por
arm×instância×seed) além do log de stdout -- permite qualquer teste
estatístico futuro (Wilcoxon/t pareado, mesma disciplina da Entrada 25/30)
ler os dados diretamente, sem precisar re-parsear texto nem rodar nada de
novo.

Custo estimado (sonda real de 1 braço/1 seed nas 74 instâncias antes de
lançar isto, ver diário): ~309s para 1 configuração completa -> ~205min
(~3,4h) para as 8 configurações × 5 seeds. Rodado em background com
monitoramento, mesmo padrão da validação completa da Tarefa #178
(Entradas 23-24).

Uso: python diag16_ablacao_74completas.py
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
from vns import bandit_vns, gvns  # noqa: E402
from bandit import HGUCBBandit, reward_positive_normalized  # noqa: E402
from ablation_randomness_bandits import (  # noqa: E402
    NoColdStartBandit, NoEpsilonGreedyBandit, NoColdStartNoEpsilonBandit,
)
from run_bandit_vns_validation import PAPER_HGUCB_POR_INSTANCIA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STEPS_MAX = 20
N_SEEDS = 5
SEED_BASE = 1000
CONJUNTOS = ("A", "B", "X", "CMT")

# (rótulo, classe_do_bandit [None = gvns clássico], alpha, drift_reset_enabled, grupo)
ARMS = [
    ("gvns_classico", None, None, True, "principal"),
    ("hgucb_completo", HGUCBBandit, 0.5, True, "principal"),
    ("sem_cold_start", NoColdStartBandit, 0.5, True, "principal"),
    ("sem_epsilon_greedy", NoEpsilonGreedyBandit, 0.5, True, "principal"),
    ("sem_drift_reset", HGUCBBandit, 0.5, False, "principal"),
    ("sem_nenhum", NoColdStartNoEpsilonBandit, 0.5, False, "principal"),
    ("hgucb_completo_a5", HGUCBBandit, 5.0, True, "stress"),
    ("sem_drift_reset_a5", HGUCBBandit, 5.0, False, "stress"),
]


def rodar(inst, cws, bandit_class, alpha, drift_reset_enabled, seed):
    rng = random.Random(seed)
    if bandit_class is None:
        resultado = gvns(inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX)
        return resultado, None

    bandit_obj = bandit_class(n_operators=5, rng=rng, p_hat_mode="q_values", alpha=alpha)
    contador = {"colapsadas": 0, "drift_sinalizado": 0, "total": 0}

    def on_inner(step, inner_iter, escolhido, melhorou, reward, drift, _b=bandit_obj, _c=contador):
        _c["total"] += 1
        if max(_b.q_values()) > 0.99:
            _c["colapsadas"] += 1
        if drift:
            _c["drift_sinalizado"] += 1

    resultado = bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
        bandit=bandit_obj, max_inner_iter=40, progressive_x_linha=True,
        reward_positive_fn=reward_positive_normalized, on_inner=on_inner,
        drift_reset_enabled=drift_reset_enabled,
    )
    contador["nunca_testado"] = sum(1 for n in bandit_obj.n if n == 0)
    contador["resets_reais"] = contador["drift_sinalizado"] if drift_reset_enabled else 0
    return resultado, contador


def main():
    print("=== Diagnóstico 16: ablação dos 3 mecanismos de aleatoriedade — 74 instâncias completas (Fase 2) ===\n")
    t_inicio = time.perf_counter()

    linhas_csv = []
    resumo_gap = {rotulo: {} for rotulo, *_ in ARMS}
    gap_por_instancia = {rotulo: {} for rotulo, *_ in ARMS}
    resumo_extra = {
        rotulo: {"colapsadas": 0, "drift_sinalizado": 0, "drift_reagido": 0,
                  "total": 0, "nunca_testado_soma": 0, "n_execucoes": 0}
        for rotulo, *_ in ARMS
    }

    for set_name in CONJUNTOS:
        arquivos = sorted((ROOT / "instances" / set_name).glob("*.vrp"))
        for vrp_path in arquivos:
            inst = load_instance(vrp_path)
            cws = repair_route_count(inst, clarke_wright_savings(inst))
            alvo_artigo = PAPER_HGUCB_POR_INSTANCIA.get(inst.name)
            alvo_str = f", artigo HG-UCB={alvo_artigo:.2f}%" if alvo_artigo is not None else ""

            print(f"--- {inst.name} (conjunto {set_name}{alvo_str}) ---")
            for rotulo, bandit_class, alpha, drift_reset_enabled, grupo in ARMS:
                gaps = []
                for s in range(N_SEEDS):
                    seed = SEED_BASE + s
                    resultado, contador = rodar(inst, cws, bandit_class, alpha, drift_reset_enabled, seed)
                    viavel, motivo = inst.is_feasible(resultado)
                    if not viavel:
                        print(f"    ERRO: {rotulo} {inst.name} seed={seed} inviável: {motivo}")
                        sys.exit(1)
                    custo = inst.solution_cost(resultado)
                    gap = 100 * (custo - inst.optimum) / inst.optimum
                    gaps.append(gap)
                    linhas_csv.append({
                        "braco": rotulo, "grupo": grupo, "conjunto": set_name,
                        "instancia": inst.name, "seed": seed, "gap_pct": gap,
                        "colapsadas": contador["colapsadas"] if contador else "",
                        "drift_sinalizado": contador["drift_sinalizado"] if contador else "",
                        "drift_reagido": contador["resets_reais"] if contador else "",
                        "total_iter": contador["total"] if contador else "",
                        "nunca_testado": contador["nunca_testado"] if contador else "",
                    })
                    if contador is not None:
                        e = resumo_extra[rotulo]
                        e["colapsadas"] += contador["colapsadas"]
                        e["drift_sinalizado"] += contador["drift_sinalizado"]
                        e["drift_reagido"] += contador["resets_reais"]
                        e["total"] += contador["total"]
                        e["nunca_testado_soma"] += contador["nunca_testado"]
                        e["n_execucoes"] += 1

                media = sum(gaps) / len(gaps)
                resumo_gap[rotulo].setdefault(set_name, []).append(media)
                gap_por_instancia[rotulo][inst.name] = media
                print(f"  [{rotulo:19s}] gap_medio={media:6.2f}%")
            print()

    dt_total = time.perf_counter() - t_inicio
    print(f"Tempo total de execução: {dt_total:.1f}s ({dt_total/60:.1f}min)\n")

    out_csv = ROOT / "results" / "diag16_ablacao_74completas.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas_csv[0].keys()))
        w.writeheader()
        w.writerows(linhas_csv)
    print(f"CSV salvo em: {out_csv}\n")

    print("=== Resumo GERAL — passada principal (alpha=0,5, média sobre as 74 instâncias) ===")
    for rotulo, _, _, _, grupo in ARMS:
        if grupo != "principal":
            continue
        todos = [g for lst in resumo_gap[rotulo].values() for g in lst]
        media_geral = sum(todos) / len(todos)
        print(f"  {rotulo:19s}  gap_medio={media_geral:6.2f}%")

    print("\n=== Resumo POR CONJUNTO (A/B/CMT/X) — passada principal ===")
    for rotulo, _, _, _, grupo in ARMS:
        if grupo != "principal":
            continue
        linha = f"  {rotulo:19s}"
        for set_name in ("A", "B", "CMT", "X"):
            lst = resumo_gap[rotulo].get(set_name, [])
            if lst:
                linha += f"  {set_name}={sum(lst)/len(lst):5.2f}%"
        print(linha)

    print("\n=== Diagnóstico dos mecanismos — passada principal (alpha=0,5) ===")
    for rotulo, _, _, _, grupo in ARMS:
        if grupo != "principal":
            continue
        e = resumo_extra[rotulo]
        if e["n_execucoes"] == 0:
            print(f"  {rotulo:19s}  (sem bandit -- sem diagnóstico de mecanismo)")
            continue
        pct_colapso = 100 * e["colapsadas"] / e["total"] if e["total"] else 0.0
        pct_drift_sinal = 100 * e["drift_sinalizado"] / e["total"] if e["total"] else 0.0
        pct_drift_reagido = 100 * e["drift_reagido"] / e["total"] if e["total"] else 0.0
        media_nunca_testado = e["nunca_testado_soma"] / e["n_execucoes"]
        print(
            f"  {rotulo:19s}  colapso={pct_colapso:5.2f}%  drift_sinalizado={pct_drift_sinal:6.3f}%  "
            f"drift_reagido(resets)={pct_drift_reagido:6.3f}%  media_operadores_nunca_testados={media_nunca_testado:.2f}  "
            f"(n_iter={e['total']}, n_execucoes={e['n_execucoes']})"
        )

    print("\n=== Passada secundária — stress-test drift-reset (alpha=5,0, 74 instâncias) ===")
    for rotulo, _, _, _, grupo in ARMS:
        if grupo != "stress":
            continue
        todos = [g for lst in resumo_gap[rotulo].values() for g in lst]
        media_geral = sum(todos) / len(todos)
        e = resumo_extra[rotulo]
        pct_drift_sinal = 100 * e["drift_sinalizado"] / e["total"] if e["total"] else 0.0
        pct_drift_reagido = 100 * e["drift_reagido"] / e["total"] if e["total"] else 0.0
        print(
            f"  {rotulo:19s}  gap_medio={media_geral:6.2f}%  drift_sinalizado={pct_drift_sinal:6.3f}%  "
            f"drift_reagido(resets)={pct_drift_reagido:6.3f}%  (n_iter={e['total']})"
        )


if __name__ == "__main__":
    main()
