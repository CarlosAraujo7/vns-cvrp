#!/usr/bin/env python3
"""
diag8_bandit_trace_v2.py
===========================
Diagnóstico do Gemini (Entrada 13 do diário) ainda pendente após o
resultado nulo do diagnóstico 7 (Entrada 15, achado real #8 -- ADWIN):
"tracing detalhado de q_i, N_i, R_t, avg_R e pi_i(t) iteração a iteração
numa instância pequena", escolhido pelo pesquisador via `AskUserQuestion`
em vez do teste de sensibilidade (vantagem só-sinal).

Diferença em relação a `diag1_bandit_trace.py` (o traço original, Entrada
9, achado real #4): aquele rodou a configuração ORIGINAL/quebrada (alpha=5,0
sem normalização) em B-n41-k6 (o pior caso) para diagnosticar o colapso do
softmax. Este roda a configuração ATUAL, pós-todas-as-correções (recompensa
normalizada -- Entrada 12 -- + alpha=5,0, o candidato recomendado da
Entrada 13) em A-n32-k5 (a instância mais simples/pequena usada em toda
esta investigação, k=5), para procurar qualquer sinal de comportamento
degenerado que ainda reste DEPOIS do colapso já ter sido resolvido -- não
reabre a pergunta já respondida (colapso), procura algo NOVO.

Reaproveita `TracingHGUCBBandit` de diag1 (subclasse só-logging, nunca
reimplementa lógica) via herança, adicionando 1 campo que faltava para
este diagnóstico específico: `r_bar` (a média de recompensas ANTERIORES,
"avg_R" na notação do Gemini) capturada IMEDIATAMENTE ANTES de
HGUCBBandit.update() mutar `_reward_sum`/`_reward_count` -- exatamente o
valor que a Eq. 11 usa como baseline nesta mesma iteração.

O que este script imprime, para inspeção manual linha a linha (Gemini,
Entrada 13): "verifique se q_i está se movendo na direção esperada
(aumentando para boas ações, diminuindo para ruins), se não está
colapsando para zero ou divergindo para valores extremos, e se pi_i(t)
está reagindo sensivelmente a q_i. Compare R_t com avg_R."

Uso: python diag8_bandit_trace_v2.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import random  # noqa: E402

from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from vns import bandit_vns  # noqa: E402
from bandit import reward_positive_normalized  # noqa: E402
from diag1_bandit_trace import TracingHGUCBBandit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_ITER_INSPECAO = 100  # tamanho da amostra impressa para inspeção manual (Gemini sugeriu 50-100)


class TracingHGUCBBanditV2(TracingHGUCBBandit):
    """Acrescenta `r_bar` (avg_R) ao traço de diag1 -- capturado ANTES de
    HGUCBBandit.update() mutar _reward_sum/_reward_count, i.e., exatamente
    o baseline que a Eq. 11 usa NESTA iteração."""

    def update(self, chosen, reward):
        r_bar = self._reward_sum / self._reward_count if self._reward_count else 0.0
        drift = super().update(chosen, reward)
        self.trace[-1]["r_bar"] = r_bar
        return drift


def fmt(lst):
    return "/".join(f"{v:.4f}" for v in lst)


def main():
    nome, subdir, seed, alpha = "A-n32-k5", "A", 1000, 5.0
    path = ROOT / "instances" / subdir / f"{nome}.vrp"
    inst = load_instance(path)
    cws = repair_route_count(inst, clarke_wright_savings(inst))

    rng = random.Random(seed)
    bandit_obj = TracingHGUCBBanditV2(n_operators=5, rng=rng, p_hat_mode="q_values", alpha=alpha)
    resultado = bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=20,
        bandit=bandit_obj, max_inner_iter=40, progressive_x_linha=True,
        reward_positive_fn=reward_positive_normalized,
    )
    viavel, motivo = inst.is_feasible(resultado)
    custo = inst.solution_cost(resultado)
    gap = 100 * (custo - inst.optimum) / inst.optimum
    trace = bandit_obj.trace
    n = len(trace)
    print(f"=== {nome} seed={seed} alpha={alpha} normalizado progressivo: "
          f"gap={gap:.2f}% viavel={viavel} ({motivo}) ===")
    print(f"Total de iterações internas registradas: {n}\n")

    out_csv = ROOT / "results" / f"diag8_trace_v2_{nome}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iter", "s", "modo", "pool_size", "escolhido", "reward", "r_bar", "drift",
                    "H_antes", "H_depois", "q_antes", "q_depois", "n_antes", "n_depois"])
        for i, e in enumerate(trace):
            w.writerow([
                i, e["s"], e["modo"], e["pool_size"], e["escolhido"],
                f"{e['reward']:.4f}", f"{e['r_bar']:.4f}", e["drift"],
                fmt(e["H_antes"]), fmt(e["H_depois"]), fmt(e["q_antes"]), fmt(e["q_depois"]),
                "/".join(map(str, e["n_antes"])), "/".join(map(str, e["n_depois"])),
            ])
    print(f"Traço completo salvo em: {out_csv}\n")

    # --- Inspeção linha a linha das primeiras N_ITER_INSPECAO iterações ---
    print(f"--- Primeiras {min(N_ITER_INSPECAO, n)} iterações (inspeção manual, pedido do Gemini/Entrada 13) ---")
    header = f"{'it':>3} {'modo':16s} {'esc':>3} {'R_t':>8} {'avg_R':>8} {'R_t-avg_R':>10} {'pi_antes':32s} {'pi_depois':32s}"
    print(header)
    for i, e in enumerate(trace[:N_ITER_INSPECAO]):
        pi_antes = "/".join(f"{v:.2f}" for v in e["q_antes"])
        pi_depois = "/".join(f"{v:.2f}" for v in e["q_depois"])
        print(f"{i:3d} {e['modo']:16s} {e['escolhido']:3d} {e['reward']:8.3f} {e['r_bar']:8.3f} "
              f"{e['reward']-e['r_bar']:10.3f} {pi_antes:32s} {pi_depois:32s}")

    # --- Resumo estatístico do run completo (sanidade, mesmo estilo do diag1) ---
    print(f"\n--- Resumo do run completo ({n} iterações) ---")
    max_q = [max(e["q_depois"]) for e in trace]
    min_q = [min(e["q_depois"]) for e in trace]
    n_colapsado = sum(1 for q in max_q if q > 0.99)
    print(f"Iterações com softmax quase one-hot (max_q>0.99): {n_colapsado}/{n} ({100*n_colapsado/n:.1f}%)")
    print(f"max(q) ao longo do tempo -- it1={max_q[0]:.3f}  it{n//4}={max_q[n//4]:.3f}  "
          f"it{n//2}={max_q[n//2]:.3f}  it{3*n//4}={max_q[3*n//4]:.3f}  it{n}={max_q[-1]:.3f}")
    print(f"min(q) ao longo do tempo -- it1={min_q[0]:.3f}  it{n//4}={min_q[n//4]:.3f}  "
          f"it{n//2}={min_q[n//2]:.3f}  it{3*n//4}={min_q[3*n//4]:.3f}  it{n}={min_q[-1]:.3f}")

    n_drift = sum(1 for e in trace if e["drift"])
    print(f"Drifts detectados: {n_drift}/{n} ({100*n_drift/n:.2f}%)")

    rp = [e["reward"] for e in trace if e["reward"] > 0]
    rn = [e["reward"] for e in trace if e["reward"] <= 0]
    print(f"Recompensas POSITIVAS (Eq.12 normalizada): n={len(rp)}  "
          f"min={min(rp):.4f}  max={max(rp):.4f}  media={sum(rp)/len(rp):.4f}")
    print(f"Recompensas NEGATIVAS (Eq.13): n={len(rn)}  "
          f"min={min(rn):.4f}  max={max(rn):.4f}  media={sum(rn)/len(rn):.4f}")

    r_bars = [e["r_bar"] for e in trace]
    print(f"avg_R (r_bar) ao longo do tempo -- it1={r_bars[0]:.4f}  it{n//4}={r_bars[n//4]:.4f}  "
          f"it{n//2}={r_bars[n//2]:.4f}  it{3*n//4}={r_bars[3*n//4]:.4f}  it{n}={r_bars[-1]:.4f}")

    from collections import Counter
    escolhas_iniciais = Counter(e["escolhido"] for e in trace[:100])
    escolhas_finais = Counter(e["escolhido"] for e in trace[-100:])
    print(f"\nOperador escolhido nas PRIMEIRAS {min(100,n)} iterações: {dict(sorted(escolhas_iniciais.items()))}")
    print(f"Operador escolhido nas ÚLTIMAS {min(100,n)} iterações: {dict(sorted(escolhas_finais.items()))}")


if __name__ == "__main__":
    main()
