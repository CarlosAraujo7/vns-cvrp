#!/usr/bin/env python3
"""
diag1_bandit_trace.py
======================
Diagnóstico #1 dos 3 sugeridos pelo Gemini (segunda opinião, Entrada 8 do
diário) para investigar o Achado Real #3: mesmo com `progressive_x_linha`,
nosso bandit ainda inverte o sinal do efeito esperado pelo artigo (piora
em vez de melhorar sobre o GVNS clássico sem bandit).

Loga o estado interno COMPLETO do HGUCBBandit a cada iteração interna (H
antes/depois, q_values antes/depois, p_hat, pool desbloqueado, contagens
n[], modo de decisão, recompensa) via `TracingHGUCBBandit` -- uma
SUBCLASSE que só adiciona logging (sempre chama super(), não reimplementa
nenhuma lógica) -- não altera bandit.py nem vns.py, zero risco às
validações já feitas e sincronizadas (fixo/progressivo, Entradas 7-8).

Instância escolhida para o traço detalhado: B-n41-k6 -- o pior caso
observado em toda a validação "progressivo" (Entrada 8: dif_media=+8,97pp
sobre o artigo) -- bom candidato a expor o mecanismo da falha, se houver.

Hipótese ESPECÍFICA sendo checada aqui (levantada por nós ao reler a Eq.11
à luz da sugestão do Gemini, não pelo próprio Gemini): a recompensa (Eq.12)
usa rs=|sol_g - sol_c| em UNIDADES DE CUSTO ABSOLUTAS (não normalizadas
para uma escala pequena/fixa), enquanto alpha=5.0 (hiperparâmetro "melhor"
do artigo, Tabela 8) é um learning rate. Se a escala típica de rs for
grande (dezenas/centenas -- plausível para custos de CVRP), um único
update() pode mover H o bastante para colapsar softmax(H) quase para
one-hot (max q -> ~1.0) já nas primeiras iterações -- e, uma vez lá, o
próprio termo (1-pi)/pi da Eq.11 fica perto de zero para todo mundo,
travando esse colapso quase permanentemente ("morte por gradiente
saturado", padrão conhecido em bandits de gradiente com learning rate mal
calibrado para a escala da recompensa). Este script torna essa hipótese
VERIFICÁVEL a partir de execução real, não apenas plausível.

Uso: python diag1_bandit_trace.py
"""
import csv
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from vns import bandit_vns  # noqa: E402
from bandit import HGUCBBandit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class TracingHGUCBBandit(HGUCBBandit):
    """Subclasse SÓ de diagnóstico -- loga estado interno completo a cada
    select()/update(). NÃO reimplementa nenhuma lógica (sempre chama
    super()), então o comportamento de busca é EXATAMENTE o mesmo de
    HGUCBBandit; só adiciona observação passiva."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = []
        self._pending = None

    def select(self):
        n_antes = list(self.n)
        never_tried = [j for j in range(self.k) if self.n[j] == 0]
        escolhido = super().select()  # incrementa self.s; NÃO muda H
        q = self.q_values()
        p_hat = self.p_hat(q)
        pool = self.unblocked_pool(q)
        if never_tried:
            modo = "cold_start"
        else:
            scores = {
                j: p_hat[j] + (self.C * math.sqrt(2.0 * math.log(self.s) / n_antes[j])
                                if n_antes[j] > 0 else float("inf"))
                for j in pool
            }
            argmax_j = max(scores, key=scores.get)
            modo = "ucb1_argmax" if escolhido == argmax_j else "epsilon_provavel"
        self._pending = {
            "s": self.s, "H_antes": list(self.H), "q_antes": q, "p_hat_antes": p_hat,
            "n_antes": n_antes, "pool": list(pool), "pool_size": len(pool),
            "escolhido": escolhido, "modo": modo,
        }
        return escolhido

    def update(self, chosen, reward):
        # Achado ao rodar (não previsto na 1a versão deste script): quando a
        # iteração ANTERIOR sinalizou drift, bandit_vns() (vns.py) NÃO chama
        # select() nesta iteração -- sorteia a vizinhança direto via
        # `rng.randrange()` e chama reset_drift() por conta própria (Algoritmo
        # 3, linhas 7-9). Nesses casos self._pending continua None aqui.
        # Registramos uma entrada mínima em vez de quebrar.
        if self._pending is None:
            self._pending = {
                "s": self.s, "H_antes": list(self.H), "q_antes": self.q_values(),
                "p_hat_antes": self.q_values(), "n_antes": list(self.n),
                "pool": list(range(self.k)), "pool_size": self.k,
                "escolhido": chosen, "modo": "drift_reset_random",
            }
        drift = super().update(chosen, reward)
        entry = self._pending
        entry.update({
            "reward": reward, "H_depois": list(self.H), "q_depois": self.q_values(),
            "n_depois": list(self.n), "drift": drift,
        })
        self.trace.append(entry)
        self._pending = None
        return drift


def fmt(lst):
    return "/".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in lst)


def main():
    nome, subdir, seed = "B-n41-k6", "B", 1000
    path = ROOT / "instances" / subdir / f"{nome}.vrp"
    inst = load_instance(path)
    cws = repair_route_count(inst, clarke_wright_savings(inst))

    rng = random.Random(seed)
    # alpha=5.0 fixado explicitamente (Entrada 26): era o default da classe
    # quando esta ferramenta de tracing foi escrita/usada; o default de
    # HGUCBBandit mudou para 0.5 depois (Entradas 6/12/25) -- preserva o
    # comportamento original (este script não é sobre alpha, é sobre
    # inspecionar o estado interno do bandit passo a passo).
    bandit_obj = TracingHGUCBBandit(n_operators=5, rng=rng, p_hat_mode="q_values", alpha=5.0)
    resultado = bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=20,
        bandit=bandit_obj, max_inner_iter=40, progressive_x_linha=True,
    )
    viavel, motivo = inst.is_feasible(resultado)
    custo = inst.solution_cost(resultado)
    gap = 100 * (custo - inst.optimum) / inst.optimum
    print(f"=== {nome} seed={seed} progressivo: gap={gap:.2f}% viavel={viavel} ({motivo}) ===")
    print(f"Total de iterações internas registradas: {len(bandit_obj.trace)}\n")

    out_csv = ROOT / "results" / "diag1_trace_B-n41-k6.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iter", "s", "modo", "pool_size", "escolhido", "reward", "drift",
                    "H_antes", "H_depois", "q_antes", "q_depois", "n_antes", "n_depois"])
        for i, e in enumerate(bandit_obj.trace):
            w.writerow([
                i, e["s"], e["modo"], e["pool_size"], e["escolhido"], f"{e['reward']:.4f}", e["drift"],
                fmt(e["H_antes"]), fmt(e["H_depois"]), fmt(e["q_antes"]), fmt(e["q_depois"]),
                "/".join(map(str, e["n_antes"])), "/".join(map(str, e["n_depois"])),
            ])
    print(f"Traço completo salvo em: {out_csv}\n")

    modos = Counter(e["modo"] for e in bandit_obj.trace)
    print(f"Distribuição de modo de decisão: {dict(modos)}")

    pool_sizes = Counter(e["pool_size"] for e in bandit_obj.trace)
    print(f"Distribuição de tamanho do pool desbloqueado: {dict(sorted(pool_sizes.items()))}")

    max_H_abs = [max(abs(h) for h in e["H_depois"]) for e in bandit_obj.trace]
    max_q = [max(e["q_depois"]) for e in bandit_obj.trace]
    n = len(bandit_obj.trace)
    marcos = [0, 4, 9, 24, 49, n - 1]
    print("\nmax|H| ao longo do tempo (iterações 1, 5, 10, 25, 50, última):")
    print("  " + "  ".join(f"it{m+1}={max_H_abs[m]:.2f}" for m in marcos if 0 <= m < n))
    print("max(q_values)=softmax(H) ao longo do tempo (mesmas iterações):")
    print("  " + "  ".join(f"it{m+1}={max_q[m]:.4f}" for m in marcos if 0 <= m < n))

    n_colapsado = sum(1 for q in max_q if q > 0.99)
    n_colapsado_95 = sum(1 for q in max_q if q > 0.95)
    print(f"\nIterações com softmax quase one-hot (max_q > 0.99): {n_colapsado}/{n} ({100*n_colapsado/n:.1f}%)")
    print(f"Iterações com softmax bem concentrado (max_q > 0.95): {n_colapsado_95}/{n} ({100*n_colapsado_95/n:.1f}%)")

    escolhas_finais = Counter(e["escolhido"] for e in bandit_obj.trace[-100:])
    escolhas_iniciais = Counter(e["escolhido"] for e in bandit_obj.trace[:100])
    print(f"\nOperador escolhido nas PRIMEIRAS {min(100,n)} iterações: {dict(escolhas_iniciais)}")
    print(f"Operador escolhido nas ÚLTIMAS {min(100,n)} iterações: {dict(escolhas_finais)}")

    rp = [e["reward"] for e in bandit_obj.trace if e["reward"] > 0]
    rn = [e["reward"] for e in bandit_obj.trace if e["reward"] <= 0]
    if rp:
        print(f"\nRecompensas POSITIVAS (Eq.12): n={len(rp)}  min={min(rp):.3f}  max={max(rp):.3f}  media={sum(rp)/len(rp):.3f}")
    if rn:
        print(f"Recompensas NEGATIVAS (Eq.13): n={len(rn)}  min={min(rn):.3f}  max={max(rn):.3f}  media={sum(rn)/len(rn):.3f}")


if __name__ == "__main__":
    main()
