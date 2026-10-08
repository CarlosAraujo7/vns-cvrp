#!/usr/bin/env python3
"""
compare_phat_variants.py
==========================
Experimento comparativo, pedido explicitamente pelo pesquisador em
01/09/2026 (via AskUserQuestion) ANTES de decidir qual leitura de "p̂" usar
na validação completa da Tarefa #178 -- ver bandit.py, docstring do módulo,
decisão (2).

Compara as duas leituras de p̂ (termo de exploitation da fórmula UCB1,
Eq. 5) isolando-as via `HGUCBBandit(p_hat_mode=...)`:

  "q_values"       -- nossa leitura original: p̂ = softmax(H), a mesma
                       grandeza usada no filtro b_t.
  "sliding_window" -- leitura defendida pelo Gemini (segunda opinião
                       independente, mesma data): p̂_j = média das últimas
                       W=5 recompensas BRUTAS recebidas quando j foi
                       escolhido -- estrutura separada de H.

Tudo o mais (Eq. 11 para H, filtro b_t sobre softmax(H), cold-start,
ε-greedy, ADWIN, Eq. 12-13 de recompensa, bandit_vns()) é IDÊNTICO nos dois
modos -- só o termo de exploitation do UCB1 muda. Isso isola exatamente a
peça em disputa; qualquer diferença de resultado é atribuível a essa única
mudança, não a uma combinação de mudanças.

Não é a validação completa da Tarefa #178 (essa usa steps_max=20, 5 seeds,
TODAS as instâncias de cada conjunto, contra as Tabelas 10-12/13) -- aqui
usamos as mesmas 3 instâncias do smoke test (tests/smoke_test_bandit_vns.py)
com 3 seeds cada, o suficiente para um sinal direcional sem o custo de
rodar o conjunto inteiro duas vezes.

Uso: python compare_phat_variants.py
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from vns import bandit_vns  # noqa: E402
from bandit import HGUCBBandit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STEPS_MAX = 20
N_SEEDS = 3
SEED_BASE = 5000
MODOS = ["q_values", "sliding_window"]


def main():
    print("=== Comparação p̂: q_values (leitura A) vs. sliding_window (leitura B, Gemini) ===\n")

    casos = [
        ("A-n32-k5", "A", 20),
        ("A-n80-k10", "raw/SetA/A", 20),
        ("X-n195-k51", "X", 10),
    ]

    resultados = {modo: {} for modo in MODOS}

    for name, subdir, steps in casos:
        path = ROOT / "instances" / subdir / f"{name}.vrp"
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        print(f"--- {name} (n={inst.n_customers}, steps={steps}) ---")
        for modo in MODOS:
            gaps = []
            tempos = []
            escolhas_agregadas = [0] * 5  # contagem de seleção por operador, somada entre seeds
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                rng = random.Random(seed)
                # alpha=5.0 fixado explicitamente (Entrada 26): era o default
                # da classe quando este diagnóstico foi rodado/registrado; o
                # default mudou para 0.5 depois (Entradas 6/12/25) -- preserva
                # o comportamento original (a comparação é sobre p_hat_mode).
                bandit_obj = HGUCBBandit(n_operators=5, rng=rng, p_hat_mode=modo, alpha=5.0)

                def on_inner(step, inner_iter, escolhido, melhorou, reward, drift, _c=escolhas_agregadas):
                    _c[escolhido] += 1

                t0 = time.perf_counter()
                resultado = bandit_vns(inst, cws, rng, steps_max=steps, bandit=bandit_obj, on_inner=on_inner)
                dt = time.perf_counter() - t0

                viavel, motivo = inst.is_feasible(resultado)
                if not viavel:
                    print(f"    ERRO: modo={modo} seed={seed} produziu solução inviável: {motivo}")
                    sys.exit(1)
                custo = inst.solution_cost(resultado)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                gaps.append(gap)
                tempos.append(dt)

            melhor = min(gaps)
            media = sum(gaps) / len(gaps)
            resultados[modo][name] = {"melhor": melhor, "media": media}
            dist = "/".join(str(c) for c in escolhas_agregadas)
            print(
                f"  [{modo:15s}] gap_melhor_de_{N_SEEDS}={melhor:6.2f}%  gap_medio={media:6.2f}%  "
                f"tempo_medio={sum(tempos)/len(tempos):5.2f}s  "
                f"distribuicao_operadores(2opt/mv1/swap/rel10/rel20)={dist}"
            )
        print()

    print("=== Resumo (gap médio-de-{} por instância) ===".format(N_SEEDS))
    for name, _, _ in casos:
        qv = resultados["q_values"][name]["media"]
        sw = resultados["sliding_window"][name]["media"]
        melhor_modo = "q_values" if qv < sw else ("sliding_window" if sw < qv else "empate")
        print(f"  {name:14s} q_values={qv:6.2f}%   sliding_window={sw:6.2f}%   melhor={melhor_modo}")

    media_geral_qv = sum(resultados["q_values"][n]["media"] for n, _, _ in casos) / len(casos)
    media_geral_sw = sum(resultados["sliding_window"][n]["media"] for n, _, _ in casos) / len(casos)
    print(f"\nMédia geral (3 instâncias): q_values={media_geral_qv:.2f}%   sliding_window={media_geral_sw:.2f}%")
    print(f"Diferença (sliding_window - q_values): {media_geral_sw - media_geral_qv:+.2f} p.p.")


if __name__ == "__main__":
    main()
