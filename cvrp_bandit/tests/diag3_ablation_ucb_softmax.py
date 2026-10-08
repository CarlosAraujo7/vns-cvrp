#!/usr/bin/env python3
"""
diag3_ablation_ucb_softmax.py
================================
Diagnóstico #3 dos 3 sugeridos pelo Gemini (Entrada 8, Achado Real #3):
isola os dois mecanismos que compõem o "híbrido" do HG-UCB -- UCB1 (Eq.5)
e o gradiente de preferência softmax (Eq.11) -- rodando cada um SOZINHO
(ver tests/ablation_bandits.py), para descobrir se a degradação observada
vem de um mecanismo específico, da COMBINAÇÃO dos dois, ou de nenhum dos
dois (o que apontaria a causa para outro lugar -- ex.: as próprias
fórmulas de recompensa Eq.12-13, compartilhadas por todas as variantes).

4 variantes comparadas, todas com progressive_x_linha=True (nossa melhor
configuração até agora -- Entradas 7-8) onde aplicável, protocolo idêntico
à Tarefa #178 (steps_max=20, 5 seeds, SEED_BASE=1000):

  gvns_classico  -- sem bandit nenhum (Entrada 3), âncora de referência.
  hgucb_completo -- HGUCBBandit padrão (softmax(H) + UCB1 híbrido).
  ucb1_only      -- só UCB1 clássico (média empírica bruta de recompensa +
                     termo de exploração), SEM softmax/H. Pool b_t
                     desabilitado (sem análogo natural -- ver
                     ablation_bandits.py).
  softmax_only   -- só o Gradient Bandit clássico (amostra proporcional a
                     softmax(H), Eq.11), SEM termo de exploração UCB1.
                     Pool b_t mantido (mesmo sentido que no HG-UCB
                     completo).

Amostra: as mesmas 5 instâncias do conjunto A de
tests/compare_progressive_variant.py + B-n41-k6 (pior caso observado na
validação completa "progressivo", Entrada 8 -- bom candidato a expor
diferenças entre as variantes).

Uso: python diag3_ablation_ucb_softmax.py
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from vns import bandit_vns, gvns  # noqa: E402
from bandit import HGUCBBandit  # noqa: E402
from ablation_bandits import UCB1OnlyBandit, SoftmaxOnlyBandit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STEPS_MAX = 20
N_SEEDS = 5
SEED_BASE = 1000

CASOS = [
    ("A-n32-k5", "A", 0.1892),
    ("A-n34-k5", "A", 0.1404),
    ("A-n38-k5", "A", 0.6835),
    ("A-n55-k9", "A", 1.2720),
    ("A-n64-k9", "A", 4.0859),
    ("B-n41-k6", "B", 0.9000),
]
VARIANTES = ["gvns_classico", "hgucb_completo", "ucb1_only", "softmax_only"]


def rodar(inst, cws, variante, seed):
    rng = random.Random(seed)
    if variante == "gvns_classico":
        return gvns(inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX)
    # alpha=5.0 fixado explicitamente nas duas linhas abaixo que o usam
    # (Entrada 26): era o default de HGUCBBandit E de SoftmaxOnlyBandit
    # quando este diagnóstico foi rodado/registrado; o default de
    # HGUCBBandit mudou para 0.5 depois (Entradas 6/12/25) -- preserva o
    # comportamento original deste script (a ablação é sobre UCB1 vs.
    # softmax, não sobre alpha).
    if variante == "hgucb_completo":
        bandit_obj = HGUCBBandit(n_operators=5, rng=rng, p_hat_mode="q_values", alpha=5.0)
    elif variante == "ucb1_only":
        bandit_obj = UCB1OnlyBandit(n_operators=5, rng=rng)
    elif variante == "softmax_only":
        bandit_obj = SoftmaxOnlyBandit(n_operators=5, rng=rng, alpha=5.0)
    else:
        raise ValueError(variante)
    return bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
        bandit=bandit_obj, max_inner_iter=40, progressive_x_linha=True,
    )


def main():
    print("=== Diagnóstico 3: ablação UCB1-only vs. softmax-only vs. HG-UCB completo vs. GVNS clássico ===\n")
    resumo = {v: [] for v in VARIANTES}

    for name, set_name, alvo_artigo in CASOS:
        path = ROOT / "instances" / set_name / f"{name}.vrp"
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        print(f"--- {name} (artigo HG-UCB={alvo_artigo:.2f}%) ---")
        for variante in VARIANTES:
            gaps = []
            t0 = time.perf_counter()
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                resultado = rodar(inst, cws, variante, seed)
                viavel, motivo = inst.is_feasible(resultado)
                if not viavel:
                    print(f"    ERRO: variante={variante} seed={seed} inviável: {motivo}")
                    sys.exit(1)
                custo = inst.solution_cost(resultado)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                gaps.append(gap)
            dt = time.perf_counter() - t0
            media = sum(gaps) / len(gaps)
            melhor = min(gaps)
            resumo[variante].append(media)
            print(f"  [{variante:15s}] gap_melhor={melhor:6.2f}%  gap_medio={media:6.2f}%  tempo_total={dt:.1f}s")
        print()

    print("=== Médias gerais (sobre as 6 instâncias testadas, média-de-5-seeds) ===")
    for variante in VARIANTES:
        print(f"  {variante:15s}: {sum(resumo[variante])/len(resumo[variante]):.2f}%")


if __name__ == "__main__":
    main()
