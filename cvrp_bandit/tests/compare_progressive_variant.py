#!/usr/bin/env python3
"""
compare_progressive_variant.py
=================================
Experimento comparativo motivado por um achado REAL da Tarefa #178 (não
hipótese prévia): a validação completa contra as Tabelas 10-12 mostrou
erro muito acima do artigo (Set A: 5,76% vs. 1,61%, consistente na maioria
das 26 instâncias). Ver vns.py, docstring de bandit_vns(), "ACHADO REAL #2".

Compara:
  progressive_x_linha=False (default, leitura literal original -- x' fixo
                              durante todo o loop interno)
  progressive_x_linha=True  (leitura alternativa -- x' avança a cada
                              melhora aceita dentro do mesmo passo externo)

Mesmas instâncias/seeds do smoke test (tests/smoke_test_bandit_vns.py) e
da comparação de p̂ (tests/compare_phat_variants.py), para reaproveitar
os mesmos pontos de referência já usados nesta linha de pesquisa.

Uso: python compare_progressive_variant.py
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from vns import bandit_vns, gvns  # noqa: E402
from bandit import HGUCBBandit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STEPS_MAX = 20
N_SEEDS = 5
SEED_BASE = 1000  # mesmo SEED_BASE da validacao Tarefa #178 -- comparavel

CASOS = [
    ("A-n32-k5", "A", 0.1892),
    ("A-n34-k5", "A", 0.1404),
    ("A-n38-k5", "A", 0.6835),
    ("A-n55-k9", "A", 1.2720),
    ("A-n64-k9", "A", 4.0859),
    ("X-n195-k51", "X", 4.9670),
]


def rodar(inst, cws, modo, seed):
    rng = random.Random(seed)
    if modo == "gvns_classico":
        return gvns(inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX)
    # alpha=5.0 fixado explicitamente (Entrada 26): era o default da classe
    # quando este diagnóstico foi rodado/registrado (Entrada 7); o default
    # mudou para 0.5 depois (Entradas 6/12/25) -- preserva o comportamento
    # original deste script (a comparação é sobre x' progressivo, não alpha).
    bandit_obj = HGUCBBandit(n_operators=5, rng=rng, p_hat_mode="q_values", alpha=5.0)
    progressive = (modo == "progressivo")
    return bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
        bandit=bandit_obj, max_inner_iter=40, progressive_x_linha=progressive,
    )


def main():
    print("=== Comparação: x' fixo (original) vs. x' progressivo vs. GVNS clássico ===\n")
    modos = ["fixo", "progressivo", "gvns_classico"]
    resumo = {m: [] for m in modos}

    for name, set_name, alvo_artigo in CASOS:
        path = ROOT / "instances" / set_name / f"{name}.vrp"
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        print(f"--- {name} (n={inst.n_customers}) -- artigo HG-UCB={alvo_artigo:.2f}% ---")
        for modo in modos:
            gaps = []
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                t0 = time.perf_counter()
                resultado = rodar(inst, cws, modo, seed)
                dt = time.perf_counter() - t0
                viavel, motivo = inst.is_feasible(resultado)
                if not viavel:
                    print(f"    ERRO: modo={modo} seed={seed} inviavel: {motivo}")
                    sys.exit(1)
                custo = inst.solution_cost(resultado)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                gaps.append(gap)
            melhor = min(gaps)
            media = sum(gaps) / len(gaps)
            resumo[modo].append(media)
            print(f"  [{modo:14s}] gap_melhor={melhor:6.2f}%  gap_medio={media:6.2f}%  (tempo últ. seed={dt:.2f}s)")
        print()

    print("=== Médias gerais (sobre as instâncias testadas) ===")
    for modo in modos:
        print(f"  {modo:14s}: {sum(resumo[modo])/len(resumo[modo]):.2f}%")
    media_artigo = sum(a for _, _, a in CASOS) / len(CASOS)
    print(f"  {'artigo HG-UCB':14s}: {media_artigo:.2f}%  (referência, não comparável 1:1 -- amostra pequena)")


if __name__ == "__main__":
    main()
