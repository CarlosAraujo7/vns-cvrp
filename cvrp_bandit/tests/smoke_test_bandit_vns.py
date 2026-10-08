#!/usr/bin/env python3
"""
smoke_test_bandit_vns.py
=========================
Teste de FUMAÇA (não a validação completa da Tarefa #178) para bandit_vns()
recém-escrito (Algoritmo 3, HG-UCB). Objetivo: confirmar que roda sem
crashar, preserva viabilidade, produz custo sensato (não pior que a
construção inicial), e coletar diagnósticos sobre o "achado real" do
parâmetro b_t=0.2 morto (ver docstring de bandit_vns() em vns.py) -- com
que frequência o loop interno bate no teto de segurança max_inner_iter em
vez de parar pela condição do próprio artigo.

Uso: python smoke_test_bandit_vns.py
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from vns import bandit_vns  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    print("=== smoke test: bandit_vns() (Algoritmo 3, HG-UCB) ===\n")
    ok_geral = True
    rng = random.Random(2026)

    casos = [
        ("A-n32-k5", "A", 20),
        ("A-n80-k10", "raw/SetA/A", 20),
        ("X-n195-k51", "X", 10),
    ]

    for name, subdir, steps in casos:
        path = ROOT / "instances" / subdir / f"{name}.vrp"
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))
        custo_cws = inst.solution_cost(cws)

        diag = {"paradas_por_bt": 0, "paradas_por_teto": 0, "total_outer_steps": 0,
                "total_inner_iters": 0, "drifts": 0, "melhoras": 0}

        def on_step(step, custo, _diag=diag):
            _diag["total_outer_steps"] = step

        def on_inner(step, inner_iter, escolhido, melhorou, reward, drift, _diag=diag):
            _diag["total_inner_iters"] += 1
            if melhorou:
                _diag["melhoras"] += 1
            if drift:
                _diag["drifts"] += 1

        t0 = time.perf_counter()
        resultado = bandit_vns(inst, cws, rng, steps_max=steps, on_step=on_step, on_inner=on_inner)
        dt = time.perf_counter() - t0

        viavel, motivo = inst.is_feasible(resultado)
        custo_final = inst.solution_cost(resultado)
        nao_piorou = custo_final <= custo_cws + 1e-6
        gap = 100 * (custo_final - inst.optimum) / inst.optimum

        # conta quantos outer steps o loop interno bateu no teto -- reconstrua
        # rodando de novo com um contador por-step seria mais preciso, mas
        # para o smoke test, o sinal agregado (inner_iters / outer_steps)
        # já indica se o teto está sendo usado pesadamente.
        media_inner_por_outer = diag["total_inner_iters"] / max(diag["total_outer_steps"], 1)

        ok = viavel and nao_piorou
        ok_geral = ok_geral and ok
        status = "OK" if ok else "FALHA"
        print(
            f"[{status}] {name} (n={inst.n_customers}, steps={steps}): "
            f"CWS={custo_cws:.2f} -> bandit_vns={custo_final:.2f} "
            f"(gap_otimo={gap:.2f}%) viavel={viavel} nao_piorou_vs_cws={nao_piorou}"
        )
        print(
            f"       tempo={dt:.2f}s  inner_iters_total={diag['total_inner_iters']} "
            f"outer_steps={diag['total_outer_steps']} media_inner/outer={media_inner_por_outer:.1f} "
            f"(k=5 operadores -- media>5 sugere que o loop interno esta indo alem do cold-start) "
            f"melhoras={diag['melhoras']} drifts_adwin={diag['drifts']}"
        )
        if not viavel:
            print(f"       motivo inviabilidade: {motivo}")

    print()
    if ok_geral:
        print("RESULTADO: OK -- bandit_vns() preserva viabilidade e não piora em relação à construção inicial.")
    else:
        print("RESULTADO: FALHOU -- ver detalhes acima.")
        sys.exit(1)


if __name__ == "__main__":
    main()
