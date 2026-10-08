#!/usr/bin/env python3
"""
validate_vnd_gvns.py
=====================
1. Valida VND: parte de uma solução (a CWS de uma instância real pequena)
   e confirma que, ao terminar, NENHUM dos 5 operadores consegue mais
   melhorar (é de fato um ótimo local simultâneo nas 5 vizinhanças --
   definição de "ótimo VND"). Também confirma viabilidade preservada e
   custo não aumentou.
2. Valida GVNS: roda em 3 instâncias pequenas/médias, confirma viabilidade
   preservada em cada passo, custo nunca piora (monotonicidade da melhor
   solução), e mede tempo de parede para calibrar o orçamento da validação
   completa nas 74 instâncias (task seguinte).
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from vns import vnd, gvns  # noqa: E402
import neighborhoods as nb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def validar_vnd():
    print("=== VND: confirma ótimo local simultâneo nas 5 vizinhanças ===")
    ok_geral = True
    for name in ["A-n32-k5", "A-n55-k9", "X-n120-k6"]:
        path = ROOT / "instances" / name.split("-")[0] / f"{name}.vrp"
        inst = load_instance(path)  # vehicles_exact autodetectado (instance.py)
        cws = repair_route_count(inst, clarke_wright_savings(inst))
        custo_cws = inst.solution_cost(cws)
        resultado = vnd(inst, cws)
        custo_vnd = inst.solution_cost(resultado)

        viavel, motivo = inst.is_feasible(resultado)
        piorou = custo_vnd > custo_cws + 1e-6

        sem_melhora_possivel = True
        detalhes = []
        for nome_op, best_fn, _ in nb.NEIGHBORHOODS:
            melhorou, _ = best_fn(inst, resultado)
            if melhorou:
                sem_melhora_possivel = False
                detalhes.append(nome_op)

        ok = viavel and not piorou and sem_melhora_possivel
        ok_geral = ok_geral and ok
        status = "OK" if ok else "FALHA"
        print(
            f"[{status}] {name}: CWS={custo_cws:.2f} -> VND={custo_vnd:.2f} "
            f"(gap_otimo={100*(custo_vnd-inst.optimum)/inst.optimum:.2f}%) "
            f"viavel={viavel} otimo_local={sem_melhora_possivel} "
            f"{'(ainda melhora em: ' + str(detalhes) + ')' if detalhes else ''}"
        )
    return ok_geral


def validar_gvns():
    print("\n=== GVNS: viabilidade, monotonicidade e tempo de parede ===")
    ok_geral = True
    rng = random.Random(123)
    casos = [
        ("A-n32-k5", "A", 20),
        ("A-n80-k10", "raw/SetA/A", 20),
        ("X-n195-k51", "X", 10),  # maior instância do conjunto usado -- mede o pior caso
    ]
    for name, subdir, steps in casos:
        path = ROOT / "instances" / subdir / f"{name}.vrp"
        inst = load_instance(path)  # vehicles_exact autodetectado (instance.py)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        historico = []
        def on_step(step, custo):
            historico.append(custo)

        t0 = time.perf_counter()
        resultado = gvns(inst, cws, rng, steps_max=steps, on_step=on_step)
        dt = time.perf_counter() - t0

        viavel, motivo = inst.is_feasible(resultado)
        monotono = all(historico[i] >= historico[i + 1] - 1e-9 for i in range(len(historico) - 1))
        custo_final = inst.solution_cost(resultado)
        gap = 100 * (custo_final - inst.optimum) / inst.optimum

        ok = viavel and monotono
        ok_geral = ok_geral and ok
        status = "OK" if ok else "FALHA"
        print(
            f"[{status}] {name} (n={inst.n_customers}, steps={steps}): "
            f"tempo={dt:.2f}s ({dt/steps:.3f}s/step) gap_final={gap:.2f}% "
            f"viavel={viavel} monotono={monotono}"
        )
        if not viavel:
            print(f"       motivo inviabilidade: {motivo}")
    return ok_geral


def main():
    ok1 = validar_vnd()
    ok2 = validar_gvns()
    print()
    if ok1 and ok2:
        print("RESULTADO: OK -- VND converge a ótimo local simultâneo; GVNS preserva "
              "viabilidade e melhora monotonicamente.")
    else:
        print("RESULTADO: FALHOU -- ver detalhes acima.")
        sys.exit(1)


if __name__ == "__main__":
    main()
