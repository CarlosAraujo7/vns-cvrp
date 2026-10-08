#!/usr/bin/env python3
"""
diag13_cmt_restantes.py
========================
Diagnóstico #13, aprovado pelo pesquisador (via `AskUserQuestion`,
respondendo à Entrada 21 -- achado real #13) e recomendado de forma
independente pelo Gemini (mesma metodologia neutra das Entradas
9/16/18/20): testa as 3 instâncias CMT que faltam (CMT2, CMT3, CMT11 --
CMT1 já testada no diagnóstico 12), para determinar se a inversão da
comparação alpha=5,0/mii=160 vs. alpha=0,5/mii=160 causada por CMT1
(Entrada 21) reflete um padrão do conjunto CMT inteiro ou é uma
idiossincrasia de uma única instância.

Reusa (não reimplementa) EXATAMENTE as mesmas funções e constantes de
`diag12_amostra_expandida.py` -- CONFIGURACOES, N_SEEDS, SEED_BASE,
rodar_uma_semente -- para garantir que os resultados sejam diretamente
comparáveis, célula a célula, com os já registrados para CMT1 na Entrada
21 (mesmo código, mesmos seeds, só instância diferente).

Uso: python diag13_cmt_restantes.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag12_amostra_expandida import (  # noqa: E402
    CONFIGURACOES, N_SEEDS, SEED_BASE, rodar_uma_semente,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# valores de referencia do artigo (Tabela 13, coluna HG-UCB) -- mesma fonte
# ja usada desde a Entrada 13, tests/run_bandit_vns_validation.py
CASOS = [
    ("CMT2", 2.33),
    ("CMT3", 2.96),
    ("CMT11", 11.72),
]


def main():
    print("=== Diagnóstico 13: CMT2/CMT3/CMT11 (segue Entrada 21, achado #13) ===\n")
    t_inicio_total = time.perf_counter()

    resultados = {}  # nome -> {rotulo: (gap, colapso)}

    for name, alvo_artigo in CASOS:
        path = ROOT / "instances" / "CMT" / f"{name}.vrp"
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        print(f"--- {name} (conjunto CMT, artigo HG-UCB={alvo_artigo:.2f}%) ---")
        t_inst = time.perf_counter()
        resultados[name] = {}
        for rotulo, alpha, mii in CONFIGURACOES:
            gaps = []
            colapsadas_inst, drift_inst, total_inst = 0, 0, 0
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                resultado, contador = rodar_uma_semente(inst, cws, alpha, mii, seed)
                viavel, motivo = inst.is_feasible(resultado)
                if not viavel:
                    print(f"    ERRO: {rotulo} seed={seed} inviável: {motivo}")
                    sys.exit(1)
                custo = inst.solution_cost(resultado)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                gaps.append(gap)
                colapsadas_inst += contador["colapsadas"]
                drift_inst += contador["drift"]
                total_inst += contador["total"]
            media = sum(gaps) / len(gaps)
            pct_colapso_inst = 100 * colapsadas_inst / total_inst if total_inst else 0.0
            pct_drift_inst = 100 * drift_inst / total_inst if total_inst else 0.0
            resultados[name][rotulo] = (media, pct_colapso_inst)
            print(f"  {rotulo:36s}  gap_medio={media:6.2f}%  "
                  f"pct_colapso_desta_instancia={pct_colapso_inst:5.1f}%  "
                  f"pct_drift_desta_instancia={pct_drift_inst:5.2f}%  (n={total_inst})")
        print(f"  (tempo desta instância: {time.perf_counter()-t_inst:.1f}s)\n")

    dt_total = time.perf_counter() - t_inicio_total
    print(f"Tempo total de execução: {dt_total:.1f}s\n")

    print("=== Resumo GERAL (média sobre CMT2+CMT3+CMT11, as 3 novas) ===")
    for rotulo, alpha, mii in CONFIGURACOES:
        vals = [resultados[name][rotulo][0] for name, _ in CASOS]
        colapsos = [resultados[name][rotulo][1] for name, _ in CASOS]
        print(f"  {rotulo:36s}  gap_medio={sum(vals)/len(vals):6.2f}%  "
              f"colapso_medio={sum(colapsos)/len(colapsos):5.1f}%")

    # Comparacao direta alpha=5,0/mii160 vs alpha=0,5/mii160, incluindo CMT1 ja
    # conhecido (Entrada 21), para ver se o padrao de CMT1 se repete nas outras 3
    cmt1_a5 = 7.03
    cmt1_a05 = 4.83
    print("\n=== alpha=5,0/mii=160 vs alpha=0,5/mii=160, TODAS as 4 instancias CMT ===")
    a5_key = "alpha=5.0 / max_inner_iter=160"
    a05_key = "alpha=0.5 / max_inner_iter=160"
    print(f"  CMT1 (já conhecida, Entrada 21)   a5.0={cmt1_a5:5.2f}%  a0.5={cmt1_a05:5.2f}%  diff={cmt1_a5-cmt1_a05:+.2f}pp")
    for name, _ in CASOS:
        v5 = resultados[name][a5_key][0]
        v05 = resultados[name][a05_key][0]
        print(f"  {name:12s}                      a5.0={v5:5.2f}%  a0.5={v05:5.2f}%  diff={v5-v05:+.2f}pp")
    todos_a5 = [cmt1_a5] + [resultados[name][a5_key][0] for name, _ in CASOS]
    todos_a05 = [cmt1_a05] + [resultados[name][a05_key][0] for name, _ in CASOS]
    print(f"  MEDIA DAS 4 INSTANCIAS CMT:        a5.0={sum(todos_a5)/4:5.2f}%  a0.5={sum(todos_a05)/4:5.2f}%  "
          f"diff={sum(todos_a5)/4-sum(todos_a05)/4:+.2f}pp")


if __name__ == "__main__":
    main()
