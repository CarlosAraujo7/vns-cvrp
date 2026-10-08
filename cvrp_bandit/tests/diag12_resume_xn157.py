#!/usr/bin/env python3
"""
diag12_resume_xn157.py
=======================
Script de RETOMADA do diagnóstico 12 (diag12_amostra_expandida.py), NÃO um
novo experimento. O processo em segundo plano do diag12 original (pid 1704,
rodando desde antes da última compactação de contexto desta sessão) foi
encerrado externamente antes de terminar: o log parou no meio da 16ª e
última instância (X-n157-k13), logo após completar só 2 das 8
configurações (`alpha=5.0/mii=20` e `alpha=5.0/mii=40`), sem nenhum
traceback Python no log e sem nenhuma linha de erro -- `ps aux` confirma
que o processo não existe mais. Isso é consistente com uma interrupção
externa do ambiente (esta sessão observou uma desconexão/reconexão do
servidor MCP `remote-devices` por volta do mesmo período), não com um erro
do código ou dos dados. As 15 primeiras instâncias (as 8 originais dos
diagnósticos 5-11 + A-n69-k9, B-n31-k5, B-n50-k8, B-n78-k10, CMT1,
X-n106-k14, X-n134-k13) já tinham completado as 8 configurações inteiras
antes da interrupção -- não precisam ser recalculadas.

Este script roda SOMENTE X-n157-k13, reimportando (não reimplementando) as
mesmas funções e constantes de diag12_amostra_expandida.py -- CASOS,
CONFIGURACOES, N_SEEDS, SEED_BASE, caminho_instancia, rodar_uma_semente --
garantindo que o resultado seja EXATAMENTE o que o diag12 original teria
produzido para esta instância, sem nenhuma lógica reimplementada à parte
(risco de divergência sutil). Como cada instância usa sua própria
`HGUCBBandit`/RNG construída do zero dentro de `rodar_uma_semente`, sem
nenhum estado compartilhado entre instâncias no script original, retomar
só a instância que faltou é equivalente a tê-la rodado dentro do batch
original.

Checagem de determinismo/regressão embutida: as duas primeiras linhas
(alpha=5,0/mii=20 e alpha=5,0/mii=40) já haviam aparecido no log parcial
antes da interrupção -- gap_medio=4,23%/colapso=17,6% e
gap_medio=4,13%/colapso=9,8%, respectivamente. Se este script não
reproduzir esses dois valores exatamente, é sinal de alerta (algo mudou
no código entre a execução original e esta retomada).

Diferença deliberada de instrumentação: imprime também pct_drift, que o
diag12 original só acumulava internamente (para o resumo geral, que nunca
chegou a ser impresso) mas não imprimia por instância -- então, para as 15
instâncias já completadas antes da interrupção, %drift por-instância
permanece indisponível a partir do log (só o %colapso e o gap% foram
impressos por instância); esta é a única instância das 16 com %drift
disponível. Registrado como limitação conhecida no diário, não escondido.

Uso: python diag12_resume_xn157.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag12_amostra_expandida import (  # noqa: E402
    CASOS, CONFIGURACOES, N_SEEDS, SEED_BASE, caminho_instancia, rodar_uma_semente,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402


def main():
    name, set_name, alvo_artigo = next(c for c in CASOS if c[0] == "X-n157-k13")
    path = caminho_instancia(name, set_name)
    inst = load_instance(path)
    cws = repair_route_count(inst, clarke_wright_savings(inst))

    print(f"=== Retomada diag12 (só a instância interrompida): {name} "
          f"(conjunto {set_name}, artigo HG-UCB={alvo_artigo:.2f}%) ===\n")
    t_inicio = time.perf_counter()

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
        print(f"  {rotulo:36s}  gap_medio={media:6.2f}%  "
              f"pct_colapso_desta_instancia={pct_colapso_inst:5.1f}%  "
              f"pct_drift_desta_instancia={pct_drift_inst:5.2f}%  (n={total_inst})")

    dt_total = time.perf_counter() - t_inicio
    print(f"\nTempo total de execução (só X-n157-k13, 8 configs x 5 seeds = 40 execuções): {dt_total:.1f}s")


if __name__ == "__main__":
    main()
