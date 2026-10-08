#!/bin/bash
# Trava do estudo AOS (diario, Entrada 67): para o runner quando as 5.920
# execucoes de B=4000 estiverem gravadas, para decidir o destino de B=20000
# com o pesquisador antes de gastar ~34 h de maquina. Roda dentro do WSL.
cd "$(dirname "$0")/../results/aos_study" || exit 1
cont() { awk -F, -v b="$1" '$2==b' full.csv | wc -l; }
until [ "$(cont 4000)" -ge 5920 ]; do sleep 60; done
pkill -f run_aos_study.py
sleep 5
echo "B4000_COMPLETO_E_EXECUCAO_PARADA $(date)" > trava_b4000.status
echo "linhas B=4000: $(cont 4000)  B=20000: $(cont 20000)" >> trava_b4000.status
