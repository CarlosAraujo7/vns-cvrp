#!/bin/bash
# Fila do Estudo 24 (quais politicas de selecao aproveitam o conjunto de 21 operadores; diario, Entrada 142). ESPERA o fim de fila_21b.sh e fila_22.sh; RETOMAVEL. NUNCA dois runners no mesmo CSV.
# Uso (PowerShell): wsl -e bash -lc "cd /mnt/c/Pesquisas/learnheuristica/cvrp_bandit && (nohup setsid bash estudo_aos/fila_24.sh >> results/aos_study/etapa24_fila_log.txt 2>&1 < /dev/null &)"
cd "$(dirname "$0")/.." || exit 1
sleep 420
while pgrep -f "fila_21b\.sh|fila_22\.sh|fila_26\.sh|fila_27\.sh|fila_28\.sh" > /dev/null; do sleep 60; done
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
echo "[$(date +%H:%M:%S)] FILA 24 INICIADA"
python3 -u estudo_aos/run_aos_etapa24.py --eval --workers 11 >> "$R/etapa24_eval_log.txt" 2>&1
echo "FILA 24 CONCLUIDA"
