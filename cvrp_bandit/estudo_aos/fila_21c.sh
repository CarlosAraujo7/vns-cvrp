#!/bin/bash
# Fila do diagnostico exploratorio S_RP do Estudo 21 (diario, Entrada 140). ESPERA o fim de fila_21.sh e de fila_21b.sh; RETOMAVEL. NUNCA dois runners no mesmo CSV.
# Uso (PowerShell): wsl -e bash -lc "cd /mnt/c/Pesquisas/learnheuristica/cvrp_bandit && (nohup setsid bash estudo_aos/fila_21c.sh >> results/aos_study/etapa21c_fila_log.txt 2>&1 < /dev/null &)"
cd "$(dirname "$0")/.." || exit 1
sleep 120
while pgrep -f "fila_23\.sh" > /dev/null; do sleep 60; done
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
echo "[$(date +%H:%M:%S)] FILA 21c INICIADA"
python3 -u estudo_aos/run_aos_etapa21c.py --eval --workers 11 >> "$R/etapa21c_eval_log.txt" 2>&1
echo "FILA 21c CONCLUIDA"
