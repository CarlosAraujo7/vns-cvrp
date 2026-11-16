#!/bin/bash
# Fila do Estudo 25 (deslocamentos de implantacao em instancias com folga; diario, Entrada 144). ESPERA o fim de fila_22.sh e de fila_24.sh; RETOMAVEL. NUNCA dois runners no mesmo CSV.
# Uso (PowerShell): wsl -e bash -lc "cd /mnt/c/Pesquisas/learnheuristica/cvrp_bandit && (nohup setsid bash estudo_aos/fila_25.sh >> results/aos_study/etapa25_fila_log.txt 2>&1 < /dev/null &)"
cd "$(dirname "$0")/.." || exit 1
sleep 600
while pgrep -f "fila_22\.sh|fila_26\.sh|fila_27\.sh|fila_24\.sh" > /dev/null; do sleep 60; done
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
echo "[$(date +%H:%M:%S)] FILA 25 INICIADA"
for reg in acc short; do
  echo "[$(date +%H:%M:%S)] avaliacao $reg"
  python3 -u estudo_aos/run_aos_etapa25.py --eval $reg --workers 11 >> "$R/etapa25_eval_${reg}_log.txt" 2>&1
done
echo "FILA 25 CONCLUIDA"
