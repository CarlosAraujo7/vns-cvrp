#!/bin/bash
# Fila do Estudo 23 (fidelidade e mecanismo da vantagem dos pesos adaptativos em P21; diario, Entrada 141). ESPERA o fim de fila_21.sh, fila_21b.sh e fila_21c.sh (precisa de S129 e de SRP) e
# roda ANTES de fila_22.sh; RETOMAVEL (cada modo pula o que ja esta no CSV). Ordem: calibracao da escala de q e conjunto podado -> avaliacao em P21 -> avaliacao em P7. NUNCA dois runners no mesmo CSV.
# Uso (PowerShell): wsl -e bash -lc "cd /mnt/c/Pesquisas/learnheuristica/cvrp_bandit && (nohup setsid bash estudo_aos/fila_23.sh >> results/aos_study/etapa23_fila_log.txt 2>&1 < /dev/null &)"
cd "$(dirname "$0")/.." || exit 1
sleep 240
while pgrep -f "fila_21\.sh|fila_21b\.sh|fila_21c\.sh" > /dev/null; do sleep 60; done
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
echo "[$(date +%H:%M:%S)] FILA 23 INICIADA"
python3 -u estudo_aos/run_aos_etapa23.py --calibrar --workers 11 >> "$R/etapa23_calibrar_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] avaliacao P21"
python3 -u estudo_aos/run_aos_etapa23.py --eval p21 --workers 11 >> "$R/etapa23_eval_p21_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] avaliacao P7"
python3 -u estudo_aos/run_aos_etapa23.py --eval p7 --workers 11 >> "$R/etapa23_eval_p7_log.txt" 2>&1
echo "FILA 23 CONCLUIDA"
