#!/bin/bash
# Fila da EMENDA 1 do Estudo 21 (S129: a mistura de P21 com 129 avaliacoes de ajuste; diario, Entrada 139). ESPERA o fim de fila_21.sh e so entao roda; RETOMAVEL (cada modo pula o
# que ja esta no CSV; o ES retoma pelo arquivo de estado COPIADO). NUNCA dois runners no mesmo CSV.
# Uso (PowerShell): wsl -e bash -lc "cd /mnt/c/Pesquisas/learnheuristica/cvrp_bandit && (nohup setsid bash estudo_aos/fila_21b.sh >> results/aos_study/etapa21b_fila_log.txt 2>&1 < /dev/null &)"
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "fila_21\.sh" > /dev/null; do sleep 60; done
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
echo "[$(date +%H:%M:%S)] FILA 21b INICIADA"
python3 -u estudo_aos/run_aos_etapa21b.py --ajuste --workers 11 >> "$R/etapa21b_ajuste_log.txt" 2>&1
python3 -u estudo_aos/run_aos_etapa21b.py --selecionar >> "$R/etapa21b_selecionar_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] validacao"
python3 -u estudo_aos/run_aos_etapa21b.py --validar --workers 11 >> "$R/etapa21b_validar_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] avaliacao"
python3 -u estudo_aos/run_aos_etapa21b.py --eval --workers 11 >> "$R/etapa21b_eval_log.txt" 2>&1
echo "FILA 21b CONCLUIDA"
