#!/bin/bash
# Fila do Estudo 27 (o numero de rotas modera a vantagem dos pesos adaptativos? manipulacao da capacidade; diario, Entradas 145 e 147). ESPERA o fim de fila_22.sh, fila_26.sh e do preparo das instancias
# (preparar_estudo26.sh e preparar_estudo27.sh); depois confere o conjunto SYN_CAP e roda a avaliacao; RETOMAVEL. NUNCA dois runners no mesmo CSV.
cd "$(dirname "$0")/.." || exit 1
sleep 150
while pgrep -f "fila_22\.sh|fila_26\.sh|preparar_estudo26\.sh|preparar_estudo27\.sh" > /dev/null; do sleep 60; done
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
echo "[$(date +%H:%M:%S)] FILA 27 INICIADA"
python3 -u estudo_aos/run_aos_etapa27.py --instancias >> "$R/etapa27_instancias_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] avaliacao"
python3 -u estudo_aos/run_aos_etapa27.py --eval --workers 11 >> "$R/etapa27_eval_log.txt" 2>&1
echo "FILA 27 CONCLUIDA"
