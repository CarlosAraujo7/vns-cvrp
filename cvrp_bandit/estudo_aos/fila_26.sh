#!/bin/bash
# Fila do Estudo 26 (replica das conclusoes de P21 em instancias NOVAS; diario, Entradas 145 e 146). ESPERA o fim de fila_22.sh e do preparo das instancias (preparar_estudo26.sh); depois confere os
# conjuntos, roda a regressao exata (uma vez) e a avaliacao; RETOMAVEL (pula o que ja esta no CSV). NUNCA dois runners no mesmo CSV.
cd "$(dirname "$0")/.." || exit 1
sleep 90
while pgrep -f "fila_22\.sh|preparar_estudo26\.sh" > /dev/null; do sleep 60; done
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
echo "[$(date +%H:%M:%S)] FILA 26 INICIADA"
python3 -u estudo_aos/run_aos_etapa26.py --instancias >> "$R/etapa26_instancias_log.txt" 2>&1
if ! grep -q "REGRESSAO EXATA: OK" "$R/etapa26_regressao_log.txt" 2>/dev/null; then       # controle de implementacao, so uma vez
  python3 -u estudo_aos/run_aos_etapa26.py --regressao --workers 11 >> "$R/etapa26_regressao_log.txt" 2>&1
fi
echo "[$(date +%H:%M:%S)] avaliacao"
python3 -u estudo_aos/run_aos_etapa26.py --eval --workers 11 >> "$R/etapa26_eval_log.txt" 2>&1
echo "FILA 26 CONCLUIDA"
