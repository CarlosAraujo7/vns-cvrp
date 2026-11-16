#!/bin/bash
# Fila do Estudo 28 (o conjunto de 21 operadores no ALNS do CVRPTW; diario, Entrada 148). ESPERA o fim de fila_22.sh, fila_26.sh e fila_27.sh; depois confere as instancias, roda a regressao exata (uma vez), o ajuste da
# mistura (ES de 16 geracoes), a selecao, a validacao e a avaliacao; RETOMAVEL (cada modo pula o que ja esta no CSV). NUNCA dois runners no mesmo CSV.
cd "$(dirname "$0")/.." || exit 1
sleep 210
while pgrep -f "fila_22\.sh|fila_26\.sh|fila_27\.sh" > /dev/null; do sleep 60; done
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
echo "[$(date +%H:%M:%S)] FILA 28 INICIADA"
python3 -u estudo_aos/run_aos_etapa28.py --instancias >> "$R/etapa28_instancias_log.txt" 2>&1
if ! grep -q "REGRESSAO EXATA: OK" "$R/etapa28_regressao_log.txt" 2>/dev/null; then
  python3 -u estudo_aos/run_aos_etapa28.py --regressao --workers 11 >> "$R/etapa28_regressao_log.txt" 2>&1
fi
echo "[$(date +%H:%M:%S)] ajuste da mistura"
python3 -u estudo_aos/run_aos_etapa28.py --ajuste --workers 11 >> "$R/etapa28_ajuste_log.txt" 2>&1
python3 -u estudo_aos/run_aos_etapa28.py --selecionar >> "$R/etapa28_selecionar_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] validacao"
python3 -u estudo_aos/run_aos_etapa28.py --validar --workers 11 >> "$R/etapa28_validar_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] avaliacao"
python3 -u estudo_aos/run_aos_etapa28.py --eval --workers 11 >> "$R/etapa28_eval_log.txt" 2>&1
echo "FILA 28 CONCLUIDA"
