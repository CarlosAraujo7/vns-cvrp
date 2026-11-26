#!/bin/bash
# Fila do Estudo 31 (clonagem de operadores; diario, Entrada 160). ESPERA o fim de fila_30.sh; depois roda a regressao exata (uma vez), o ajuste da mistura de 21 logits (ES de 8 geracoes = 33 avaliacoes), a selecao
# e a avaliacao nas 51 instancias do Estudo 21; RETOMAVEL (cada modo pula o que ja esta no CSV). NUNCA dois runners no mesmo CSV.
cd "$(dirname "$0")/.." || exit 1
sleep 120
while pgrep -f "fila_30\.sh" > /dev/null; do sleep 60; done
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
echo "[$(date +%H:%M:%S)] FILA 31 INICIADA"
if ! grep -q "REGRESSAO EXATA: OK" "$R/etapa31_regressao_log.txt" 2>/dev/null; then
  python3 -u estudo_aos/run_aos_etapa31.py --regressao --workers 11 >> "$R/etapa31_regressao_log.txt" 2>&1
fi
echo "[$(date +%H:%M:%S)] ajuste da mistura (clones)"
python3 -u estudo_aos/run_aos_etapa31.py --ajuste --workers 11 >> "$R/etapa31_ajuste_log.txt" 2>&1
python3 -u estudo_aos/run_aos_etapa31.py --selecionar >> "$R/etapa31_selecionar_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] avaliacao"
python3 -u estudo_aos/run_aos_etapa31.py --eval --workers 11 >> "$R/etapa31_eval_log.txt" 2>&1
echo "FILA 31 CONCLUIDA"
