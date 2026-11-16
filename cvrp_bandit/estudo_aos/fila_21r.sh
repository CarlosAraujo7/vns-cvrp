#!/bin/bash
# RETOMADA de P12 do Estudo 21 (diario, Entrada 142: a ordem das filas mudou para rodar antes o Estudo 23, o diagnostico S_RP e a emenda S129). Os passos, os arquivos e a ordem sao os de P12 em
# fila_21.sh; RETOMAVEL (o ES retoma pelo arquivo de estado; cada modo pula o que ja esta no CSV). ESPERA o fim de fila_23.sh, fila_21c.sh e fila_21b.sh. NUNCA dois runners no mesmo CSV.
cd "$(dirname "$0")/.." || exit 1
sleep 480
while pgrep -f "fila_23\.sh|fila_21c\.sh|fila_21b\.sh|fila_22\.sh|fila_24\.sh" > /dev/null; do sleep 60; done
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
P=P12
echo "[$(date +%H:%M:%S)] FILA 21r (P12) INICIADA"
for m in static RP; do
  echo "[$(date +%H:%M:%S)] ajuste $P/$m"
  python3 -u estudo_aos/run_aos_etapa21.py --pool $P --ajuste $m --workers 11 >> "$R/etapa21_ajuste_${m}_${P}_log.txt" 2>&1
done
python3 -u estudo_aos/run_aos_etapa21.py --pool $P --selecionar >> "$R/etapa21_selecionar_${P}_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] validacao $P"
python3 -u estudo_aos/run_aos_etapa21.py --pool $P --validar --workers 11 >> "$R/etapa21_validar_${P}_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] avaliacao $P"
python3 -u estudo_aos/run_aos_etapa21.py --pool $P --eval --workers 11 >> "$R/etapa21_eval_${P}_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] $P CONCLUIDO"
echo "FILA 21r CONCLUIDA"
