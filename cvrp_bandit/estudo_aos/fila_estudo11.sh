#!/bin/bash
# Fila do Estudo 11 (pre-registro: diario, Entrada 106): espera a fila dos Estudos 9 e 10 terminar.
cd "$(dirname "$0")" || exit 1
RES=../results/aos_study
PY=../direcao1_local/venv/bin/python3
until grep -q "FILA_910_COMPLETA" $RES/fila910_log.txt 2>/dev/null; do sleep 60; done
$PY -u run_aos_etapa11.py --tuning-static --workers 11 >> $RES/etapa11_tuning_log.txt 2>&1
$PY -u run_aos_etapa11.py --selecionar >> $RES/etapa11_tuning_log.txt 2>&1
echo "ajuste do Estudo 11 concluido em $(date)" >> $RES/fila11_log.txt
$PY -u run_aos_etapa11.py --eval74 --workers 11 >> $RES/etapa11_eval74_log.txt 2>&1
echo "eval74 do Estudo 11 concluida em $(date)" >> $RES/fila11_log.txt
$PY -u run_aos_etapa11.py --evalL --workers 11 >> $RES/etapa11_evalL_log.txt 2>&1
echo "FILA_11_COMPLETA $(date)" >> $RES/fila11_log.txt
