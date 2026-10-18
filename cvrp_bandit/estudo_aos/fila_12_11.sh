#!/bin/bash
# Fila dos Estudos 12 e 11 (pre-registros: diario, Entradas 109 e 106). Espera a fila 9/10 (inclui as referencias VNS do Estudo 9).
cd "$(dirname "$0")" || exit 1
RES=../results/aos_study
PY=../direcao1_local/venv/bin/python3
until grep -q "FILA_910_COMPLETA" $RES/fila910_log.txt 2>/dev/null; do sleep 60; done
$PY -u run_aos_etapa12.py --tuning-static --workers 11 >> $RES/etapa12_tuning_log.txt 2>&1
$PY -u run_aos_etapa12.py --selecionar >> $RES/etapa12_tuning_log.txt 2>&1
echo "ajuste do Estudo 12 concluido em $(date)" >> $RES/fila12_log.txt
$PY -u run_aos_etapa12.py --eval --workers 11 >> $RES/etapa12_eval_log.txt 2>&1
echo "avaliacao do Estudo 12 concluida em $(date)" >> $RES/fila12_log.txt
$PY -u run_aos_etapa11.py --tuning-static --workers 11 >> $RES/etapa11_tuning_log.txt 2>&1
$PY -u run_aos_etapa11.py --selecionar >> $RES/etapa11_tuning_log.txt 2>&1
echo "ajuste do Estudo 11 concluido em $(date)" >> $RES/fila12_log.txt
$PY -u run_aos_etapa11.py --eval74 --workers 11 >> $RES/etapa11_eval74_log.txt 2>&1
$PY -u run_aos_etapa11.py --evalL --workers 11 >> $RES/etapa11_evalL_log.txt 2>&1
echo "FILA_12_11_COMPLETA $(date)" >> $RES/fila12_log.txt
