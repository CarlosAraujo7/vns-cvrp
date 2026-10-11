#!/bin/bash
# Fila dos Estudos 10 e 9 (pre-registros: diario, Entradas 104 e 103). Ordem: ajuste do Estudo 10 (SYN_TUNE_L) ->
# ajuste do Estudo 9 (SYN_TUNE) -> selecao -> avaliacao do Estudo 10 -> avaliacao do Estudo 9 (ALNS) -> VNS ao mesmo orcamento.
# Resumivel: cada runner pula as chaves ja gravadas; basta relancar.  Roda no WSL (venv de direcao1_local).
cd "$(dirname "$0")" || exit 1
RES=../results/aos_study
PY=../direcao1_local/venv/bin/python3
$PY -u run_aos_etapa10.py --tuning-static --workers 11 >> $RES/etapa10_tuning_log.txt 2>&1
$PY -u run_aos_etapa10.py --selecionar >> $RES/etapa10_tuning_log.txt 2>&1
echo "ajuste do Estudo 10 concluido em $(date)" >> $RES/fila910_log.txt
$PY -u run_aos_etapa9.py --tuning-grid --workers 11 >> $RES/etapa9_tuning_log.txt 2>&1
$PY -u run_aos_etapa9.py --selecionar >> $RES/etapa9_tuning_log.txt 2>&1
echo "ajuste do Estudo 9 concluido em $(date)" >> $RES/fila910_log.txt
$PY -u run_aos_etapa10.py --eval --workers 11 >> $RES/etapa10_eval_log.txt 2>&1
echo "avaliacao do Estudo 10 concluida em $(date)" >> $RES/fila910_log.txt
$PY -u run_aos_etapa9.py --eval --workers 11 >> $RES/etapa9_eval_log.txt 2>&1
echo "avaliacao do Estudo 9 (ALNS) concluida em $(date)" >> $RES/fila910_log.txt
$PY -u run_aos_etapa9.py --vns --workers 11 >> $RES/etapa9_vns_log.txt 2>&1
echo "FILA_910_COMPLETA $(date)" >> $RES/fila910_log.txt
