#!/bin/bash
# Fila: (1) espera o piloto das politicas do ALNS terminar; (2) Estudo 8 (avaliacao, pre-registro: Entrada 101);
# (3) gera as instancias sinteticas GRANDES (Estudo 10) com o HGS de referencia numa maquina ociosa (o HGS e limitado
# por tempo: contencao de CPU pioraria a referencia); (4) grava as solucoes iniciais das novas instancias.
cd "$(dirname "$0")" || exit 1
RES=../results/aos_study
PY=../direcao1_local/venv/bin/python3
until grep -q "^concluido" $RES/etapa9_pilot_log.txt 2>/dev/null; do sleep 30; done
$PY -u run_aos_etapa8.py --full --workers 11 > $RES/etapa8_full_log.txt 2>&1
echo "Estudo 8 concluido em $(date)" >> $RES/fila_log.txt
$PY -u gerar_instancias_sinteticas.py SYN_TUNE_L SYN_VAL_L > $RES/gerar_L_log.txt 2>&1
echo "instancias grandes geradas em $(date)" >> $RES/fila_log.txt
$PY -u gerar_solucoes_iniciais.py > $RES/gerar_init_L_log.txt 2>&1
echo "FILA_COMPLETA $(date)" >> $RES/fila_log.txt
