#!/bin/bash
# Fila: espera o piloto 2 do Estudo 8 terminar e roda a calibracao do ALNS (Estudo 9) em SYN_TUNE.
cd "$(dirname "$0")" || exit 1
PY=../direcao1_local/venv/bin/python3
until grep -q "^concluido" ../results/aos_study/etapa8_pilot2_log.txt 2>/dev/null; do sleep 15; done
$PY -u run_aos_etapa9.py --calib --workers 11 > ../results/aos_study/etapa9_calib_log.txt 2>&1
