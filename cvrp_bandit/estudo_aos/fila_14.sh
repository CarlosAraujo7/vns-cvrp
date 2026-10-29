#!/bin/bash
# Fila do Estudo 14 (CVRPTW, VND): ajuste -> selecao -> avaliacoes. Retomavel (cada modo pula o que ja esta no CSV).
cd "$(dirname "$0")" || exit 1
PY=../direcao1_local/venv/bin/python3
LOG=../results/aos_study/fila14_log.txt
run() {
  nome="${1#--}"
  $PY -u run_aos_etapa14.py "$@" >> "../results/aos_study/etapa14_${nome}_log.txt" 2>&1
  echo "$1 concluido em $(date)" >> "$LOG"
}
run --tuning-static-v --workers 11
run --selecionar-v
run --eval-v --workers 11
run --eval-pc --workers 11
echo "FILA_14_COMPLETA $(date)" >> "$LOG"
