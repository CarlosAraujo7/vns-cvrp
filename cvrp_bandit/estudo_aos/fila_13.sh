#!/bin/bash
# Fila do Estudo 13 (TSP): ajuste -> selecao -> avaliacoes. Retomavel (cada modo pula o que ja esta no CSV).
cd "$(dirname "$0")" || exit 1
PY=../direcao1_local/venv/bin/python3
LOG=../results/aos_study/fila13_log.txt
run() {
  nome="${1#--}"
  $PY -u run_aos_etapa13.py "$@" >> "../results/aos_study/etapa13_${nome}_log.txt" 2>&1
  echo "$1 concluido em $(date)" >> "$LOG"
}
run --tuning-static-v --workers 11
run --selecionar-v
run --tuning-a --workers 11
run --selecionar-a
run --eval-v --workers 11
run --eval-pc --workers 11
run --eval-a --workers 11
run --eval-ref --workers 11
echo "FILA_13_COMPLETA $(date)" >> "$LOG"
