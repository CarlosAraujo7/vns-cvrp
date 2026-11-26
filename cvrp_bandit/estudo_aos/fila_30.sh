#!/bin/bash
# Fila do Estudo 30 (o que os pesos adaptativos acrescentam a uma faixa de remocao calibrada; diario, Entrada 159). Roda a regressao exata (uma vez, contra etapa26_eval.csv), a calibracao da escala de q do RP,
# a selecao, a validacao (que gera os pesos do SRPq) e a avaliacao nas 98 instancias novas; RETOMAVEL (cada modo pula o que ja esta no CSV). NUNCA dois runners no mesmo CSV.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "fila_2[0-9]\.sh" > /dev/null; do sleep 60; done
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
echo "[$(date +%H:%M:%S)] FILA 30 INICIADA"
if ! grep -q "REGRESSAO EXATA: OK" "$R/etapa30_regressao_log.txt" 2>/dev/null; then
  python3 -u estudo_aos/run_aos_etapa30.py --regressao --workers 11 >> "$R/etapa30_regressao_log.txt" 2>&1
fi
echo "[$(date +%H:%M:%S)] calibracao da escala de q do RP"
python3 -u estudo_aos/run_aos_etapa30.py --calibrar --workers 11 >> "$R/etapa30_calibrar_log.txt" 2>&1
python3 -u estudo_aos/run_aos_etapa30.py --selecionar >> "$R/etapa30_selecionar_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] validacao"
python3 -u estudo_aos/run_aos_etapa30.py --validar --workers 11 >> "$R/etapa30_validar_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] avaliacao"
python3 -u estudo_aos/run_aos_etapa30.py --eval --workers 11 >> "$R/etapa30_eval_log.txt" 2>&1
echo "FILA 30 CONCLUIDA"
