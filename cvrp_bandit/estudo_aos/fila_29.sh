#!/bin/bash
# Fila do Estudo 29 (os bracos adaptativos ajustados no ALNS do CVRPTW com 21 operadores; diario, Entrada 156). Roda a regressao exata (uma vez, contra etapa28_eval.csv), o ajuste por busca aleatoria do RP e
# do HG-UCB (33 configuracoes cada), a selecao, a validacao e a avaliacao; RETOMAVEL (cada modo pula o que ja esta no CSV). NUNCA dois runners no mesmo CSV.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "fila_2[0-8]\.sh" > /dev/null; do sleep 60; done
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
echo "[$(date +%H:%M:%S)] FILA 29 INICIADA"
if ! grep -q "REGRESSAO EXATA: OK" "$R/etapa29_regressao_log.txt" 2>/dev/null; then
  python3 -u estudo_aos/run_aos_etapa29.py --regressao --workers 11 >> "$R/etapa29_regressao_log.txt" 2>&1
fi
for f in RP HG; do
  echo "[$(date +%H:%M:%S)] ajuste $f"
  python3 -u estudo_aos/run_aos_etapa29.py --ajuste $f --workers 11 >> "$R/etapa29_ajuste_${f}_log.txt" 2>&1
done
python3 -u estudo_aos/run_aos_etapa29.py --selecionar >> "$R/etapa29_selecionar_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] validacao"
python3 -u estudo_aos/run_aos_etapa29.py --validar --workers 11 >> "$R/etapa29_validar_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] avaliacao"
python3 -u estudo_aos/run_aos_etapa29.py --eval --workers 11 >> "$R/etapa29_eval_log.txt" 2>&1
echo "FILA 29 CONCLUIDA"
