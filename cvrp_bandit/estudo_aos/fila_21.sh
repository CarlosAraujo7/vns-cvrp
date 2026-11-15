#!/bin/bash
# Fila do Estudo 21 (tamanho do conjunto de operadores no ALNS do CVRP; diario, Entrada 137). RETOMAVEL: cada modo pula o que ja esta no CSV e o ES retoma
# pelo arquivo de estado. Ordem: regressao exata -> P21 (ajuste da mistura, ajuste de RP, selecao, validacao, avaliacao) -> P12 (idem). NUNCA dois runners no mesmo CSV.
# Uso (PowerShell): wsl -e bash -lc "cd /mnt/c/Pesquisas/learnheuristica/cvrp_bandit && (nohup setsid bash estudo_aos/fila_21.sh >> results/aos_study/etapa21_fila_log.txt 2>&1 < /dev/null &)"
cd "$(dirname "$0")/.." || exit 1
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
echo "[$(date +%H:%M:%S)] FILA 21 INICIADA"
python3 -u estudo_aos/run_aos_etapa21.py --regressao --workers 11 >> "$R/etapa21_regressao_log.txt" 2>&1
for P in P21 P12; do
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
done
echo "FILA 21 CONCLUIDA"
