#!/bin/bash
# Fila do Estudo 22 (deslocamentos de implantacao do conjunto de 21 operadores; diario, Entradas 138 e 140). ESPERA o fim de fila_21.sh, fila_21b.sh e fila_21c.sh (precisa das misturas S129 da
# emenda do Estudo 21) e so entao roda; RETOMAVEL (cada modo pula o que ja esta no CSV). Ordem: regressao exata -> calibracao do desvio do RRT -> avaliacoes (cls, short, acc, longabc, longx).
# NUNCA dois runners no mesmo CSV.
# Uso (PowerShell): wsl -e bash -lc "cd /mnt/c/Pesquisas/learnheuristica/cvrp_bandit && (nohup setsid bash estudo_aos/fila_22.sh >> results/aos_study/etapa22_fila_log.txt 2>&1 < /dev/null &)"
cd "$(dirname "$0")/.." || exit 1
sleep 180
while pgrep -f "fila_21\.sh|fila_21b\.sh|fila_21c\.sh" > /dev/null; do sleep 60; done
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
echo "[$(date +%H:%M:%S)] FILA 22 INICIADA"
python3 -u estudo_aos/run_aos_etapa22.py --regressao --workers 11 >> "$R/etapa22_regressao_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] calibracao do RRT"
python3 -u estudo_aos/run_aos_etapa22.py --calibrar --workers 11 >> "$R/etapa22_calibrar_log.txt" 2>&1
for reg in cls short acc longabc longx; do
  echo "[$(date +%H:%M:%S)] avaliacao $reg"
  python3 -u estudo_aos/run_aos_etapa22.py --eval $reg --workers 11 >> "$R/etapa22_eval_${reg}_log.txt" 2>&1
done
echo "FILA 22 CONCLUIDA"
