#!/bin/bash
# RETOMADA do Estudo 20 depois que o WSL parou (a maquina dormiu em 13-15/11/2026 com o TSP completo e o CVRP no ajuste de static65): so o CVRP.
# O TSP esta completo (ajuste, selecao, validacao e avaliacao) e a regressao exata ja passou duas vezes, por isso este script nao a repete; os passos, os
# arquivos e a ordem sao os de fila_20.sh. RETOMAVEL (cada modo pula o que ja esta no CSV; o ES retoma pelo arquivo de estado). NUNCA dois runners no mesmo CSV.
# Uso (PowerShell): wsl -e bash -lc "cd /mnt/c/Pesquisas/learnheuristica/cvrp_bandit && (nohup setsid bash estudo_aos/fila_20_cvrp.sh >> results/aos_study/etapa20_fila_log.txt 2>&1 < /dev/null &)"
cd "$(dirname "$0")/.." || exit 1
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
P=CVRP
for m in static65 RP tilt; do
  python3 -u estudo_aos/run_aos_etapa20.py --problema $P --tuning $m --workers 11 >> "$R/etapa20_tuning_${m}_${P}_log.txt" 2>&1
done
python3 -u estudo_aos/run_aos_etapa20.py --problema $P --selecionar >> "$R/etapa20_selecionar_${P}_log.txt" 2>&1
python3 -u estudo_aos/run_aos_etapa20.py --problema $P --validar --workers 11 >> "$R/etapa20_validar_${P}_log.txt" 2>&1
python3 -u estudo_aos/run_aos_etapa20.py --problema $P --eval --workers 11 >> "$R/etapa20_eval_${P}_log.txt" 2>&1
echo "FILA 20 CVRP CONCLUIDA"
