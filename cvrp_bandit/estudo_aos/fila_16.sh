#!/bin/bash
# fila do Estudo 16 (ALNS no CVRPTW): ajuste -> selecao -> avaliacao -> referencias VND. RETOMAVEL: cada modo pula o que ja esta no CSV
# (a busca estatica retoma pelo arquivo de estado). Se o WSL reiniciar, rode de novo este script. NUNCA dois runners no mesmo CSV.
# Uso (PowerShell): wsl -e bash -lc "cd /mnt/c/Pesquisas/learnheuristica/cvrp_bandit && (nohup setsid bash estudo_aos/fila_16.sh > results/aos_study/etapa16_fila_log.txt 2>&1 < /dev/null &)"
cd "$(dirname "$0")/.." || exit 1
source direcao1_local/venv/bin/activate
set -e
python3 -u estudo_aos/run_aos_etapa16.py --tuning-a --workers 11 > results/aos_study/etapa16_tuning-a_log.txt 2>&1
python3 -u estudo_aos/run_aos_etapa16.py --selecionar-a > results/aos_study/etapa16_selecionar-a_log.txt 2>&1
python3 -u estudo_aos/run_aos_etapa16.py --eval-a --workers 11 > results/aos_study/etapa16_eval-a_log.txt 2>&1
python3 -u estudo_aos/run_aos_etapa16.py --eval-ref --workers 11 > results/aos_study/etapa16_eval-ref_log.txt 2>&1
echo "FILA 16 CONCLUIDA"
