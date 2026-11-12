#!/bin/bash
# fila do Estudo 19 (replicacao confirmatoria do indicio do Estudo 18): regressao exata -> avaliacao (4 bracos congelados, SYN_VAL_W e Solomon com seeds novas).
# RETOMAVEL: o modo de avaliacao pula o que ja esta no CSV. Se o WSL reiniciar, rode de novo este script. NUNCA dois runners no mesmo CSV.
# Uso (PowerShell): wsl -e bash -lc "cd /mnt/c/Pesquisas/learnheuristica/cvrp_bandit && (nohup setsid bash estudo_aos/fila_19.sh >> results/aos_study/etapa19_fila_log.txt 2>&1 < /dev/null &)"
cd "$(dirname "$0")/.." || exit 1
source direcao1_local/venv/bin/activate
set -e
python3 -u estudo_aos/run_aos_etapa19.py --regressao --workers 11 >> results/aos_study/etapa19_regressao_log.txt 2>&1
python3 -u estudo_aos/run_aos_etapa19.py --eval --workers 11 >> results/aos_study/etapa19_eval_log.txt 2>&1
echo "FILA 19 CONCLUIDA"
