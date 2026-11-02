#!/bin/bash
# fila do Estudo 17 (selecao adaptativa do conjunto de operadores da busca local no ILS do PyVRP): ajuste -> selecao -> avaliacao.
# RETOMAVEL: cada modo pula o que ja esta no CSV (a busca estatica retoma pelo arquivo de estado). Se o WSL reiniciar, rode de novo este
# script. NUNCA dois runners no mesmo CSV. O orcamento e TEMPO de parede: nao rode outras cargas pesadas enquanto a fila roda.
# Uso (PowerShell): wsl -e bash -lc "cd /mnt/c/Pesquisas/learnheuristica/cvrp_bandit && (nohup setsid bash estudo_aos/fila_17.sh > results/aos_study/etapa17_fila_log.txt 2>&1 < /dev/null &)"
cd "$(dirname "$0")/.." || exit 1
source direcao1_local/venv/bin/activate
set -e
python3 -u estudo_aos/run_aos_etapa17.py --tuning --workers 11 > results/aos_study/etapa17_tuning_log.txt 2>&1
python3 -u estudo_aos/run_aos_etapa17.py --selecionar > results/aos_study/etapa17_selecionar_log.txt 2>&1
python3 -u estudo_aos/run_aos_etapa17.py --eval --workers 11 > results/aos_study/etapa17_eval_log.txt 2>&1
echo "FILA 17 CONCLUIDA"
