#!/bin/bash
# fila do Estudo 18 (politicas com estado e esforco de ajuste igual, ALNS no CVRPTW): regressao exata -> ajuste (8 metodos) -> selecao ->
# validacao -> avaliacao. RETOMAVEL: cada modo pula o que ja esta no CSV (as buscas ES retomam pelo arquivo de estado). Se o WSL reiniciar,
# rode de novo este script. NUNCA dois runners no mesmo CSV. O orcamento e em avaliacoes de distancia (independe do hardware), entao outras
# cargas so retardam a fila. Ordem: static18 primeiro (sched, loglin e tilt partem dele).
# Uso (PowerShell): wsl -e bash -lc "cd /mnt/c/Pesquisas/learnheuristica/cvrp_bandit && (nohup setsid bash estudo_aos/fila_18.sh > results/aos_study/etapa18_fila_log.txt 2>&1 < /dev/null &)"
cd "$(dirname "$0")/.." || exit 1
source direcao1_local/venv/bin/activate
set -e
python3 -u estudo_aos/run_aos_etapa18.py --regressao --workers 11 >> results/aos_study/etapa18_regressao_log.txt 2>&1
for m in static18 RP18 ctxucb linucb qlearn sched loglin tilt; do
  python3 -u estudo_aos/run_aos_etapa18.py --tuning "$m" --workers 11 >> "results/aos_study/etapa18_tuning_${m}_log.txt" 2>&1
done
python3 -u estudo_aos/run_aos_etapa18.py --selecionar >> results/aos_study/etapa18_selecionar_log.txt 2>&1
python3 -u estudo_aos/run_aos_etapa18.py --validar --workers 11 >> results/aos_study/etapa18_validar_log.txt 2>&1
python3 -u estudo_aos/run_aos_etapa18.py --eval --workers 11 >> results/aos_study/etapa18_eval_log.txt 2>&1
echo "FILA 18 CONCLUIDA"
