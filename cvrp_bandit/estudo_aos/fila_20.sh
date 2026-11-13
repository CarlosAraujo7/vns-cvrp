#!/bin/bash
# fila do Estudo 20 (generalidade do hibrido `tilt` no ALNS do TSP e no ALNS do CVRP; esforco total de ajuste controlado): regressao exata ->
# para cada problema (TSP primeiro, e mais rapido): ajuste (static65 = continuacao do ES dos Estudos 12/13, RP, tilt) -> selecao -> validacao -> avaliacao.
# RETOMAVEL: cada modo pula o que ja esta no CSV (a busca ES retoma pelo arquivo de estado). Se o WSL reiniciar, rode de novo este script.
# NUNCA dois runners no mesmo CSV. O orcamento e em avaliacoes de distancia (independe do hardware), entao outras cargas so retardam a fila.
# Uso (PowerShell): wsl -e bash -lc "cd /mnt/c/Pesquisas/learnheuristica/cvrp_bandit && (nohup setsid bash estudo_aos/fila_20.sh >> results/aos_study/etapa20_fila_log.txt 2>&1 < /dev/null &)"
cd "$(dirname "$0")/.." || exit 1
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
python3 -u estudo_aos/run_aos_etapa20.py --regressao --workers 11 >> $R/etapa20_regressao_log.txt 2>&1
for P in TSP CVRP; do
  for m in static65 RP tilt; do
    python3 -u estudo_aos/run_aos_etapa20.py --problema $P --tuning $m --workers 11 >> "$R/etapa20_tuning_${m}_${P}_log.txt" 2>&1
  done
  python3 -u estudo_aos/run_aos_etapa20.py --problema $P --selecionar >> "$R/etapa20_selecionar_${P}_log.txt" 2>&1
  python3 -u estudo_aos/run_aos_etapa20.py --problema $P --validar --workers 11 >> "$R/etapa20_validar_${P}_log.txt" 2>&1
  python3 -u estudo_aos/run_aos_etapa20.py --problema $P --eval --workers 11 >> "$R/etapa20_eval_${P}_log.txt" 2>&1
done
echo "FILA 20 CONCLUIDA"
