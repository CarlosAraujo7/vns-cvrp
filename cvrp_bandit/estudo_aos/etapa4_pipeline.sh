#!/bin/bash
# Pipeline da Etapa 4 (diario, Entrada 83): encadeia, na ordem pre-registrada, as etapas
# resumiveis. Cada etapa pode ser relancada sem perder trabalho.
#   1) espera o fim da grade (--tuning-grid)    2) --tuning-static (busca (1+4)-ES)
#   3) --selecionar (grava as configuracoes)     4) --eval74           5) --val
# Roda no WSL:  nohup setsid bash etapa4_pipeline.sh > ../results/aos_study/etapa4_pipeline_log.txt 2>&1 < /dev/null &
cd "$(dirname "$0")" || exit 1
source ../direcao1_local/venv/bin/activate
LOG=../results/aos_study
while pgrep -f "run_aos_etapa4.py --tuning-grid" > /dev/null; do sleep 20; done
echo "$(date) grade terminou"
python3 -u run_aos_etapa4.py --tuning-grid --workers 11 > $LOG/etapa4_grid_log_final.txt 2>&1   # completa o que faltar (resumivel)
echo "$(date) grade verificada"
python3 -u run_aos_etapa4.py --tuning-static --workers 11 > $LOG/etapa4_static_log.txt 2>&1
echo "$(date) busca estatica terminou"
python3 -u run_aos_etapa4.py --selecionar > $LOG/etapa4_selecao_log.txt 2>&1
echo "$(date) selecao gravada"
python3 -u run_aos_etapa4.py --eval74 --workers 11 > $LOG/etapa4_eval74_log.txt 2>&1
echo "$(date) eval74 terminou"
python3 -u run_aos_etapa4.py --val --workers 11 > $LOG/etapa4_val_log.txt 2>&1
echo "$(date) val terminou -- ETAPA4_COMPLETA"
