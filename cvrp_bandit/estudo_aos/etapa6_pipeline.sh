#!/bin/bash
# Pipeline da Etapa 6 (pre-registro: diario, Entrada 91, commit 4b8fa86). Espera a Etapa 5 terminar (a CPU esta
# saturada por ela), roda as 74 instancias (E1 e E2) e depois a validacao sintetica exploratoria. Resumivel: cada
# runner pula as chaves ja gravadas; basta relancar este script.  Roda no WSL (venv de direcao1_local).
cd "$(dirname "$0")" || exit 1
RES=../results/aos_study
PY=../direcao1_local/venv/bin/python3
until grep -q "^concluido" $RES/etapa5_full_log.txt 2>/dev/null; do sleep 30; done
echo "Etapa 5 concluida em $(date); iniciando Etapa 6" >> $RES/etapa6_pipeline_log.txt
$PY -u run_aos_etapa6.py --full --workers 11 >> $RES/etapa6_full_log.txt 2>&1
echo "Etapa 6 (74 instancias) concluida em $(date)" >> $RES/etapa6_pipeline_log.txt
$PY -u run_aos_etapa6.py --val --workers 11 >> $RES/etapa6_val_log.txt 2>&1
echo "ETAPA6_PIPELINE_COMPLETO $(date)" >> $RES/etapa6_pipeline_log.txt
