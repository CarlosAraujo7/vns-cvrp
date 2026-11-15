#!/bin/bash
# Fila do Estudo 22 (deslocamentos de implantacao: classe de instancias, orcamento, regra de aceitacao; diario, Entrada 138). RETOMAVEL: cada modo pula o que ja esta no CSV e o ES
# retoma pelo arquivo de estado. Ordem: regressao exata -> calibracao do desvio do RRT -> ajuste de S33r -> selecao -> validacao -> avaliacoes (cls, short, acc, longabc, longx).
# NUNCA dois runners no mesmo CSV; NAO lancar enquanto a fila do Estudo 21 estiver rodando.
# Uso (PowerShell): wsl -e bash -lc "cd /mnt/c/Pesquisas/learnheuristica/cvrp_bandit && (nohup setsid bash estudo_aos/fila_22.sh >> results/aos_study/etapa22_fila_log.txt 2>&1 < /dev/null &)"
cd "$(dirname "$0")/.." || exit 1
source direcao1_local/venv/bin/activate
set -e
R=results/aos_study
echo "[$(date +%H:%M:%S)] FILA 22 INICIADA"
python3 -u estudo_aos/run_aos_etapa22.py --regressao --workers 11 >> "$R/etapa22_regressao_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] calibracao do RRT"
python3 -u estudo_aos/run_aos_etapa22.py --calibrar --workers 11 >> "$R/etapa22_calibrar_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] ajuste de S33r"
python3 -u estudo_aos/run_aos_etapa22.py --ajuste --workers 11 >> "$R/etapa22_ajuste_log.txt" 2>&1
python3 -u estudo_aos/run_aos_etapa22.py --selecionar >> "$R/etapa22_selecionar_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] validacao"
python3 -u estudo_aos/run_aos_etapa22.py --validar --workers 11 >> "$R/etapa22_validar_log.txt" 2>&1
for reg in cls short acc longabc longx; do
  echo "[$(date +%H:%M:%S)] avaliacao $reg"
  python3 -u estudo_aos/run_aos_etapa22.py --eval $reg --workers 11 >> "$R/etapa22_eval_${reg}_log.txt" 2>&1
done
echo "FILA 22 CONCLUIDA"
