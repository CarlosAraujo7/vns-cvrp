#!/bin/bash
# Retoma as filas dos Estudos 21 e 22 depois de uma parada do WSL (o app fechou): confere a integridade dos CSV de execucoes e relanca as quatro filas, que sao retomaveis e esperam umas
# pelas outras. Uso (PowerShell): wsl -e bash -lc "bash /mnt/c/Pesquisas/learnheuristica/cvrp_bandit/estudo_aos/retomar.sh"
cd "$(dirname "$0")/.." || exit 1
PY=direcao1_local/venv/bin/python3
R=results/aos_study
if pgrep -f "fila_2" > /dev/null; then echo "filas ja rodando:"; pgrep -af "fila_2" | cut -c1-70; exit 0; fi
for f in $R/etapa21_ajuste_static_P12.csv $R/etapa21_eval_P21.csv $R/etapa21_ajuste_RP_P12.csv $R/etapa21_validacao_P12.csv $R/etapa21_eval_P12.csv \
         $R/etapa21b_ajuste_static_P21.csv $R/etapa21c_eval_P21.csv $R/etapa23_eval_p21.csv; do
  if [ -f "$f" ]; then $PY estudo_aos/verifica_csv.py "$f"; fi
done
echo "estado do ES de P12:"; cat $R/etapa21_ajuste_static_P12_estado.json 2>/dev/null; echo
(nohup setsid bash estudo_aos/fila_21r.sh >> $R/etapa21_fila_log.txt 2>&1 < /dev/null &)
sleep 2
(nohup setsid bash estudo_aos/fila_21b.sh >> $R/etapa21b_fila_log.txt 2>&1 < /dev/null &)
(nohup setsid bash estudo_aos/fila_21c.sh >> $R/etapa21c_fila_log.txt 2>&1 < /dev/null &)
(nohup setsid bash estudo_aos/fila_23.sh >> $R/etapa23_fila_log.txt 2>&1 < /dev/null &)
(nohup setsid bash estudo_aos/fila_22.sh >> $R/etapa22_fila_log.txt 2>&1 < /dev/null &)
(nohup setsid bash estudo_aos/fila_24.sh >> $R/etapa24_fila_log.txt 2>&1 < /dev/null &)
(nohup setsid bash estudo_aos/fila_25.sh >> $R/etapa25_fila_log.txt 2>&1 < /dev/null &)
sleep 5
pgrep -af "fila_2" | cut -c1-70
date +%H:%M:%S
