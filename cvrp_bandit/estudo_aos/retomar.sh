#!/bin/bash
# Retoma a cadeia de filas dos Estudos 21 a 25 depois de uma parada do WSL (o app fechou): confere a integridade dos CSV de execucoes e relanca SO as filas que ainda nao terminaram (a marca
# "FILA nn CONCLUIDA" no log de cada fila), na ordem 23 -> 21c -> 21b -> 22 -> 26 -> 24 -> 25 -> 21r; elas sao retomaveis e esperam umas pelas outras (pgrep). Nao faz nada se alguma fila ja esta rodando.
# (Se o preparo das instancias dos Estudos 26 e 27 foi interrompido: bash estudo_aos/preparar_estudo26.sh e preparar_estudo27.sh, retomaveis, ANTES de relancar.)
# Uso (PowerShell): wsl -e bash /mnt/c/Pesquisas/learnheuristica/cvrp_bandit/estudo_aos/retomar.sh
# Para parar tudo antes (reordenar): parar_filas.sh. NUNCA dois runners no mesmo CSV.
cd "$(dirname "$0")/.." || exit 1
PY=direcao1_local/venv/bin/python3
R=results/aos_study
if pgrep -f "fila_2" > /dev/null; then echo "filas ja rodando:"; pgrep -af "fila_2" | cut -c1-70; exit 0; fi
for f in $R/etapa21_ajuste_static_P12.csv $R/etapa21_ajuste_RP_P12.csv $R/etapa21_validacao_P12.csv $R/etapa21_eval_P12.csv $R/etapa21_eval_P21.csv \
         $R/etapa21b_ajuste_static_P21.csv $R/etapa21c_eval_P21.csv $R/etapa23_eval_p21.csv \
         $R/etapa22_eval_longabc.csv $R/etapa22_eval_longx.csv $R/etapa26_eval.csv $R/etapa27_eval.csv $R/etapa28_ajuste_static.csv $R/etapa28_validacao.csv $R/etapa28_eval.csv $R/etapa24_eval_P21.csv $R/etapa25_eval_acc.csv $R/etapa25_eval_short.csv \
         $R/etapa29_ajuste_RP.csv $R/etapa29_ajuste_HG.csv $R/etapa29_validacao.csv $R/etapa29_eval.csv \
         $R/etapa30_calibracao_q.csv $R/etapa30_validacao.csv $R/etapa30_eval.csv; do
  if [ -f "$f" ]; then $PY estudo_aos/verifica_csv.py "$f" || echo "(a verificacao de $f falhou)"; fi
done
echo "estado do ES de P12:"; cat $R/etapa21_ajuste_static_P12_estado.json 2>/dev/null; echo
lanca() {   # script, log, marca de conclusao no log
  if grep -q "$3" "$R/$2" 2>/dev/null; then echo "$1: concluida (marca '$3' no log)"; return; fi
  (nohup setsid bash estudo_aos/$1 >> "$R/$2" 2>&1 < /dev/null &)
  echo "$1: lancada"
  sleep 1
}
lanca fila_23.sh etapa23_fila_log.txt "FILA 23 CONCLUIDA"
lanca fila_21c.sh etapa21c_fila_log.txt "FILA 21c CONCLUIDA"
lanca fila_21b.sh etapa21b_fila_log.txt "FILA 21b CONCLUIDA"
lanca fila_22.sh etapa22_fila_log.txt "FILA 22 CONCLUIDA"
lanca fila_26.sh etapa26_fila_log.txt "FILA 26 CONCLUIDA"
lanca fila_27.sh etapa27_fila_log.txt "FILA 27 CONCLUIDA"
lanca fila_28.sh etapa28_fila_log.txt "FILA 28 CONCLUIDA"
lanca fila_24.sh etapa24_fila_log.txt "FILA 24 CONCLUIDA"
lanca fila_25.sh etapa25_fila_log.txt "FILA 25 CONCLUIDA"
lanca fila_21r.sh etapa21_fila_log.txt "FILA 21r CONCLUIDA"
lanca fila_29.sh etapa29_fila_log.txt "FILA 29 CONCLUIDA"
lanca fila_30.sh etapa30_fila_log.txt "FILA 30 CONCLUIDA"
sleep 5
pgrep -af "estudo_aos/fila_2" | cut -c1-70
date +%H:%M:%S
