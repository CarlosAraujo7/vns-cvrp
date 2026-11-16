#!/bin/bash
# Reinicia SO as filas que ESPERAM (26, 27, 28, 24, 25, 21r), sem tocar na fila que esta rodando (22): serve para mudar a cadeia depois de editar os scripts das filas em espera (bash le o script aos poucos:
# NUNCA editar um script de fila enquanto ele roda). Uso (PowerShell): wsl -e bash /mnt/c/Pesquisas/learnheuristica/cvrp_bandit/estudo_aos/relancar_esperas.sh parar | lancar
# parar: mata as filas em espera (padroes com colchetes). lancar: relanca as que nao estao rodando nem concluidas, na ordem 26 -> 24 -> 25 -> 21r.
cd "$(dirname "$0")/.." || exit 1
R=results/aos_study
if [ "$1" = "parar" ]; then
  pkill -f "[b]ash estudo_aos/fila_26.sh"; pkill -f "[b]ash estudo_aos/fila_27.sh"; pkill -f "[b]ash estudo_aos/fila_28.sh"; pkill -f "[b]ash estudo_aos/fila_24.sh"; pkill -f "[b]ash estudo_aos/fila_25.sh"; pkill -f "[b]ash estudo_aos/fila_21r.sh"
  sleep 1
  pgrep -af "estudo_aos/fila_2" | cut -c1-70
  exit 0
fi
lanca() {   # script, log, marca de conclusao no log
  if pgrep -f "[b]ash estudo_aos/$1" > /dev/null; then echo "$1: ja rodando"; return; fi
  if grep -q "$3" "$R/$2" 2>/dev/null; then echo "$1: concluida"; return; fi
  (nohup setsid bash estudo_aos/$1 >> "$R/$2" 2>&1 < /dev/null &)
  echo "$1: lancada"
  sleep 1
}
lanca fila_26.sh etapa26_fila_log.txt "FILA 26 CONCLUIDA"
lanca fila_27.sh etapa27_fila_log.txt "FILA 27 CONCLUIDA"
lanca fila_28.sh etapa28_fila_log.txt "FILA 28 CONCLUIDA"
lanca fila_24.sh etapa24_fila_log.txt "FILA 24 CONCLUIDA"
lanca fila_25.sh etapa25_fila_log.txt "FILA 25 CONCLUIDA"
lanca fila_21r.sh etapa21_fila_log.txt "FILA 21r CONCLUIDA"
sleep 3
pgrep -af "estudo_aos/fila_2" | cut -c1-70
date +%H:%M:%S
