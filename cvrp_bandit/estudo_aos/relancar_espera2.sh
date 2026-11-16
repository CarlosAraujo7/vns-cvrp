#!/bin/bash
# Relanca SO as filas que esperam (24, 25 e 21r), sem tocar nas que estao rodando (22). Padroes com colchetes para nao casar com a propria linha de comando.
cd "$(dirname "$0")/.." || exit 1
R=results/aos_study
pkill -f "[b]ash estudo_aos/fila_21r.sh"; pkill -f "[b]ash estudo_aos/fila_24.sh"; pkill -f "[b]ash estudo_aos/fila_25.sh"; sleep 2
(nohup setsid bash estudo_aos/fila_24.sh >> $R/etapa24_fila_log.txt 2>&1 < /dev/null &)
(nohup setsid bash estudo_aos/fila_25.sh >> $R/etapa25_fila_log.txt 2>&1 < /dev/null &)
(nohup setsid bash estudo_aos/fila_21r.sh >> $R/etapa21_fila_log.txt 2>&1 < /dev/null &)
sleep 3
pgrep -af "estudo_aos/fila_2" | cut -c1-60
date +%H:%M:%S
