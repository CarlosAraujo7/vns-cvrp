#!/bin/bash
# Para as filas e os runners (padroes com colchetes para nao casar com a propria linha de comando) e relanca a cadeia na ORDEM: 23 -> 21c -> 21b -> 21r (P12) -> 22.
cd "$(dirname "$0")/.." || exit 1
R=results/aos_study
pkill -f "[f]ila_2" ; sleep 1
pkill -f "[r]un_aos_etapa2" ; sleep 3
pkill -9 -f "[r]un_aos_etapa2" ; sleep 2
echo "processos restantes: $(pgrep -f '[r]un_aos_etapa2|[f]ila_2' | wc -l)"
(nohup setsid bash estudo_aos/fila_23.sh >> $R/etapa23_fila_log.txt 2>&1 < /dev/null &)
sleep 2
(nohup setsid bash estudo_aos/fila_21c.sh >> $R/etapa21c_fila_log.txt 2>&1 < /dev/null &)
(nohup setsid bash estudo_aos/fila_21b.sh >> $R/etapa21b_fila_log.txt 2>&1 < /dev/null &)
(nohup setsid bash estudo_aos/fila_21r.sh >> $R/etapa21_fila_log.txt 2>&1 < /dev/null &)
(nohup setsid bash estudo_aos/fila_22.sh >> $R/etapa22_fila_log.txt 2>&1 < /dev/null &)
(nohup setsid bash estudo_aos/fila_24.sh >> $R/etapa24_fila_log.txt 2>&1 < /dev/null &)
sleep 5
pgrep -af "estudo_aos/fila_2" | cut -c1-70
date +%H:%M:%S
