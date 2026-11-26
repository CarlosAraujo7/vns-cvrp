#!/bin/bash
# Para TODAS as filas (a em execucao e as que esperam) e os runners com o pool de processos (forkserver), sem relancar nada. Os padroes tem colchetes para nao casar com a propria linha
# de comando. Depois, relancar com retomar.sh (que confere a integridade dos CSV). Uso (PowerShell): wsl -e bash -lc "bash /mnt/c/Pesquisas/learnheuristica/cvrp_bandit/estudo_aos/parar_filas.sh"
cd "$(dirname "$0")/.." || exit 1
date +%d/%m\ %H:%M:%S
echo "etapa22_eval_longabc.csv (bytes) antes: $(wc -c < results/aos_study/etapa22_eval_longabc.csv 2>/dev/null)"
pkill -f "[f]ila_[23]"; sleep 1
pkill -f "[r]un_aos_etapa[23]"; sleep 3
pkill -9 -f "[m]ultiprocessing.forkserver"; pkill -9 -f "[r]un_aos_etapa[23]"; sleep 3
echo "filas restantes: $(pgrep -f '[f]ila_[23]' | wc -l); runners restantes: $(pgrep -f '[r]un_aos_etapa[23]' | wc -l); pool restante: $(pgrep -f '[m]ultiprocessing.forkserver' | wc -l)"
ps -eo pid,etime,pcpu,cmd | grep -E '[v]env/bin/python3|[f]ila_[23]|[r]un_aos' | cut -c1-120
echo "etapa22_eval_longabc.csv (bytes) depois: $(wc -c < results/aos_study/etapa22_eval_longabc.csv 2>/dev/null)"
uptime
date +%d/%m\ %H:%M:%S
