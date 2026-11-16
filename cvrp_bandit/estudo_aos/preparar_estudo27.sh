#!/bin/bash
# Prepara as instancias do Estudo 27 (diario, Entradas 145 e 147): as variantes de capacidade de SYN_REP (SYN_CAP, 120 instancias, referencia HGS 2 x 0,2 s por cliente) e as solucoes iniciais. ESPERA o fim
# de preparar_estudo26.sh (precisa de SYN_REP; evita a concorrencia do HGS por processador). RETOMAVEL.
cd "$(dirname "$0")/.." || exit 1
sleep 60
while pgrep -f "preparar_estudo26\.sh" > /dev/null; do sleep 30; done
source direcao1_local/venv/bin/activate
export HGS_THREADS=3
echo "[$(date +%H:%M:%S)] SYN_CAP"
python3 -u estudo_aos/gerar_variantes_capacidade.py
echo "[$(date +%H:%M:%S)] solucoes iniciais"
python3 -u estudo_aos/gerar_solucoes_iniciais.py
echo "[$(date +%H:%M:%S)] PREPARO DO ESTUDO 27 CONCLUIDO"
