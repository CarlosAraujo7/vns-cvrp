#!/bin/bash
# Prepara as instancias NOVAS do Estudo 26 (diario, Entradas 145 e 146): (1) SYN_REP, 60 instancias sinteticas (20 de 100, 20 de 200 e 20 de 300 clientes; semente 20261601; referencia HGS 2 x 0,2 s por
# cliente) e (2) X_FRESH, 38 instancias X de 200 a 401 clientes baixadas do repositorio publico vidalt/HGS-CVRP (referencia HGS 2 x 0,4 s por cliente); (3) as solucoes iniciais (Clarke-Wright + reparo) de
# todas as instancias novas. Usa poucas linhas (HGS_THREADS) para nao atrapalhar a fila de execucoes. RETOMAVEL (nao regenera o que ja existe). Nao toca nos dados de nenhum estudo anterior.
cd "$(dirname "$0")/.." || exit 1
source direcao1_local/venv/bin/activate
export HGS_THREADS=3
echo "[$(date +%H:%M:%S)] SYN_REP"
python3 -u estudo_aos/gerar_instancias_sinteticas.py SYN_REP
echo "[$(date +%H:%M:%S)] X_FRESH"
python3 -u estudo_aos/preparar_x_fresh.py
echo "[$(date +%H:%M:%S)] solucoes iniciais"
python3 -u estudo_aos/gerar_solucoes_iniciais.py
echo "[$(date +%H:%M:%S)] PREPARO DO ESTUDO 26 CONCLUIDO"
