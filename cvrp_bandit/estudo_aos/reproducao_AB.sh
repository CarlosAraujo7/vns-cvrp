#!/bin/bash
# Reexecuta os 5 experimentos de reproducao (GVNS e 4 variantes do HG-UCB) nos conjuntos A e B com a
# solucao inicial FIXADA em disco (diario, Entrada 88). Os CSVs originais foram arquivados em
# results/aos_study/superseded_init_nao_det/reproducao/. Roda no WSL (venv de direcao1_local).
cd "$(dirname "$0")/../tests" || exit 1
source ../direcao1_local/venv/bin/activate
for S in A B; do
  python3 -u run_gvns_baseline.py $S
  python3 -u run_bandit_vns_validation.py $S fixo
  python3 -u run_bandit_vns_validation.py $S progressivo
  python3 -u run_bandit_vns_validation.py $S progressivo normalizado_alpha5.0
  python3 -u run_bandit_vns_validation.py $S progressivo normalizado_alpha0.5
done
echo REPRODUCAO_AB_COMPLETA
