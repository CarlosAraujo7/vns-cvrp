#!/bin/bash
# Compila o HGS-CVRP com a selecao adaptativa do tipo de movimento (Estudo 7; diario, Entrada 97).
# Origem: copia modificada do HGS-CVRP (T. Vidal, MIT; src/hgs_aos/LICENSE) em src/hgs_aos/Program. A copia original
# e o binario de referencia (external/HGS-CVRP/hgs) NAO sao alterados. Roda no WSL (g++ >= 11).
# Uso: bash direcao1_local/build_hgs_aos.sh   ->   external/HGS-CVRP-aos/hgs_aos
set -e
cd "$(dirname "$0")/.."
mkdir -p external/HGS-CVRP-aos
g++ -std=c++17 -O3 -DNDEBUG src/hgs_aos/Program/*.cpp -o external/HGS-CVRP-aos/hgs_aos
echo "compilado: external/HGS-CVRP-aos/hgs_aos"
