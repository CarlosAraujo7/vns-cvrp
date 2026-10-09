#!/bin/bash
# Compila main.tex e supplement.tex com o Tectonic (binario Linux em ferramentas/, copiado para /tmp porque o WSL apaga /tmp ao reiniciar).
cd "$(dirname "$0")" || exit 1
[ -x /tmp/tectonic ] || { cp ../../ferramentas/tectonic /tmp/tectonic && chmod +x /tmp/tectonic; }
for d in main supplement; do
  /tmp/tectonic -X compile $d.tex --outdir build --keep-logs 2>&1 | grep -E "^error|^!|Output written" | head -5
done
