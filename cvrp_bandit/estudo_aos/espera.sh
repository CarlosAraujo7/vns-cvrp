#!/bin/bash
# Espera ate `max_s` segundos enquanto houver um processo cujo comando contenha `padrao`; depois mostra as ultimas `n` linhas do log. Uso: bash espera.sh padrao max_s log [n]
padrao="$1"; max_s="$2"; log="$3"; n="${4:-15}"
t=0
while pgrep -f "$padrao" > /dev/null && [ "$t" -lt "$max_s" ]; do sleep 10; t=$((t + 10)); done
if pgrep -f "$padrao" > /dev/null; then echo "(ainda rodando apos ${t}s)"; else echo "(processo terminou)"; fi
date +%H:%M:%S
tail -n "$n" "$log"
