#!/usr/bin/env bash
for tam in 10 20 50 100; do
    echo "=== iniciando graph_size=$tam: $(date) ==="
    python3 kool_finetune_runner.py --graph_size "$tam" > "log_finetune_${tam}.txt" 2>&1
    echo "=== terminou graph_size=$tam: $(date) (exit code $?) ==="
done
echo "=== TUDO CONCLUIDO: $(date) ==="
