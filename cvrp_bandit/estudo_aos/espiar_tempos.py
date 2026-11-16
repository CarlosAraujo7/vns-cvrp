#!/usr/bin/env python3
"""Olhada nos tempos e iteracoes por braco de um CSV de avaliacao parcial (diagnostico de desempenho da fila). Uso: python espiar_tempos.py arquivo.csv"""
import sys

import pandas as pd

d = pd.read_csv(sys.argv[1])
print(len(d), "linhas")
print(d.groupby("braco").agg(n=("seed", "size"), tempo_medio=("tempo_s", "mean"), tempo_max=("tempo_s", "max"), iteracoes=("iteracoes", "mean"),
                              n_clientes=("n_customers", "mean"), gap=("gap_pct", "mean")).round(1).to_string())
