#!/usr/bin/env python3
"""Confere a integridade de um CSV de execucoes depois de uma parada abrupta: numero de campos por linha, chaves duplicadas e a ultima linha. Uso: python verifica_csv.py arquivo.csv"""
import csv
import sys
from collections import Counter

caminho = sys.argv[1]
with open(caminho, encoding="utf-8", newline="") as f:
    linhas = list(csv.reader(f))
h = linhas[0]
dados = linhas[1:]
ruins = [i for i, r in enumerate(dados, 1) if len(r) != len(h)]
print(f"{caminho}: {len(dados)} linhas de dados; {len(h)} campos no cabecalho; linhas com numero de campos diferente: {ruins[:5]}")
ib, ic, ii, isd, inv = (h.index(c) for c in ("braco", "config_id", "instancia", "seed", "nivel"))
chaves = Counter((r[ib], r[ic], r[inv], r[ii], r[isd]) for r in dados if len(r) == len(h))
dup = [k for k, v in chaves.items() if v > 1]
print(f"chaves duplicadas: {len(dup)}; ultima linha: {dados[-1][:6] if dados else None}")
