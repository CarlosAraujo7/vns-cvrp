#!/usr/bin/env python3
"""Gera so as figuras pedidas (nomes das funcoes de gerar_figuras.py na linha de comando); evita regerar todas. Uso: python fig_um.py fig_deslocamentos fig_replica"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gerar_figuras as g  # noqa: E402

for nome in sys.argv[1:]:
    getattr(g, nome)()
    print("ok", nome, flush=True)
