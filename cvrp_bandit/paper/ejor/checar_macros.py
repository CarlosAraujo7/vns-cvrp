#!/usr/bin/env python3
"""
checar_macros.py
================
Controle de qualidade do manuscrito: (i) todo \\num{chave} usado nas secoes
existe em numeros.json (macro ausente aparece em vermelho como ??chave?? no
PDF e, em modo texto, gera erro de LaTeX); (ii) nenhum \\pend{...} restante
quando chamado com --final; (iii) caracteres de controle nos .tex
(corrupcao por escape). Sai com codigo 1 se algo falhar.
Uso: python checar_macros.py [--final]
"""
import glob
import json
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
BS = chr(92)
RE_NUM = re.compile(re.escape(BS + "num{") + r"([^}]*)\}")
RE_PEND = re.compile(re.escape(BS + "pend{"))


def main():
    final = "--final" in sys.argv
    n = json.loads((AQUI / "numeros.json").read_text(encoding="utf-8"))
    problemas = 0
    usadas = set()
    for f in sorted(glob.glob(str(AQUI / "sec_*.tex"))) + [str(AQUI / "main.tex")]:
        b = Path(f).read_bytes()
        ctrl = [i for i, c in enumerate(b) if c < 32 and c not in (9, 10, 13)]
        t = b.decode("utf-8")
        ks = RE_NUM.findall(t)
        usadas |= set(ks)
        ausentes = sorted({k for k in ks if k not in n})
        npend = len(RE_PEND.findall(t))
        print(f"{Path(f).name:20s} usos de num: {len(ks):4d}  ausentes: {len(ausentes)}  pend: {npend}  controle: {len(ctrl)}")
        if ausentes:
            problemas += 1
            print("   AUSENTES:", ausentes)
        if ctrl:
            problemas += 1
        if final and npend:
            problemas += 1
    print(f"{len(usadas)} chaves distintas usadas de {len(n)} geradas")
    # requisitos do guia de autores do EJOR (conferido em 09/10/2026): highlights 3-5 itens de ate 85 caracteres;
    # 1-5 palavras-chave; abstract de ate 250 palavras
    hl = [l.rstrip("\n") for l in (AQUI / "highlights.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"highlights: {len(hl)} itens, maior com {max(len(l) for l in hl)} caracteres")
    if not 3 <= len(hl) <= 5 or max(len(l) for l in hl) > 85:
        problemas += 1
        print("   PROBLEMA: highlights fora do limite (3-5 itens, 85 caracteres)")
    main = (AQUI / "main.tex").read_text(encoding="utf-8")
    kw = re.search(re.escape(BS + "begin{keyword}") + r"(.*?)" + re.escape(BS + "end{keyword}"), main, re.S)
    if kw:
        nkw = len([k for k in re.split(re.escape(BS + "sep"), kw.group(1)) if k.strip()])
        print(f"palavras-chave: {nkw}")
        if nkw > 5:
            problemas += 1
            print("   PROBLEMA: mais de 5 palavras-chave")
    ab = re.search(re.escape(BS + "begin{abstract}") + r"(.*?)" + re.escape(BS + "end{abstract}"), main, re.S)
    if ab:
        if re.search(r"(?<!" + re.escape(BS) + r")%", ab.group(1)):
            problemas += 1
            print("   PROBLEMA: '%' sem barra no resumo (vira comentario e corta o texto)")
        txt = RE_NUM.sub("0.0", ab.group(1))
        txt = re.sub(re.escape(BS) + r"[a-zA-Z]+\{?", " ", txt)
        palavras = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'.\-%]*", txt))
        print(f"abstract: ~{palavras} palavras (limite 250)")
        if palavras > 250:
            problemas += 1
            print("   PROBLEMA: abstract acima de 250 palavras")
    print("CHECAGEM:", "OK" if not problemas else f"{problemas} PROBLEMA(S)")
    sys.exit(1 if problemas else 0)


if __name__ == "__main__":
    main()
