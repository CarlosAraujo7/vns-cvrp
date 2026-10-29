#!/usr/bin/env python3
"""
verifica_solomon_bks.py
=======================
Estudo 14: valida o leitor de instancias de Solomon, a convencao de custo (distancia e tempo truncados a uma casa decimal, guardados em
decimos) e a simulacao de viabilidade (capacidade e janelas de tempo) contra as 56 melhores solucoes conhecidas (BKS) distribuidas
pela CVRPLIB: o custo recomputado por nossa funcao deve igualar o BKS publicado e a solucao deve ser viavel.
Saida: results/aos_study/etapa14_solomon_bks_check.csv
"""
import csv
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from tw_inst import ler_solomon, ler_bks, construcao_inicial  # noqa: E402

# BKS publicados na pagina VRPTW da CVRPLIB (conferidos em 29/10/2026), em unidades de distancia
PUBLICADO = {
    "C101": 827.3, "C102": 827.3, "C103": 826.3, "C104": 822.9, "C105": 827.3, "C106": 827.3, "C107": 827.3, "C108": 827.3, "C109": 827.3,
    "C201": 589.1, "C202": 589.1, "C203": 588.7, "C204": 588.1, "C205": 586.4, "C206": 586.0, "C207": 585.8, "C208": 585.8,
    "R101": 1637.7, "R102": 1466.6, "R103": 1208.7, "R104": 971.5, "R105": 1355.3, "R106": 1234.6, "R107": 1064.6, "R108": 932.1,
    "R109": 1146.9, "R110": 1068.0, "R111": 1048.7, "R112": 948.6,
    "R201": 1143.2, "R202": 1029.6, "R203": 870.8, "R204": 731.3, "R205": 949.8, "R206": 875.9, "R207": 794.0, "R208": 701.0,
    "R209": 854.8, "R210": 900.5, "R211": 746.7,
    "RC101": 1619.8, "RC102": 1457.4, "RC103": 1258.0, "RC104": 1132.3, "RC105": 1513.7, "RC106": 1372.7, "RC107": 1207.8, "RC108": 1114.2,
    "RC201": 1261.8, "RC202": 1092.3, "RC203": 923.7, "RC204": 783.5, "RC205": 1154.0, "RC206": 1051.1, "RC207": 962.9, "RC208": 776.1,
}


def main():
    raiz = RAIZ / "instances" / "SOLOMON"
    linhas, ok = [], 0
    for nome, pub in PUBLICADO.items():
        inst = ler_solomon(raiz / f"{nome}.txt")
        rotas = ler_bks(raiz / "bks" / f"{nome}.sol")
        viavel, msg = inst.is_feasible(rotas)
        custo = inst.solution_cost(rotas)
        igual = abs(custo - round(10 * pub)) == 0
        ini = construcao_inicial(inst)
        linhas.append({"instancia": nome, "n": inst.n_customers, "bks_publicado": pub, "custo_recomputado": custo / 10, "viavel": viavel,
                       "igual": igual, "rotas_bks": len(rotas), "custo_inicial": inst.solution_cost(ini) / 10,
                       "rotas_inicial": len(ini), "inicial_viavel": inst.is_feasible(ini)[0]})
        ok += viavel and igual
    saida = RAIZ / "results" / "aos_study" / "etapa14_solomon_bks_check.csv"
    with open(saida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    print(f"BKS viaveis e com custo igual ao publicado: {ok} de {len(PUBLICADO)}; inicial viavel: {sum(r['inicial_viavel'] for r in linhas)}")
    sys.exit(0 if ok == len(PUBLICADO) else 1)


if __name__ == "__main__":
    main()
