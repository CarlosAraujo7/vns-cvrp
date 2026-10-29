#!/usr/bin/env python3
"""
gerar_instancias_tw.py
======================
Estudo 14 (CVRPTW): gera o conjunto SINTETICO DE AJUSTE `instances/SYN_TUNE_W` (24 instancias de 100 clientes, 4 de cada familia de
Solomon: C1, C2, R1, R2, RC1, RC2), disjunto das 56 instancias de Solomon usadas na avaliacao, e calcula a referencia de cada uma com o
PyVRP (melhor de duas execucoes independentes de `--tempo` s cada, sementes 1 e 2).

Geracao (parametros das familias de Solomon: capacidade, horizonte, tempo de servico, tipo de distribuicao dos clientes e largura
das janelas): coordenadas inteiras em [0,100]^2 (R uniforme; C em 10 aglomerados; RC metade e metade); demanda U{1..40} (C: {10,20,30});
as janelas sao construidas ao redor dos instantes de chegada de uma rota de varredura angular viavel, o que garante instancia viavel;
a largura e estreita nas familias do tipo 1 e larga nas do tipo 2. Saida no formato de Solomon (lido por tw_inst.ler_solomon).

Uso: python estudo_aos/gerar_instancias_tw.py [--tempo 20] [--workers 8]
"""
import argparse
import csv
import math
import multiprocessing as mp
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from tw_inst import ler_solomon  # noqa: E402

SEMENTE = 20261030
FAMILIAS = {  # tipo: (capacidade, horizonte, servico, deposito, largura_min, largura_max)
    "C1": (200, 1236, 90, (40, 50), 40, 140),
    "C2": (700, 3390, 90, (40, 50), 300, 700),
    "R1": (200, 230, 10, (35, 35), 10, 60),
    "R2": (1000, 1000, 10, (35, 35), 100, 400),
    "RC1": (200, 240, 10, (40, 25), 10, 60),
    "RC2": (1000, 960, 10, (40, 25), 100, 400),
}


def trunc10(dx, dy):
    return int(math.floor(10 * math.sqrt(dx * dx + dy * dy) + 1e-9))


def gera(nome, tipo, n, seed):
    rng = random.Random(seed)
    Q, H, S, dep, wmin, wmax = FAMILIAS[tipo]
    pts = []
    if tipo.startswith("R") and not tipo.startswith("RC"):
        pts = [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
    else:
        nclu = 10 if tipo.startswith("C") else 5
        centros = [(rng.randint(10, 90), rng.randint(10, 90)) for _ in range(nclu)]
        nc = n if tipo.startswith("C") else n // 2
        for k in range(nc):
            cx, cy = centros[k % nclu]
            pts.append((min(100, max(0, int(round(rng.gauss(cx, 5))))), min(100, max(0, int(round(rng.gauss(cy, 5)))))))
        pts += [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n - nc)]
        rng.shuffle(pts)
    dem = [rng.choice([10, 20, 30]) if tipo.startswith("C") else rng.randint(1, 40) for _ in range(n)]
    # rota de varredura angular viavel (so capacidade e horizonte) para ancorar as janelas
    ang0 = rng.random() * 2 * math.pi
    ordem = sorted(range(n), key=lambda i: (math.atan2(pts[i][1] - dep[1], pts[i][0] - dep[0]) - ang0) % (2 * math.pi))
    rotas, atual, carga, t, prev = [], [], 0, 0, None
    chegada = {}
    for i in ordem:
        d_prev = trunc10(pts[i][0] - (dep[0] if prev is None else pts[prev][0]), pts[i][1] - (dep[1] if prev is None else pts[prev][1])) / 10
        d_back = trunc10(pts[i][0] - dep[0], pts[i][1] - dep[1]) / 10
        if atual and (carga + dem[i] > Q or t + d_prev + S + d_back > 0.95 * H):
            rotas.append(atual)
            atual, carga, t, prev = [], 0, 0, None
            d_prev = d_back
        t += d_prev
        chegada[i] = t
        t += S
        carga += dem[i]
        atual.append(i)
        prev = i
    rotas.append(atual)
    linhas = []
    for i in range(n):
        a = chegada[i]
        w = rng.randint(wmin, wmax)
        e = max(0, int(a - rng.randint(0, w)))
        l = e + w
        d_back = trunc10(pts[i][0] - dep[0], pts[i][1] - dep[1]) / 10
        l = int(min(l, H - d_back - S))
        l = max(l, int(math.ceil(a)))
        e = min(e, l)
        linhas.append((i + 1, pts[i][0], pts[i][1], dem[i], e, l, S))
    cab = [nome, "", "VEHICLE", "NUMBER     CAPACITY", f"  {max(25, len(rotas) + 5)}         {Q}", "", "CUSTOMER",
           "CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME", " ",
           f"    0      {dep[0]}         {dep[1]}          0          0       {H}          0   "]
    cab += [f"    {c}      {x}         {y}         {d}        {e}       {l}         {s}   " for c, x, y, d, e, l, s in linhas]
    return "\n".join(cab) + "\n"


def _ref(args):
    caminho, tempo, seed = args
    from solver_pyvrp_tw import resolver_pyvrp
    inst = ler_solomon(caminho)
    rotas = resolver_pyvrp(inst, tempo, seed)
    ok, _ = inst.is_feasible(rotas)
    return Path(caminho).stem, seed, inst.solution_cost(rotas) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tempo", type=float, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dest = RAIZ / "instances" / "SYN_TUNE_W"
    dest.mkdir(parents=True, exist_ok=True)
    nomes = []
    k = 0
    for rep in range(4):
        for tipo in FAMILIAS:
            nome = f"W-synt{k:02d}-{tipo}"
            (dest / f"{nome}.txt").write_text(gera(nome, tipo, 100, SEMENTE + k))
            nomes.append((nome, tipo))
            k += 1
    tarefas = [(str(dest / f"{n}.txt"), a.tempo, sd) for n, _ in nomes for sd in (1, 2)]
    with mp.Pool(a.workers) as pool:
        res = pool.map(_ref, tarefas)
    por = {}
    for nome, seed, c in res:
        por.setdefault(nome, {})[seed] = c
    with open(dest / "ref.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instancia", "n", "tipo", "ref", "run1", "run2", "concordam"])
        for nome, tipo in nomes:
            r1, r2 = por[nome][1], por[nome][2]
            assert r1 is not None and r2 is not None, nome
            w.writerow([nome, 100, tipo, min(r1, r2), r1, r2, r1 == r2])
    print(f"{len(nomes)} instancias; referencias concordantes: {sum(por[n][1] == por[n][2] for n, _ in nomes)}")


if __name__ == "__main__":
    main()
