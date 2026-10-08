#!/usr/bin/env python3
"""
solver_lkh3.py
==============
Wrapper para o binário LKH3 (compilado em external/LKH-3.0.7/LKH -- fonte
e licença confirmados em diario_pesquisa.md, Entradas 32/34) como
comparador externo na Direção 1.

DUAS descobertas empíricas importantes (validadas em A-n32-k5 e CMT1
antes de escrever este wrapper -- ver diario_pesquisa.md, Entrada 36):

1) LKH3 não tem um modo nativo "pelo menos K rotas, mais se for
   vantajoso" para CVRP. O parâmetro VEHICLES, com MTSP_MIN_SIZE=1 (que
   exige >=1 cliente por "caixeiro"), funciona como K EXATO -- documentado
   em LKH-3_PARAMETERS.pdf ("Specifies the minimum number of cities each
   salesman must visit"). Para vehicles_exact=False (conjuntos X/CMT),
   este wrapper varre VEHICLES de n_vehicles_ref até
   n_vehicles_ref+vehicle_slack, uma execução por valor, e devolve a
   melhor solução viável entre as tentativas -- nunca abaixo do piso,
   permite mais se for vantajoso. Mesma constante vehicle_slack=5 usada
   em solver_ortools.py, por simetria entre os dois comparadores.

2) LKH3 segue a convenção TSPLIB estrita para EDGE_WEIGHT_TYPE=EUC_2D:
   SEMPRE arredonda a distância internamente, mesmo para instâncias CMT
   (que no nosso projeto usam distância float, sem arredondar -- ver
   instance.py). Confirmado empiricamente em CMT1: LKH3 reportou
   "Cost = 521" (arredondado, sem sentido físico no nosso referencial),
   mas as MESMAS rotas, recalculadas com `inst.solution_cost()` (distância
   float correta), deram exatamente 524.611 -- o ótimo verdadeiro. Por
   isso este wrapper NUNCA usa o custo que o LKH3 imprime/retorna --
   sempre recalcula via `inst.solution_cost(routes)` depois de converter
   as rotas, a única fonte de verdade de custo usada em todo o resto do
   projeto.

Formato de leitura da solução: usa SINTEF_SOLUTION_FILE (não
OUTPUT_TOUR_FILE), porque seu formato ("Route N : clientes...") já usa a
MESMA convenção de numeração de cliente (1..dimension-1, pulando o
depósito) que os .sol de referência da CVRPLIB -- confirmado testando
diretamente contra A-n32-k5 (bateu 784, o ótimo exato). Isso permite
reaproveitar `instance._sol_routes_to_vrp_ids`, já validado, em vez de
reimplementar a conversão de ids.

Uso:
    from solver_lkh3 import solve_lkh3
    routes, wall_time = solve_lkh3(inst, time_limit_s=10.0, seed=1)
"""
import subprocess
import tempfile
import time
from pathlib import Path

from instance import _sol_routes_to_vrp_ids

LKH_BIN = Path(__file__).resolve().parent.parent / "external" / "LKH-3.0.7" / "LKH"
VEHICLE_SLACK = 5  # mesma constante de solver_ortools.py, por simetria


def _ler_sintef(path, depot_id):
    """Lê o SINTEF_SOLUTION_FILE do LKH3 ('Route N : c1 c2 ...', clientes
    numerados 1..dimension-1 pulando o depósito -- mesma convenção dos
    .sol de referência) e devolve rotas já convertidas para ids reais do
    .vrp via o mesmo helper usado em instance.py."""
    sintef_routes = []
    for linha in path.read_text().splitlines():
        linha = linha.strip()
        if not linha.lower().startswith("route"):
            continue
        after_colon = linha.split(":", 1)[1].strip()
        sintef_routes.append([int(x) for x in after_colon.split()])
    return _sol_routes_to_vrp_ids(sintef_routes, depot_id)


def _rodar_uma_vez(inst, vehicles, time_limit_s, seed, workdir):
    """Roda LKH3 uma vez com VEHICLES fixo (semântica EXATA -- ver
    docstring do módulo). Retorna (routes_ou_None, wall_time_s)."""
    tag = f"{inst.name}_{vehicles}v_seed{seed}"
    sintef_path = workdir / f"{tag}.sintef"
    par_path = workdir / f"{tag}.par"
    par_path.write_text(
        f"PROBLEM_FILE = {inst.source_vrp}\n"
        f"VEHICLES = {vehicles}\n"
        f"MTSP_MIN_SIZE = 1\n"
        f"TIME_LIMIT = {time_limit_s}\n"
        f"SEED = {seed if seed else 1}\n"
        f"RUNS = 1\n"
        f"TRACE_LEVEL = 0\n"
        f"SINTEF_SOLUTION_FILE = {sintef_path}\n"
    )
    t0 = time.perf_counter()
    try:
        subprocess.run(
            [str(LKH_BIN), str(par_path)],
            capture_output=True, text=True, timeout=time_limit_s + 30,
        )
    except subprocess.TimeoutExpired:
        pass  # trata como "sem solucao" -- checado abaixo pela ausencia do arquivo
    wall = time.perf_counter() - t0
    if not sintef_path.exists():
        return None, wall
    routes = _ler_sintef(sintef_path, inst.depot_id)
    return routes, wall


def solve_lkh3(inst, time_limit_s, seed=0, vehicle_slack=VEHICLE_SLACK, workdir=None):
    """Resolve `inst` via LKH3. Para vehicles_exact=True (A/B), uma única
    execução com VEHICLES=n_vehicles_ref. Para vehicles_exact=False (X/CMT),
    varre VEHICLES de n_vehicles_ref a n_vehicles_ref+vehicle_slack,
    dividindo o orçamento de tempo igualmente entre as tentativas, e
    devolve a melhor solução viável (por custo REAL via
    `inst.solution_cost`, nunca pelo valor que o LKH3 imprime -- ver
    docstring do módulo, achado #27 do diário).

    Retorna (routes, wall_time_s_total): routes é lista de listas de ids
    de cliente reais (convenção de instance.py). Levanta RuntimeError se
    nenhuma tentativa achar solução viável dentro do tempo.
    """
    with tempfile.TemporaryDirectory(prefix="lkh3_") as tmp:
        wd = Path(workdir) if workdir else Path(tmp)
        wd.mkdir(parents=True, exist_ok=True)

        if inst.vehicles_exact:
            candidatos = [inst.n_vehicles_ref]
            tempo_cada = time_limit_s
        else:
            candidatos = list(range(inst.n_vehicles_ref, inst.n_vehicles_ref + vehicle_slack + 1))
            tempo_cada = time_limit_s / len(candidatos)

        melhor_routes = None
        melhor_custo = float("inf")
        tempo_total = 0.0
        for k in candidatos:
            routes, wall = _rodar_uma_vez(inst, k, tempo_cada, seed, wd)
            tempo_total += wall
            if routes is None:
                continue
            viavel, _ = inst.is_feasible(routes)
            if not viavel:
                continue
            custo = inst.solution_cost(routes)
            if custo < melhor_custo:
                melhor_custo = custo
                melhor_routes = routes

        if melhor_routes is None:
            raise RuntimeError(
                f"LKH3 nao achou solucao viavel para {inst.name} "
                f"(candidatos de VEHICLES={candidatos}, tempo_cada={tempo_cada:.2f}s)"
            )
        return melhor_routes, tempo_total
