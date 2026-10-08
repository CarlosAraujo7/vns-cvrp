"""
solver_hgs.py
=============
Wrapper do HGS-CVRP (Vidal 2022, "Hybrid Genetic Search for the CVRP:
Open-Source Implementation and SWAP* Neighborhood", Computers & OR 140;
código MIT em https://github.com/vidalt/HGS-CVRP) -- o solver de
referência de estado da arte para CVRP. Entra na Direção 1 como TETO de
qualidade (diário, Entrada 61), não como rival direto do VNS em Python:
serve para calibrar onde estamos de fato em relação ao estado da arte.

Binário esperado em `external/HGS-CVRP/hgs` (compilado com
`g++ -std=c++17 -O3 -DNDEBUG Program/*.cpp -o hgs`; ver
direcao1_local/README.md). Roda dentro do WSL/Linux.

Convenções:
- IDs: o HGS numera clientes 1..n pela ORDEM do arquivo .vrp (depósito
  primeiro). Aqui isso é remapeado para os ids reais de `instance.py`.
- Distância: `-round 1` se inst.round_distances, senão `-round 0` (CMT).
- Frota: A/B (vehicles_exact=True) passam `-veh K`; para o HGS isso é um
  TETO de frota, não igualdade -- a solução pode usar menos rotas, e
  quem decide a viabilidade é sempre `inst.is_feasible` (mesmo critério
  dos demais solvers). X/CMT: frota livre (default do HGS).
- O custo impresso pelo HGS nunca é confiado: sempre recalculado via
  `inst.solution_cost` (mesma lição do achado #27 do LKH3).
- `-t` do HGS mede tempo de CPU (single-thread, equivale a parede em
  máquina sem contenção); o tempo de parede real é medido aqui.
"""
import re
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HGS_BIN = ROOT / "external" / "HGS-CVRP" / "hgs"


def _ler_solucao(sol_path, ids_por_ordem):
    routes = []
    for linha in sol_path.read_text().splitlines():
        m = re.match(r"Route #\d+:\s*(.*)", linha)
        if m:
            clientes = [int(x) for x in m.group(1).split()]
            if clientes:
                routes.append([ids_por_ordem[c] for c in clientes])
    return routes


def solve_hgs(inst, time_limit_s, seed=0):
    """Retorna (routes, wall_time_s). Levanta RuntimeError se o HGS não
    produzir arquivo de solução."""
    ids_por_ordem = list(inst.coords)
    if ids_por_ordem[0] != inst.depot_id:
        raise RuntimeError(f"{inst.name}: depósito não é o primeiro nó do arquivo; "
                           f"remapeamento de ids do HGS não é válido.")
    with tempfile.TemporaryDirectory(prefix="hgs_") as tmp:
        sol_path = Path(tmp) / "out.sol"
        cmd = [str(HGS_BIN), str(inst.source_vrp), str(sol_path),
               "-t", f"{time_limit_s:.4f}", "-seed", str(seed),
               "-round", "1" if inst.round_distances else "0", "-log", "0"]
        if inst.vehicles_exact:
            cmd += ["-veh", str(inst.n_vehicles_ref)]
        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit_s + 60)
        except subprocess.TimeoutExpired:
            pass
        wall = time.perf_counter() - t0
        if not sol_path.exists():
            raise RuntimeError(f"HGS não produziu solução para {inst.name} (t={time_limit_s:.2f}s)")
        return _ler_solucao(sol_path, ids_por_ordem), wall
