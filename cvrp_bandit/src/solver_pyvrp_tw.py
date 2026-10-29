"""
solver_pyvrp_tw.py
==================
Solucao de referencia do CVRPTW (Estudo 14) com o PyVRP (Wouda, Lan & Kool 2024; MIT): constroi o problema a partir da MESMA matriz de
distancias truncadas (decimos) da TWInst e devolve as rotas; o custo e a viabilidade sao SEMPRE recomputados por tw_inst (nunca lidos do
PyVRP). Objetivo: distancia total, frota livre ate inst.nveic veiculos.
"""
import pyvrp
from pyvrp import Model
from pyvrp.stop import MaxRuntime


def resolver_pyvrp(inst, tempo, seed=1):
    m = Model()
    xs, ys = inst.coords[:, 0], inst.coords[:, 1]
    dep = m.add_depot(x=int(round(xs[0])), y=int(round(ys[0])), tw_early=int(inst.e[0]), tw_late=int(inst.l[0]))
    m.add_vehicle_type(num_available=inst.nveic, capacity=int(inst.Q), start_depot=dep, end_depot=dep)
    loc = [dep]
    for c in range(1, inst.N):
        loc.append(m.add_client(x=int(round(xs[c])), y=int(round(ys[c])), delivery=int(inst.dem[c]), service_duration=int(inst.s[c]),
                                tw_early=int(inst.e[c]), tw_late=int(inst.l[c])))
    for i in range(inst.N):
        for j in range(inst.N):
            if i != j:
                m.add_edge(loc[i], loc[j], distance=int(inst.D[i, j]), duration=int(inst.D[i, j]))
    res = m.solve(stop=MaxRuntime(tempo), seed=seed, display=False)
    rotas = [list(r.visits()) for r in res.best.routes()]
    return rotas
