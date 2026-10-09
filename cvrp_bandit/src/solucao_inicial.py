"""
solucao_inicial.py
==================
Solucao inicial DETERMINISTICA e COMPARTILHADA (diario, Entrada 88, achado F1).

Motivo: `construction.repair_route_count` recorre, nas instancias A/B de
empacotamento apertado (K exato), a um fallback com CP-SAT do OR-Tools
(`_pack_exact_k_feasible`: limite de TEMPO de 12 s e `num_search_workers =
cpu_count()`), que NAO e deterministico -- a solucao encontrada depende de
timing e carga, e cada processo-worker dos experimentos a calculava uma vez
e a guardava em cache, de modo que o ponto de partida dessas instancias
variava entre workers (violando "mesma solucao inicial para todos os
bracos"). Aqui a solucao inicial de cada instancia e calculada UMA vez
(`estudo_aos/gerar_solucoes_iniciais.py`), gravada em
`instances/_init/<nome>.json` e carregada por todos os runners; faltar o
arquivo e um ERRO (nunca se recalcula em silencio).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT_DIR = ROOT / "instances" / "_init"


def caminho(nome):
    return INIT_DIR / f"{nome}.json"


def obter_inicial(inst):
    """Rotas iniciais (lista de listas de ids de cliente) gravadas para `inst`."""
    p = caminho(inst.name)
    if not p.exists():
        raise FileNotFoundError(
            f"solucao inicial de {inst.name} nao encontrada em {p}; rode "
            f"estudo_aos/gerar_solucoes_iniciais.py (sequencial, antes dos experimentos)")
    dados = json.loads(p.read_text(encoding="utf-8"))
    rotas = [[int(c) for c in r] for r in dados["rotas"]]
    return rotas


def gravar_inicial(inst, rotas, origem):
    INIT_DIR.mkdir(parents=True, exist_ok=True)
    caminho(inst.name).write_text(json.dumps({
        "instancia": inst.name, "n_rotas": len(rotas), "custo": inst.solution_cost(rotas),
        "origem": origem, "rotas": rotas}), encoding="utf-8")
