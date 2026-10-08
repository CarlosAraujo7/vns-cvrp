#!/usr/bin/env python3
"""
instance.py
===========
Leitor de instâncias CVRP no formato CVRPLIB (estilo TSPLIB) e utilitários
de distância/custo de rota.

Formato lido (confirmado por inspeção direta dos arquivos baixados de
https://galgos.inf.puc-rio.br/cvrplib/ em 29/08/2026 -- ver diario_pesquisa.md,
Entrada 3):

    NAME : <nome>
    COMMENT : <texto livre, às vezes contém o valor ótimo>
    TYPE : CVRP
    DIMENSION : <n+1, inclui o depósito>
    EDGE_WEIGHT_TYPE : EUC_2D
    CAPACITY : <capacidade do veículo>
    NODE_COORD_SECTION
    <id> <x> <y>
    ...
    DEMAND_SECTION
    <id> <demanda>
    ...
    DEPOT_SECTION
    <id do depósito>
    -1
    EOF

Cada .vrp tem um .sol irmão:
    Route #1: <clientes...>
    Route #2: <clientes...>
    ...
    Cost <valor>

Convenção de arredondamento de distância -- IMPORTANTE
--------------------------------------------------------
Descoberta empírica (não estava documentada explicitamente nos arquivos):
os conjuntos A, B e X usam distância Euclidiana ARREDONDADA para o inteiro
mais próximo (convenção clássica CVRPLIB/TSPLIB), enquanto o conjunto CMT
(na variante empacotada aqui, que é exatamente a variante "sem arredondar
a matriz de custo" citada por Kalatzantonakis et al. 2023, Seção 6.1) usa
distância Euclidiana em ponto flutuante, sem arredondar. Evidência: os
arquivos .sol de CMT têm custo com 3 casas decimais (ex.: CMT1 -> 524.611)
enquanto A/B/X têm custo inteiro (ex.: A-n32-k5 -> 784).

Em vez de deduzir a convenção a partir do nome do conjunto (frágil), esta
implementação AUTODETECTA por instância: lê a rota de referência do .sol,
calcula o custo total com e sem arredondamento, e adota a convenção cujo
resultado bate com o "Cost" declarado no .sol (tolerância 1e-3 no caso
contínuo). Se nenhuma das duas bater, levanta erro -- não assume nada
silenciosamente. Isso serve também de validação do leitor em si.

Restrição de número de veículos (K) -- BUG REAL encontrado e corrigido em
29/08/2026 (ver diario_pesquisa.md, Entrada 3): a formulação matemática do
CVRP usada por Kalatzantonakis et al. (2023), Seção 3, Equações 2c/2d,
exige IGUALDADE (exatamente K veículos saem e voltam ao depósito), não
"no máximo K". A primeira versão deste leitor não capturava K nenhum,
então o solver podia (e ocasionalmente fazia) usar MAIS rotas que o
permitido -- descoberto porque o GVNS baseline "bateu" o ótimo comprovado
de B-n57-k7 e B-n51-k7 usando 8 rotas em vez das 7 exigidas (confirmado
via busca de literatura: Letchford & Salazar-González 2004 provam 1153 e
1032 como ótimos exatos por branch-and-cut -- não há erro nesses valores,
o erro era não impor K no meu modelo). Confirmado empiricamente: para os
conjuntos A, B e CMT, TODA rota de referência usa exatamente o "No of
trucks" do COMMENT (quando presente) e sempre bate com len(reference_routes);
para o conjunto X (Uchoa et al. 2017), o próprio artigo original declara
que o número de rotas é só um LIMITE INFERIOR ("the X set... obeys the
rule of not setting a fixed number of routes"), confirmado empiricamente
aqui também (X-n195-k51 usa 53 rotas na referência, não as 51 do nome).
Por isso `vehicles_exact` é AUTODETECTADO por `load_instance()` a partir do
nome da instância (ver `_vehicles_exact_default()`), podendo ser sobrescrito
explicitamente por quem chama; `n_vehicles_ref` guarda sempre
len(reference_routes) como o valor-alvo (exato ou limite inferior, conforme
`vehicles_exact`).

CORREÇÃO IMPORTANTE (29/08/2026, diario_pesquisa.md, Entrada 3): a
suposição original de que o conjunto CMT também exige EXATAMENTE K
veículos (por analogia com A/B, e por uma checagem indireta de que as
rotas de referência de CMT1/2/3/11 "batem" com o K do nome) estava
ERRADA. Fonte primária definitiva -- Uchoa et al. (2017), o próprio artigo
que introduziu o conjunto X e documenta explicitamente as convenções dos
conjuntos mais antigos -- afirma sem ambiguidade que a série E/M/A/B
FIXA o número de rotas, mas "the CMT versions explicitly did not fix K"
(a re-publicação CMT dessas mesmas instâncias E/M abandonou tanto o
arredondamento de distância QUANTO a fixação de K, não só o
arredondamento). O fato de as referências de CMT1/2/3/11 usarem
coincidentemente exatamente K rotas não contradiz isso -- só significa
que, para essas 4 instâncias específicas, usar mais que K veículos não
reduz o custo total (ao contrário de B-n51-k7/B-n57-k7, onde usar 8
veículos em vez dos 7 "fixos" reduz o custo -- ver a mesma Entrada 3 para
a análise completa de por que isso acontece: o objetivo do CVRP puro
[Eq. 1a do artigo] só soma distância, sem custo fixo por veículo, então
nada impede que usar 1 veículo a mais seja estritamente mais barato,
exceto quando K é fixado por restrição de igualdade). Conclusão: CMT
agora é tratado como `vehicles_exact=False` (limite inferior, igual ao
conjunto X), não `True`.
"""
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CVRPInstance:
    name: str
    dimension: int  # inclui o depósito
    capacity: int
    coords: dict      # id (1-based) -> (x, y)
    demands: dict      # id (1-based) -> demanda
    depot_id: int
    round_distances: bool  # True = nint(dist), False = dist em ponto flutuante
    optimum: float          # "Cost" do .sol de referência
    reference_routes: list  # lista de listas de ids de cliente (sem depósito)
    n_vehicles_ref: int = 0       # numero de rotas na solucao de referencia
    vehicles_exact: bool = True   # True: EXATAMENTE n_vehicles_ref rotas (A/B/CMT).
                                   # False: PELO MENOS n_vehicles_ref rotas (X).
    source_vrp: str = field(repr=False, default="")
    source_sol: str = field(repr=False, default="")

    @property
    def n_customers(self):
        return self.dimension - 1

    @property
    def customers(self):
        return [i for i in self.coords if i != self.depot_id]

    def dist(self, i, j):
        if i == j:
            return 0.0
        xi, yi = self.coords[i]
        xj, yj = self.coords[j]
        d = math.hypot(xi - xj, yi - yj)
        return round(d) if self.round_distances else d

    def route_cost(self, route):
        """Custo de UMA rota (lista de ids de cliente, sem depósito nas pontas)."""
        if not route:
            return 0.0
        c = self.dist(self.depot_id, route[0])
        for a, b in zip(route, route[1:]):
            c += self.dist(a, b)
        c += self.dist(route[-1], self.depot_id)
        return c

    def solution_cost(self, routes):
        return sum(self.route_cost(r) for r in routes)

    def route_demand(self, route):
        return sum(self.demands[i] for i in route)

    def is_feasible(self, routes):
        """Cobertura exata de clientes (sem repetição, sem omissão) + capacidade
        + restrição de número de veículos (ver docstring do módulo)."""
        routes_nao_vazias = [r for r in routes if len(r) > 0]
        seen = []
        for r in routes_nao_vazias:
            if self.route_demand(r) > self.capacity + 1e-6:
                return False, f"rota excede capacidade: {r}"
            seen.extend(r)
        if sorted(seen) != sorted(self.customers):
            faltando = set(self.customers) - set(seen)
            repetidos = [c for c in set(seen) if seen.count(c) > 1]
            return False, f"cobertura inválida (faltando={faltando}, repetidos={repetidos})"
        n_rotas = len(routes_nao_vazias)
        if self.n_vehicles_ref:
            if self.vehicles_exact and n_rotas != self.n_vehicles_ref:
                return False, (
                    f"numero de veiculos incorreto: {n_rotas} rotas, "
                    f"instancia exige exatamente {self.n_vehicles_ref}"
                )
            if not self.vehicles_exact and n_rotas < self.n_vehicles_ref:
                return False, (
                    f"numero de veiculos abaixo do limite inferior: {n_rotas} rotas, "
                    f"instancia exige pelo menos {self.n_vehicles_ref}"
                )
        return True, "ok"


def _parse_vrp(path: Path):
    coords = {}
    demands = {}
    depot_id = None
    dimension = None
    capacity = None
    name = None
    section = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line == "EOF":
                continue
            if line.startswith("NAME"):
                name = line.split(":", 1)[1].strip()
                continue
            if line.startswith("DIMENSION"):
                dimension = int(line.split(":", 1)[1].strip())
                continue
            if line.startswith("CAPACITY"):
                capacity = int(float(line.split(":", 1)[1].strip()))
                continue
            if line.startswith("COMMENT") or line.startswith("TYPE") or line.startswith("EDGE_WEIGHT_TYPE"):
                continue
            if line == "NODE_COORD_SECTION":
                section = "coord"
                continue
            if line == "DEMAND_SECTION":
                section = "demand"
                continue
            if line == "DEPOT_SECTION":
                section = "depot"
                continue
            parts = line.split()
            if section == "coord":
                idx = int(parts[0])
                coords[idx] = (float(parts[1]), float(parts[2]))
            elif section == "demand":
                idx = int(parts[0])
                demands[idx] = int(float(parts[1]))
            elif section == "depot":
                val = int(parts[0])
                if val != -1 and depot_id is None:
                    depot_id = val
    assert name is not None, f"NAME não encontrado em {path}"
    assert dimension is not None, f"DIMENSION não encontrado em {path}"
    assert capacity is not None, f"CAPACITY não encontrado em {path}"
    assert depot_id is not None, f"DEPOT_SECTION não encontrado em {path}"
    assert len(coords) == dimension, (
        f"{path}: esperado {dimension} coordenadas, achado {len(coords)}"
    )
    assert len(demands) == dimension, (
        f"{path}: esperado {dimension} demandas, achado {len(demands)}"
    )
    return name, dimension, capacity, coords, demands, depot_id


def _parse_sol(path: Path):
    """Lê rotas e custo do .sol.

    IMPORTANTE (bug real encontrado e corrigido em 29/08/2026, ver
    diario_pesquisa.md Entrada 3): os índices de cliente no .sol da CVRPLIB
    NÃO usam o mesmo id do NODE_COORD_SECTION do .vrp. O .vrp numera todos
    os nós de 1 a DIMENSION, incluindo o depósito (convencionalmente id 1).
    O .sol numera só os CLIENTES, de 1 a DIMENSION-1, pulando o depósito.
    Ou seja, "cliente 21" no .sol é o nó de id 22 no .vrp (quando o depósito
    é o nó 1). Confirmado por verificação numérica manual (recalculando a
    rota #1 de A-n32-k5 nas duas convenções e comparando com o Cost=784
    declarado -- só a convenção com deslocamento bate).

    Esta função retorna os ids EXATAMENTE como aparecem no .sol (sem
    deslocar); quem chama (`load_instance`) aplica o deslocamento correto
    conhecendo o depot_id real da instância.
    """
    routes = []
    cost = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.lower().startswith("route"):
                after_colon = line.split(":", 1)[1].strip()
                routes.append([int(x) for x in after_colon.split()])
            elif line.lower().startswith("cost"):
                cost = float(line.split()[1])
    assert cost is not None, f"Cost não encontrado em {path}"
    return routes, cost


def _sol_routes_to_vrp_ids(sol_routes, depot_id):
    """Converte numeração de cliente do .sol (1..n, pulando o depósito)
    para os ids reais do .vrp (1..DIMENSION, incluindo o depósito)."""
    def map_id(c):
        return c if c < depot_id else c + 1

    return [[map_id(c) for c in route] for route in sol_routes]


def _vehicles_exact_default(name):
    """Autodetecta a convenção de número de veículos a partir do NOME da
    instância (não do diretório -- mais robusto a reorganizações de
    pastas): conjuntos X e CMT usam K como LIMITE INFERIOR apenas (ver
    docstring do módulo para a correção de 29/08/2026 sobre CMT); os
    demais (A, B, e potencialmente E/M/P se algum dia forem adicionados)
    fixam K como EXATO, convenção clássica confirmada em Uchoa et al.
    (2017)."""
    if name.startswith("X") or name.startswith("CMT"):
        return False
    return True


def load_instance(vrp_path, sol_path=None, tol=1e-2, vehicles_exact=None) -> CVRPInstance:
    vrp_path = Path(vrp_path)
    if sol_path is None:
        sol_path = vrp_path.with_suffix(".sol")
    sol_path = Path(sol_path)

    name, dimension, capacity, coords, demands, depot_id = _parse_vrp(vrp_path)
    ref_routes_sol_ids, declared_cost = _parse_sol(sol_path)
    ref_routes = _sol_routes_to_vrp_ids(ref_routes_sol_ids, depot_id)

    # autodetecta arredondamento comparando com a rota de referência do .sol
    def cost_with(round_flag):
        total = 0.0
        for r in ref_routes:
            c = 0.0
            prev = depot_id
            for node in r:
                dx = coords[prev][0] - coords[node][0]
                dy = coords[prev][1] - coords[node][1]
                d = math.hypot(dx, dy)
                c += round(d) if round_flag else d
                prev = node
            dx = coords[prev][0] - coords[depot_id][0]
            dy = coords[prev][1] - coords[depot_id][1]
            d = math.hypot(dx, dy)
            c += round(d) if round_flag else d
            total += c
        return total

    cost_round = cost_with(True)
    cost_float = cost_with(False)
    match_round = abs(cost_round - declared_cost) <= max(tol, 0.5)
    match_float = abs(cost_float - declared_cost) <= tol

    if match_round and not match_float:
        round_distances = True
    elif match_float and not match_round:
        round_distances = False
    elif match_round and match_float:
        # instâncias com coordenadas inteiras onde as duas convenções coincidem
        round_distances = True
    else:
        raise ValueError(
            f"{vrp_path.name}: nenhuma convenção de distância reproduz o Cost "
            f"declarado ({declared_cost}). Calculado arredondado={cost_round}, "
            f"float={cost_float}. Não assumindo nada -- investigar."
        )

    if vehicles_exact is None:
        vehicles_exact = _vehicles_exact_default(name)

    inst = CVRPInstance(
        name=name,
        dimension=dimension,
        capacity=capacity,
        coords=coords,
        demands=demands,
        depot_id=depot_id,
        round_distances=round_distances,
        optimum=declared_cost,
        reference_routes=ref_routes,
        n_vehicles_ref=len(ref_routes),
        vehicles_exact=vehicles_exact,
        source_vrp=str(vrp_path),
        source_sol=str(sol_path),
    )
    return inst


def load_all(instances_dir, vehicles_exact=None):
    """Carrega todas as instâncias .vrp de um diretório (não recursivo).
    vehicles_exact=None (padrão) autodetecta por instância a partir do
    nome (ver _vehicles_exact_default) -- correto mesmo se o diretório
    misturar instâncias de convenções diferentes."""
    instances_dir = Path(instances_dir)
    result = []
    for vrp_file in sorted(instances_dir.glob("*.vrp")):
        result.append(load_instance(vrp_file, vehicles_exact=vehicles_exact))
    return result
