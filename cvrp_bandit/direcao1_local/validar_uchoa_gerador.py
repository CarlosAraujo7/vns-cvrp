#!/usr/bin/env python3
"""
validar_uchoa_gerador.py
==========================
Valida `uchoa_gerador_core.gerar_instancia_uchoa` de duas formas:

(1) COMPARAÇÃO DIRETA byte-a-byte contra o gerador OFICIAL (rodado como
    subprocesso real, a partir de external/uchoa_generator/generator.py),
    para várias combinações de (n, depot_pos, cust_pos, demand_type,
    avg_route_size, seed). Se as fórmulas e a ordem de consumo de
    aleatoriedade estiverem corretas, o resultado deve ser IDÊNTICO --
    mesmo depósito, mesmos clientes (mesma ordem), mesmas demandas (mesma
    ordem pós-shuffle) e mesma capacidade.

(2) CHECAGENS ESTRUTURAIS (sem crash, ranges válidos, sem pontos
    duplicados, capacidade>=1, feasibility básica) em ~40 combinações
    aleatórias cobrindo todo o espaço de 378 atributos, para pegar
    qualquer erro que só apareça em cantos não cobertos pelo teste (1).

Uso: python validar_uchoa_gerador.py
"""
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uchoa_gerador_core import gerar_instancia_uchoa, COMBINACOES_VALIDAS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GERADOR_OFICIAL = ROOT / "external" / "uchoa_generator" / "generator.py"


def rodar_gerador_oficial(n, depot_pos, cust_pos, demand_type, avg_route_size, instance_id, seed, workdir):
    """Roda o script oficial como subprocesso isolado (PYTHONHASHSEED fixo
    por precaução, embora hash de tuplas de int não seja afetado por ele)
    e retorna (depot_xy, clientes_xy, demandas, capacidade) parseados do
    .vrp gerado -- mesma ordem de V usada internamente pelo script (depot
    primeiro, depois list(S))."""
    cmd = [sys.executable, str(GERADOR_OFICIAL), str(n), str(depot_pos), str(cust_pos),
           str(demand_type), str(avg_route_size), str(instance_id), str(seed)]
    env = {"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"}
    r = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"gerador oficial falhou: {r.stdout}\n{r.stderr}")

    arquivos = list(Path(workdir).glob("XML*.vrp"))
    if len(arquivos) != 1:
        raise RuntimeError(f"esperava 1 arquivo .vrp, achei {len(arquivos)}: {arquivos}")
    linhas = arquivos[0].read_text().splitlines()
    arquivos[0].unlink()  # limpa para a próxima chamada não confundir globs

    capacidade = None
    coords = {}
    demandas = {}
    secao = None
    for ln in linhas:
        ln = ln.strip()
        if ln.startswith("CAPACITY"):
            capacidade = int(ln.split(":")[1].strip())
        elif ln == "NODE_COORD_SECTION":
            secao = "coord"
        elif ln == "DEMAND_SECTION":
            secao = "demand"
        elif ln == "DEPOT_SECTION":
            secao = None
        elif secao == "coord" and ln:
            partes = ln.split()
            coords[int(partes[0])] = (int(partes[1]), int(partes[2]))
        elif secao == "demand" and ln:
            partes = ln.split()
            demandas[int(partes[0])] = int(partes[1])

    depot_xy = coords[1]
    clientes_xy = [coords[i] for i in range(2, n + 2)]
    demandas_lista = [demandas[i] for i in range(2, n + 2)]
    return depot_xy, clientes_xy, demandas_lista, capacidade


def teste_comparacao_direta():
    print("=== (1) Comparação byte-a-byte com o gerador oficial (subprocesso) ===\n")
    casos = [
        # (n, depot_pos, cust_pos, demand_type, avg_route_size, seed)
        (20, 1, 1, 1, 1, 111),
        (20, 2, 2, 2, 2, 222),
        (20, 3, 3, 3, 3, 333),
        (30, 1, 2, 6, 4, 444),   # tipo 6 (quadrante) -- não é embaralhado
        (30, 2, 3, 7, 2, 555),   # tipo 7 (poucos grandes) -- depende de route_size
        (50, 3, 1, 4, 5, 666),
        (50, 1, 3, 5, 6, 777),
        (10, 2, 2, 6, 1, 888),   # n pequeno + tipo 6 clusterizado
        (100, 2, 3, 7, 3, 42),   # tamanho real usado no fine-tuning do Kool
        (100, 1, 1, 1, 1, 1000),
    ]
    n_ok, n_falhas = 0, 0
    with tempfile.TemporaryDirectory() as workdir:
        for (n, dp, cp, dt, rs, seed) in casos:
            oficial = rodar_gerador_oficial(n, dp, cp, dt, rs, instance_id=1, seed=seed, workdir=workdir)
            nosso = gerar_instancia_uchoa(n, dp, cp, dt, rs, rng=random.Random(seed))

            tag = f"n={n} depot_pos={dp} cust_pos={cp} demand_type={dt} route_size={rs} seed={seed}"
            ok = (oficial[0] == nosso[0]) and (oficial[1] == nosso[1]) and \
                 (oficial[2] == nosso[2]) and (oficial[3] == nosso[3])
            if ok:
                print(f"  OK    {tag}")
                n_ok += 1
            else:
                n_falhas += 1
                print(f"  FALHA {tag}")
                if oficial[0] != nosso[0]:
                    print(f"        depot: oficial={oficial[0]} nosso={nosso[0]}")
                if oficial[1] != nosso[1]:
                    n_diff = sum(1 for a, b in zip(oficial[1], nosso[1]) if a != b)
                    print(f"        clientes: {n_diff}/{n} posicoes diferentes "
                          f"(len oficial={len(oficial[1])}, len nosso={len(nosso[1])})")
                if oficial[2] != nosso[2]:
                    n_diff = sum(1 for a, b in zip(oficial[2], nosso[2]) if a != b)
                    print(f"        demandas: {n_diff}/{n} posicoes diferentes")
                if oficial[3] != nosso[3]:
                    print(f"        capacidade: oficial={oficial[3]} nosso={nosso[3]}")
    print(f"\n  Resultado: {n_ok}/{n_ok + n_falhas} casos IDÊNTICOS ao gerador oficial.\n")
    return n_falhas == 0


def teste_estrutural():
    print("=== (2) Checagens estruturais em combinações aleatórias (378 possíveis) ===\n")
    rng = random.Random(2026)
    n_casos = 40
    problemas = []
    for _ in range(n_casos):
        dp, cp, dt, rs = rng.choice(COMBINACOES_VALIDAS)
        n = rng.choice([10, 20, 30, 50, 75, 100])
        seed = rng.randint(0, 10**6)
        try:
            depot, clientes, demandas, capacidade = gerar_instancia_uchoa(
                n, dp, cp, dt, rs, rng=random.Random(seed))
        except Exception as e:
            problemas.append(f"CRASH n={n} dp={dp} cp={cp} dt={dt} rs={rs} seed={seed}: {e}")
            continue

        tag = f"n={n} dp={dp} cp={cp} dt={dt} rs={rs} seed={seed}"
        if len(clientes) != n:
            problemas.append(f"{tag}: len(clientes)={len(clientes)} != n")
        if len(demandas) != n:
            problemas.append(f"{tag}: len(demandas)={len(demandas)} != n")
        todos_pontos = [depot] + clientes
        if len(set(todos_pontos)) != len(todos_pontos):
            problemas.append(f"{tag}: pontos duplicados (depot incluso)")
        for (x, y) in todos_pontos:
            if not (0 <= x <= 1000 and 0 <= y <= 1000):
                problemas.append(f"{tag}: ponto fora de [0,1000]: ({x},{y})")
                break
        if any(d <= 0 for d in demandas):
            problemas.append(f"{tag}: demanda <= 0 encontrada")
        if capacidade < 1:
            problemas.append(f"{tag}: capacidade={capacidade} < 1")
        if capacidade < max(demandas):
            problemas.append(f"{tag}: capacidade={capacidade} < max(demandas)={max(demandas)} "
                              f"(rota nao conseguiria atender o maior cliente sozinha)")

    if problemas:
        print(f"  {len(problemas)} problema(s) encontrado(s):")
        for p in problemas:
            print(f"    - {p}")
    else:
        print(f"  Todas as {n_casos} combinações passaram (sem crash, ranges válidos, "
              f"sem duplicatas, capacidade >= max(demanda)).")
    print()
    return len(problemas) == 0


def teste_reprodutibilidade():
    print("=== (3) Reprodutibilidade: mesma seed -> mesmo resultado ===\n")
    a = gerar_instancia_uchoa(50, 2, 3, 7, 3, rng=random.Random(999))
    b = gerar_instancia_uchoa(50, 2, 3, 7, 3, rng=random.Random(999))
    ok = a == b
    print(f"  {'OK' if ok else 'FALHA'}: duas chamadas com random.Random(999) "
          f"{'produziram' if ok else 'NAO produziram'} resultado idêntico.\n")
    return ok


def teste_amostragem_combinacoes():
    print("=== (4) amostrar_combinacao cobre o espaço de 378 combinações ===\n")
    from uchoa_gerador_core import amostrar_combinacao
    rng = random.Random(7)
    vistas = set()
    for _ in range(20000):
        vistas.add(amostrar_combinacao(rng))
    ok = len(COMBINACOES_VALIDAS) == 378 and len(vistas) == 378
    print(f"  COMBINACOES_VALIDAS tem {len(COMBINACOES_VALIDAS)} entradas "
          f"(esperado 378 = 3x3x7x6).")
    print(f"  Em 20000 amostras, {len(vistas)}/378 combinações distintas apareceram.")
    print(f"  {'OK' if ok else 'FALHA'}\n")
    return ok


def main():
    resultados = {
        "comparacao_direta": teste_comparacao_direta(),
        "estrutural": teste_estrutural(),
        "reprodutibilidade": teste_reprodutibilidade(),
        "amostragem_combinacoes": teste_amostragem_combinacoes(),
    }
    print("=== RESUMO ===")
    for nome, ok in resultados.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {nome}")
    if all(resultados.values()):
        print("\nTODOS OS TESTES PASSARAM.")
        sys.exit(0)
    else:
        print("\nALGUM TESTE FALHOU -- nao usar o gerador em producao ate corrigir.")
        sys.exit(1)


if __name__ == "__main__":
    main()
