#!/usr/bin/env python3
"""
Validacao de src/alns_loop_rrt.py (Estudo 22, diario, Entrada 138):
  (1) com aceitacao = "sa" o laco e IDENTICO ao de alns_loop.rodar_alns, execucao a execucao (custo, iteracoes, avaliacoes, inviaveis, usos, escores, avaliacoes por
      operador), com tres politicas (uniforme, Ropke e Pisinger, mistura estatica) em tres instancias de ajuste e duas seeds;
  (2) com aceitacao = "rrt": solucoes viaveis; deterministico dada a seed; com delta0 = 0 nenhuma solucao pior que o recorde e aceita (nenhum sigma3) e com delta0 > 0 ha
      sigma3, em numero crescente com delta0; o custo final nunca supera o da solucao inicial.
Uso: python tests/test_alns_loop_rrt.py     (do diretorio cvrp_bandit, venv de direcao1_local)
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "estudo_aos"))
from alns_loop import rodar_alns, construir_politica_alns, SIGMA3  # noqa: E402
from alns_loop_rrt import rodar_alns_aceitacao  # noqa: E402
from aos_vnd import EstaticaPolitica  # noqa: E402
import run_aos_etapa4 as r4  # noqa: E402
from run_aos_study import _instancia  # noqa: E402
from run_aos_etapa3 import _proxy  # noqa: E402


def politicas(nome, seed):
    if nome == "static":
        return ((EstaticaPolitica([0.3, 0.2, 0.1, 0.4], rng=random.Random(seed * 31 + 1)), "score"),
                (EstaticaPolitica([0.1, 0.7, 0.2], rng=random.Random(seed * 31 + 2)), "score"))
    return (construir_politica_alns(nome, 4, random.Random(seed * 31 + 1)), construir_politica_alns(nome, 3, random.Random(seed * 31 + 2)))


def rodar(fn, nome_inst, pol, seed, orc_frac, **kw):
    inst, cws = _instancia("SYN_TUNE", nome_inst)
    px = _proxy("SYN_TUNE", nome_inst)
    px.chamadas = 0
    (pd_, md), (pr_, mr) = politicas(pol, seed)
    max_rotas = inst.n_vehicles_ref if inst.vehicles_exact else None
    best, info = fn(px, [list(r) for r in cws], random.Random(seed), int(orc_frac * inst.n_customers ** 2), pd_, md, pr_, mr, lambda: px.chamadas, w0=0.02, tf_fator=1e-3,
                    max_rotas=max_rotas, **kw)
    return best, info, px.chamadas, px.solution_cost(best), px.solution_cost(cws), px.is_feasible(best)[0]


def main():
    nomes = sorted(r4.nomes("SYN_TUNE"))
    escolhidas = [nomes[0], nomes[8], nomes[17]]
    n_ok = 0
    for ni in escolhidas:
        for pol in ("U", "RP", "static"):
            for seed in (1, 2):
                a = rodar(rodar_alns, ni, pol, seed, 120)
                b = rodar(rodar_alns_aceitacao, ni, pol, seed, 120, aceitacao="sa")
                assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and abs(a[3] - b[3]) < 1e-9, (ni, pol, seed)
                n_ok += 1
    print(f"(1) aceitacao 'sa' identica ao laco de alns_loop.py em {n_ok} execucoes (custo, solucao, iteracoes, avaliacoes, usos, escores)")
    ni = escolhidas[1]
    contagem = {}
    for d0 in (0.0, 0.01, 0.05):
        sig3 = [0]

        def on_it(it, jd, jr, sigma, _s=sig3):
            _s[0] += int(abs(sigma - SIGMA3) < 1e-12)

        r1 = rodar(rodar_alns_aceitacao, ni, "RP", 3, 150, aceitacao="rrt", rrt_delta0=d0, on_iteracao=on_it)
        r2 = rodar(rodar_alns_aceitacao, ni, "RP", 3, 150, aceitacao="rrt", rrt_delta0=d0)
        assert r1[0] == r2[0] and r1[1] == r2[1], "nao deterministico"
        assert r1[5] and r1[3] <= r1[4] + 1e-9, (d0, "inviavel ou pior que a inicial")
        contagem[d0] = sig3[0]
    assert contagem[0.0] == 0 and contagem[0.01] > 0 and contagem[0.05] >= contagem[0.01], contagem
    print(f"(2) rrt: viavel, deterministico, custo final <= inicial; aceitos piores por delta0 {contagem} (delta0 = 0 nao aceita nenhum)")
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
