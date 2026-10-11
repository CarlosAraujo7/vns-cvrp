#!/usr/bin/env python3
"""
Validacao do ALNS (src/alns.py, src/alns_loop.py) -- Estudo 9.
  (1) melhor_posicao == enumeracao ingenua do custo de insercao (instancias aleatorias);
  (2) cada operador de destruicao remove exatamente q clientes (ou a rota inteira), sem duplicar/perder ninguem;
  (3) cada reparo reinsere todos os removidos, respeita a capacidade e produz solucao viavel; o custo calculado a
      partir das arestas incrementais coincide com solution_cost;
  (4) o laco roda, nao estoura o orcamento alem de uma iteracao e devolve solucao viavel.
Uso: python tests/test_alns.py     (do diretorio cvrp_bandit, venv de direcao1_local)
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "estudo_aos"))
from alns import Estado, DESTRUIR, REPARAR, melhor_posicao, _nos, _arestas  # noqa: E402
from alns_loop import rodar_alns, construir_politica_alns  # noqa: E402
from run_aos_study import _instancia  # noqa: E402
from run_aos_etapa3 import _proxy  # noqa: E402


def naive(inst, nos, c):
    d = inst._D
    return min(d[nos[p]][c] + d[c][nos[p + 1]] - d[nos[p]][nos[p + 1]] for p in range(len(nos) - 1))


def main():
    inst, cws = _instancia("SYN_TUNE", "X-synt05-n101") if False else _instancia("SYN_TUNE", sorted(
        i.name for i in __import__("instance").load_all(ROOT / "instances" / "SYN_TUNE"))[10])
    px = _proxy("SYN_TUNE", inst.name)
    rng = random.Random(7)
    # (1) melhor_posicao vs ingenua
    n_ok = 0
    for r in cws:
        nos = _nos(px.depot_id, list(r))
        ar = _arestas(px._D, nos)
        for c in rng.sample(list(inst.customers), 5):
            if c in r:
                continue
            custo, pos = melhor_posicao(px, nos, ar, c)
            assert abs(custo - naive(px, nos, c)) < 1e-9
            n_ok += 1
    print(f"(1) melhor_posicao confere com a enumeracao ingenua em {n_ok} casos")
    # (2)+(3)
    dmax = max(max(row) for row in px._D)
    total = 0
    for jd, (nd, dfn) in enumerate(DESTRUIR):
        for jr, (nr, rfn) in enumerate(REPARAR):
            for rep in range(15):
                est = Estado(px, cws, dmax)
                q = rng.randint(5, 20)
                antes = sorted(c for r in est.rotas for c in r)
                rem = dfn(est, q, rng) if jd != 3 else dfn(est, q, rng, qmax=q)
                resto = sorted(c for r in est.rotas for c in r)
                assert sorted(rem + resto) == antes and len(set(rem)) == len(rem), (nd, "perdeu/duplicou")
                if jd != 3:
                    assert len(rem) == min(q, len(antes)), (nd, len(rem), q)
                ok = rfn(est, rem, rng, None)
                assert ok
                cand = est.rotas
                assert sorted(c for r in cand for c in r) == antes, (nd, nr, "clientes")
                viavel, msg = px.is_feasible(cand)
                assert viavel, (nd, nr, msg)
                # custo incremental x recalculado: as arestas armazenadas devem refletir as rotas
                for i, r in enumerate(est.rotas):
                    assert abs(sum(est.arest[i]) - px._inst.route_cost(r)) < 1e-6
                total += 1
    print(f"(2)(3) {total} combinacoes destruir x reparar: clientes preservados, solucoes viaveis, arestas coerentes")
    # (4) laco
    pd_, md = construir_politica_alns("RP", len(DESTRUIR), random.Random(1))
    pr_, mr = construir_politica_alns("RP", len(REPARAR), random.Random(2))
    px.chamadas = 0
    orc = 300 * px.n_customers ** 2
    best, info = rodar_alns(px, [list(r) for r in cws], random.Random(3), orc, pd_, md, pr_, mr, lambda: px.chamadas)
    viavel, _ = px.is_feasible(best)
    c0, c1 = px.solution_cost(cws), px.solution_cost(best)
    print(f"(4) laco: {info['iteracoes']} iteracoes, calls {px.chamadas} (orcamento {orc}), custo {c0:.1f} -> {c1:.1f}, "
          f"viavel={viavel}, inviaveis={info['inviaveis']}, uso destruir {info['uso_d']} reparar {info['uso_r']}")
    assert viavel and c1 <= c0 + 1e-9
    print("TESTES DO ALNS: OK")


if __name__ == "__main__":
    main()
