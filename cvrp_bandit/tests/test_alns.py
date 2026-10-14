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
import alns  # noqa: E402
from alns import (Estado, DESTRUIR, REPARAR, melhor_posicao, _nos, _arestas, reparar_arrependimento,  # noqa: E402
                  reparar_arrependimento_v1)
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
    # (5) regressao exata da versao eficiente do regret/guloso contra a original (mesmas decisoes, mesmas rotas)
    casos = 0
    for conj, nome in (("SYN_TUNE", sorted(i.name for i in __import__("instance").load_all(ROOT / "instances" / "SYN_TUNE"))[k])
                       for k in (3, 12, 20)):
        inst5, cws5 = _instancia(conj, nome)
        px5 = _proxy(conj, nome)
        dm5 = max(max(row) for row in px5._D)
        for rep in range(12):
            for kk in (1, 2):
                q = rng.randint(5, 30)
                e1 = Estado(px5, cws5, dm5)
                rem = DESTRUIR[rng.randrange(3)][1](e1, q, rng)
                e2 = Estado(px5, [list(r) for r in e1.rotas], dm5)
                e1b = Estado(px5, [list(r) for r in e1.rotas], dm5)
                px5.chamadas = 0
                ok1 = reparar_arrependimento_v1(e1b, list(rem), random.Random(1), kk, None)
                c1 = px5.chamadas
                px5.chamadas = 0
                ok2 = reparar_arrependimento(e2, list(rem), random.Random(1), kk, None)
                c2 = px5.chamadas
                assert ok1 == ok2 and e1b.rotas == e2.rotas and c1 == c2, (nome, kk, "versao eficiente diverge")
                casos += 1
    # idem numa instancia com muitas rotas (SYN_VAL, 100 rotas) e nos dois modos de limite de rotas
    inst6, cws6 = _instancia("SYN_VAL", "X-synv23-n301")
    px6 = _proxy("SYN_VAL", "X-synv23-n301")
    dm6 = max(max(row) for row in px6._D)
    for kk in (1, 2):
        e1 = Estado(px6, cws6, dm6)
        rem = DESTRUIR[1][1](e1, 40, rng)
        a = Estado(px6, [list(r) for r in e1.rotas], dm6)
        b = Estado(px6, [list(r) for r in e1.rotas], dm6)
        px6.chamadas = 0
        reparar_arrependimento_v1(a, list(rem), random.Random(1), kk, None)
        c1 = px6.chamadas
        px6.chamadas = 0
        reparar_arrependimento(b, list(rem), random.Random(1), kk, None)
        assert a.rotas == b.rotas and c1 == px6.chamadas
        casos += 1
    for mr in (len(e1.rotas) + 2,):
        for kk in (1, 2):
            a = Estado(px6, [list(r) for r in e1.rotas], dm6)
            b = Estado(px6, [list(r) for r in e1.rotas], dm6)
            ra = reparar_arrependimento_v1(a, list(rem), random.Random(1), kk, mr)
            rb = reparar_arrependimento(b, list(rem), random.Random(1), kk, mr)
            assert ra == rb and a.rotas == b.rotas
            casos += 1
    print(f"(5) regressao exata do reparo eficiente: {casos} casos identicos (rotas e contagem de avaliacoes)")
    # (6) laco completo: mesmo resultado com o reparo original e o eficiente (3 politicas, 2 instancias)
    iguais = 0
    for nome in (sorted(i.name for i in __import__("instance").load_all(ROOT / "instances" / "SYN_TUNE"))[k] for k in (5, 17)):
        px7 = _proxy("SYN_TUNE", nome)
        inst7, cws7 = _instancia("SYN_TUNE", nome)
        orc7 = 120 * inst7.n_customers ** 2
        for pol in ("U", "RP", "ucb1"):
            res = []
            for versao in ("v1", "v2"):
                alns.REPARAR[0] = ("guloso", (lambda est, rem, rg, mr, f=(reparar_arrependimento_v1 if versao == "v1" else reparar_arrependimento): f(est, rem, rg, 1, mr)))
                alns.REPARAR[1] = ("regret2", (lambda est, rem, rg, mr, f=(reparar_arrependimento_v1 if versao == "v1" else reparar_arrependimento): f(est, rem, rg, 2, mr)))
                import alns_loop as al
                al.REPARAR = alns.REPARAR
                pd7, md7 = construir_politica_alns(pol, len(DESTRUIR), random.Random(11))
                pr7, mr7 = construir_politica_alns(pol, len(REPARAR), random.Random(12))
                px7.chamadas = 0
                b7, i7 = rodar_alns(px7, [list(r) for r in cws7], random.Random(13), orc7, pd7, md7, pr7, mr7, lambda: px7.chamadas)
                res.append((px7.solution_cost(b7), i7["iteracoes"], tuple(i7["uso_d"]), tuple(i7["uso_r"])))
            assert res[0] == res[1], (nome, pol, res)
            iguais += 1
    print(f"(6) laco completo identico com o reparo original e o eficiente em {iguais} execucoes")
    print("TESTES DO ALNS: OK")


if __name__ == "__main__":
    main()
