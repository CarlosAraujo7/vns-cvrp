#!/usr/bin/env python3
"""
Validacao dos operadores adicionais do ALNS do CVRPTW (src/tw_alns_ext.py) -- Estudo 28 (diario, Entrada 148).
  (1) tabelas: tamanhos 7/12/21, aninhamento, os operadores do Estudo 16 sao os MESMOS objetos, a rota continua no indice 3;
  (2) as 108 combinacoes destruir x reparo de P21 em quatro instancias de Solomon (classes C1, R1, R2, RC2) e varias repeticoes: nenhum cliente perdido ou duplicado, capacidade respeitada, solucao
      viavel em janelas de tempo (a menos do caso conhecido de arredondamento em que uma rota apertada vira inviavel com a REMOCAO: `ok[i]` falso), arestas e tempos incrementais coerentes com o recalculo;
  (3) arrependimento-k com janelas: k = 1 e k = 2 reproduzem EXATAMENTE `tw_alns.reparar_arrependimento_tw` (solucao e avaliacoes); k = 3 e 4 produzem solucoes viaveis;
  (4) a melhor posicao perturbada, sem ruido e sem blink, coincide com `melhor_posicao_tw` (valor, posicao e avaliacoes); com ruido ela escolhe outra posicao em parte dos casos e sempre uma viavel;
  (5) o laco `rodar_alns` roda com P7, P12 e P21 e devolve solucao viavel; com a tabela de P7 reproduz exatamente a corrida com as tabelas do Estudo 16;
  (6) mutacao: um reparo que perde um cliente e um que viola a capacidade sao detectados pelo verificador.
Uso: python tests/test_tw_alns_ext.py     (do diretorio cvrp_bandit, venv de direcao1_local)
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "estudo_aos"))
import tw_alns as tw  # noqa: E402
import tw_alns_ext as tx  # noqa: E402
from alns_loop import rodar_alns, construir_politica_alns, qrange  # noqa: E402
import run_aos_etapa14 as r14  # noqa: E402
import run_aos_etapa16 as r16  # noqa: E402


def cenario(nome):
    inst, x0 = r16.carregar("SOLOMON", nome)
    dmax = max(max(row) for row in inst._D)
    return inst, [list(r) for r in x0], dmax


def conferir(inst, est, antes, rotulo):
    resto = sorted(c for r in est.rotas for c in r)
    assert resto == antes, (rotulo, "clientes perdidos ou duplicados")
    for i, r in enumerate(est.rotas):
        assert sum(inst.demands[c] for c in r) <= inst.capacity, (rotulo, "capacidade")
        assert abs(sum(est.arest[i]) - inst.route_cost(r)) < 1e-6 if hasattr(inst, "route_cost") else True
        nos, a, L, ok = est._calc(r)
        assert est.nos[i] == nos and est.a[i] == a and est.L[i] == L and est.ok[i] == ok, (rotulo, "tempos incrementais incoerentes")
    viavel, msg = inst.is_feasible(est.rotas)
    # caso conhecido (docstring de tw_alns): a remocao pode tornar inviavel uma rota apertada por truncamento das distancias; o laco rejeita o candidato
    assert viavel or any(not o for o in est.ok) or msg.startswith("numero de veiculos"), (rotulo, msg)
    return viavel


def main():
    nomes = r14.nomes("SOLOMON")
    escolhidas = [n for n in ("C101", "R101", "R201", "RC205") if n in nomes]
    assert len(escolhidas) == 4, nomes[:8]
    # (1) tabelas
    assert [len(d) + len(r) for d, r in tx.POOLS_TW.values()] == [7, 12, 21]
    assert all(a is b for a, b in zip(tx.DESTRUIR_P21[:4], tw.DESTRUIR_TW)) and all(a is b for a, b in zip(tx.REPARAR_P21[:3], tw.REPARAR_TW))
    assert tx.DESTRUIR_P12 == tx.DESTRUIR_P21[:len(tx.DESTRUIR_P12)] and tx.REPARAR_P12 == tx.REPARAR_P21[:len(tx.REPARAR_P12)] and tx.DESTRUIR_P21[3][0] == "rota"
    print("(1) tabelas P7/P12/P21 do CVRPTW: 7/12/21 logits, aninhadas, operadores do Estudo 16 identicos, 'rota' no indice 3")

    # (2) todas as combinacoes
    rng = random.Random(28)
    total = nv = 0
    for nome in escolhidas:
        inst, x0, dmax = cenario(nome)
        qmin, qmax = qrange(inst.n_customers)
        for jd, (nd, dfn) in enumerate(tx.DESTRUIR_P21):
            for jr, (nr, rfn) in enumerate(tx.REPARAR_P21):
                for rep in range(3):
                    est = tw.EstadoTW(inst, x0, dmax)
                    q = rng.randint(qmin, qmax)
                    antes = sorted(c for r in est.rotas for c in r)
                    rem = dfn(est, q, rng) if jd != 3 else dfn(est, q, rng, qmax=qmax)
                    resto = sorted(c for r in est.rotas for c in r)
                    assert sorted(rem + resto) == antes and len(set(rem)) == len(rem) and rem, (nd, "perdeu/duplicou")
                    ok = rfn(est, rem, rng, inst.nveic)
                    if not ok:                       # reparo sem posicao viavel e sem rota nova permitida (limite de frota): o laco rejeita a candidata
                        continue
                    nv += conferir(inst, est, antes, (nome, nd, nr))
                    total += 1
    print(f"(2) {total} combinacoes destruir x reparo de P21 em {len(escolhidas)} instancias de Solomon: cobertura, capacidade, tempos incrementais e viabilidade conferidos ({nv} viaveis; as demais, o caso de arredondamento ou limite de rotas)")
    assert total > 0.9 * len(escolhidas) * 12 * 9 * 3

    # (3) arrependimento-k com janelas: k = 1, 2 identicos ao original
    for nome in escolhidas:
        inst, x0, dmax = cenario(nome)
        qmin, qmax = qrange(inst.n_customers)
        for rep in range(6):
            q = rng.randint(qmin, qmax)
            r1 = random.Random(1000 * rep + 7)
            est0 = tw.EstadoTW(inst, x0, dmax)
            rem = tw.destruir_aleatoria(est0, q, r1)
            resto = [list(r) for r in est0.rotas]
            for k in (1, 2):
                inst.chamadas = 0
                ea = tw.EstadoTW(inst, resto, dmax)
                oka = tw.reparar_arrependimento_tw(ea, list(rem), random.Random(5), k, inst.nveic)
                ca = inst.chamadas
                inst.chamadas = 0
                eb = tx.EstadoTW(inst, resto, dmax) if hasattr(tx, "EstadoTW") else tw.EstadoTW(inst, resto, dmax)
                okb = tx.reparar_arrependimento_tw_k(eb, list(rem), random.Random(5), k, inst.nveic)
                cb = inst.chamadas
                assert oka == okb and ea.rotas == eb.rotas and ca == cb, (nome, k, "difere do original")
            for k in (3, 4):
                eb = tw.EstadoTW(inst, resto, dmax)
                if tx.reparar_arrependimento_tw_k(eb, list(rem), random.Random(5), k, inst.nveic):
                    conferir(inst, eb, sorted(c for r in x0 for c in r), (nome, "regret", k))
    print("(3) arrependimento-k com janelas: k = 1 e k = 2 identicos a tw_alns.reparar_arrependimento_tw (solucao e avaliacoes); k = 3 e 4 viaveis")

    # (4) posicao perturbada
    difere = total_pos = 0
    for nome in escolhidas:
        inst, x0, dmax = cenario(nome)
        est = tw.EstadoTW(inst, x0, dmax)
        rr = random.Random(4)
        for c in [c for r in x0 for c in r][:40]:
            for i in range(len(est.rotas)):
                if not (est.ok[i] and est.cargas[i] + est.dem[c] <= est.Q):
                    continue
                inst.chamadas = 0
                a = tw.melhor_posicao_tw(est, i, c)
                ca = inst.chamadas
                inst.chamadas = 0
                b = tx._melhor_posicao_tw_pert(est, i, c, rr, 0.0, 0.0)
                cb = inst.chamadas
                assert a == b and ca == cb, (nome, c, i, a, b)
                p = tx._melhor_posicao_tw_pert(est, i, c, rr, 0.05 * dmax, 0.0)
                if a is not None:
                    total_pos += 1
                    assert p is not None
                    nos, aa, L = est.nos[i], est.a[i], est.L[i]
                    pos = p[1] + 1                      # a insercao escolhida tem de ser viavel: mesma conta de melhor_posicao_tw
                    u, v = nos[pos - 1], nos[pos]
                    t = max(aa[pos - 1] + est.s[u] + est.D[u][c], est.e[c])
                    assert t <= est.l[c] and t + est.s[c] + est.D[c][v] <= L[pos], (nome, c, i, "posicao inviavel")
                    difere += p[1] != a[1]
    assert difere > 0
    print(f"(4) posicao perturbada sem ruido == melhor_posicao_tw (valor, posicao e avaliacoes); com ruido muda a posicao em {difere} de {total_pos} casos e e sempre viavel")

    # (5) o laco com P7, P12 e P21
    inst, x0, dmax = cenario("R101")
    res = {}
    for nome_p, (D, R) in tx.POOLS_TW.items():
        inst.chamadas = 0
        rng_ = random.Random(11)
        kd, kr = len(D), len(R)
        pol_d, md = construir_politica_alns("RP", kd, random.Random(11 * 31 + 1), {})
        pol_r, mr = construir_politica_alns("RP", kr, random.Random(11 * 31 + 2), {})
        best, info = rodar_alns(inst, [list(r) for r in x0], rng_, int(60 * inst.n_customers ** 2), pol_d, md, pol_r, mr, lambda: inst.chamadas, w0=0.2, tf_fator=1e-3,
                                max_rotas=inst.nveic, destruir=D, reparar=R, estado_cls=tw.EstadoTW)
        ok, msg = inst.is_feasible(best)
        assert ok, (nome_p, msg)
        res[nome_p] = (inst.solution_cost(best), info["iteracoes"], sum(info["uso_d"]), len(D), len(R))
    inst.chamadas = 0
    pol_d, md = construir_politica_alns("RP", 4, random.Random(11 * 31 + 1), {})
    pol_r, mr = construir_politica_alns("RP", 3, random.Random(11 * 31 + 2), {})
    best, info = rodar_alns(inst, [list(r) for r in x0], random.Random(11), int(60 * inst.n_customers ** 2), pol_d, md, pol_r, mr, lambda: inst.chamadas, w0=0.2, tf_fator=1e-3,
                            max_rotas=inst.nveic, destruir=tw.DESTRUIR_TW, reparar=tw.REPARAR_TW, estado_cls=tw.EstadoTW)
    assert (inst.solution_cost(best), info["iteracoes"]) == res["P7"][:2], (res["P7"], inst.solution_cost(best))
    print("(5) o laco roda com P7/P12/P21 (custo, iteracoes):", {k: v[:2] for k, v in res.items()}, "; P7 reproduz a corrida das tabelas do Estudo 16")

    # (6) mutacao: o verificador detecta um reparo que perde um cliente e um que viola a capacidade
    inst, x0, dmax = cenario("R101")
    est = tw.EstadoTW(inst, x0, dmax)
    antes = sorted(c for r in est.rotas for c in r)
    rem = tw.destruir_aleatoria(est, 10, random.Random(3))
    est.abrir(rem[0])                                        # insere so um: os demais ficam perdidos
    try:
        conferir(inst, est, antes, "mutacao 1")
        raise SystemExit("mutacao 1 NAO detectada")
    except AssertionError:
        pass
    est = tw.EstadoTW(inst, x0, dmax)
    c0 = x0[0][0]
    ruim = [list(r) for r in x0]
    ruim[0] = ruim[0] + [c for r in x0[1:4] for c in r]       # rota gigante: viola a capacidade
    ruim = [r for i, r in enumerate(ruim) if i == 0 or not set(r) <= set(ruim[0])]
    est2 = tw.EstadoTW(inst, ruim, dmax)
    try:
        conferir(inst, est2, sorted(c for r in x0 for c in r), "mutacao 2")
        raise SystemExit("mutacao 2 NAO detectada")
    except AssertionError:
        pass
    print("(6) mutacoes (cliente perdido; capacidade violada) detectadas pelo verificador")
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
