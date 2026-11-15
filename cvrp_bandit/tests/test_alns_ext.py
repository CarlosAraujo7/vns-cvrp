#!/usr/bin/env python3
"""
Validacao dos operadores adicionais do ALNS (src/alns_ext.py) -- Estudo 21 (diario, Entrada 136).
  (1) tabelas: tamanhos 7/12/21, aninhamento, os operadores do Estudo 9 sao os MESMOS objetos, a rota continua no indice 3;
  (2) 108 combinacoes destruir x reparo de P21, em duas instancias e varias repeticoes: nenhum cliente perdido ou duplicado, solucao viavel, arestas
      incrementais coerentes com o custo recalculado;
  (3) propriedades de cada destruicao nova: radial remove exatamente q; sequencia remove cadeias CONTIGUAS (uma por rota) e no maximo q; demanda remove
      os q de demanda mais proxima da do semente; rota_pequena remove uma rota de tamanho minimo; leve/forte removem 0,4 q / 2 q;
  (4) arrependimento-k incremental == referencia ingenua (k = 2, 3, 4; com e sem limite de rotas) e k = 2 == alns.reparar_arrependimento;
  (5) a melhor posicao perturbada, sem ruido e sem blink, coincide com `melhor_posicao`; o ruido muda a escolha em parte dos casos;
  (6) contabilidade: demanda nao gasta avaliacao; radial e sequencia gastam exatamente o numero de clientes na solucao; reparos sequenciais gastam
      a mesma quantidade de avaliacoes que `reparar_barato` quando nao ha perturbacao;
  (7) o laco `rodar_alns` roda com cada conjunto e devolve solucao viavel; com a tabela de P7 reproduz exatamente a corrida padrao;
  (8) mutacao: um reparo que perde um cliente e um que viola a capacidade sao detectados pelo verificador.
Uso: python tests/test_alns_ext.py     (do diretorio cvrp_bandit, venv de direcao1_local)
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "estudo_aos"))
import alns  # noqa: E402
from alns import Estado, DESTRUIR, REPARAR, reparar_arrependimento, reparar_barato, melhor_posicao, _nos, _arestas  # noqa: E402
import alns_ext as ax  # noqa: E402
from alns_loop import rodar_alns, construir_politica_alns, qrange  # noqa: E402
import run_aos_etapa4 as r4  # noqa: E402
from run_aos_study import _instancia  # noqa: E402
from run_aos_etapa3 import _proxy  # noqa: E402


def conferir(px, est, antes, rotulo, completo=True):
    resto = sorted(c for r in est.rotas for c in r)
    if completo:
        assert resto == antes, (rotulo, "clientes perdidos ou duplicados")
        viavel, msg = px.is_feasible(est.rotas)
        # nas instancias de K minimo, um reparo que funde rotas gera um candidato com menos rotas que o limite: o laco o rejeita (conta como
        # `inviaveis`); aqui so se aceita ESSA mensagem (capacidade e cobertura sao conferidas por is_feasible antes da contagem de rotas)
        assert viavel or msg.startswith("numero de veiculos abaixo do limite inferior"), (rotulo, msg)
    for i, r in enumerate(est.rotas):
        assert abs(sum(est.arest[i]) - px._inst.route_cost(r)) < 1e-6, (rotulo, "arestas incoerentes")


def cenario(conj, nome):
    inst, cws = _instancia(conj, nome)
    px = _proxy(conj, nome)
    dmax = max(max(row) for row in px._D)
    return px, [list(r) for r in cws], dmax


def main():
    nome_syn = sorted(r4.nomes("SYN_TUNE"))[10]
    nome_x = sorted(r4.nomes("X"))[3]
    cen = [("SYN_TUNE", nome_syn), ("X", nome_x)]
    # (1) tabelas
    assert [len(d) + len(r) for d, r in ax.POOLS.values()] == [7, 12, 21]
    assert ax.DESTRUIR_P21[:4] == DESTRUIR and ax.REPARAR_P21[:3] == REPARAR
    assert ax.DESTRUIR_P12 == ax.DESTRUIR_P21[:len(ax.DESTRUIR_P12)] and ax.REPARAR_P12 == ax.REPARAR_P21[:len(ax.REPARAR_P12)]
    assert ax.DESTRUIR_P21[3][0] == "rota" and all(a is b for a, b in zip(ax.DESTRUIR_P21[:4], DESTRUIR))
    print("(1) tabelas P7/P12/P21: 7/12/21 logits, aninhadas, operadores do Estudo 9 identicos, 'rota' no indice 3")

    # (2) todas as combinacoes
    rng = random.Random(21)
    total = 0
    for conj, nome in cen:
        px, cws, dmax = cenario(conj, nome)
        qmin, qmax = qrange(px.n_customers)
        for jd, (nd, dfn) in enumerate(ax.DESTRUIR_P21):
            for jr, (nr, rfn) in enumerate(ax.REPARAR_P21):
                for rep in range(4):
                    est = Estado(px, cws, dmax)
                    q = rng.randint(qmin, qmax)
                    antes = sorted(c for r in est.rotas for c in r)
                    rem = dfn(est, q, rng) if jd != 3 else dfn(est, q, rng, qmax=qmax)
                    resto = sorted(c for r in est.rotas for c in r)
                    assert sorted(rem + resto) == antes and len(set(rem)) == len(rem) and rem, (nd, "perdeu/duplicou")
                    conferir(px, est, resto + sorted(rem) if False else resto, (nd, nr), completo=False)
                    ok = rfn(est, rem, rng, None)
                    assert ok, (nd, nr)
                    conferir(px, est, antes, (nd, nr))
                    total += 1
                    if jd >= 4 and jr >= 3 and rep == 0:                    # com limite de rotas (caminho nao incremental dos reparos por arrependimento)
                        est = Estado(px, cws, dmax)
                        rem = dfn(est, q, rng)
                        assert rfn(est, rem, rng, len(cws) + 2), (nd, nr, "limite de rotas")
                        conferir(px, est, antes, (nd, nr, "max_rotas"))
    print(f"(2) {total} combinacoes destruir x reparo de P21 em 2 instancias: clientes preservados, solucoes viaveis, arestas coerentes")

    # (3) propriedades das destruicoes novas
    px, cws, dmax = cenario("SYN_TUNE", nome_syn)
    n = px.n_customers
    for rep in range(30):
        q = rng.randint(8, 25)
        est = Estado(px, cws, dmax)
        assert len(ax.destruir_radial(est, q, rng)) == q
        est = Estado(px, cws, dmax)
        original = [list(r) for r in est.rotas]
        rem = ax.destruir_sequencia(est, q, rng)
        assert 1 <= len(rem) <= q
        for r in original:
            idx = [j for j, c in enumerate(r) if c in set(rem)]
            assert not idx or idx == list(range(idx[0], idx[-1] + 1)), "cadeia nao contigua"
            assert len(idx) <= ax.LMAX_CADEIA
        est = Estado(px, cws, dmax)
        rem = ax.destruir_demanda(est, q, rng)
        assert len(rem) == q
        dm = [est.dem[c] for c in rem]
        todas = sorted(est.dem[c] for r in original for c in r)
        # os removidos formam o conjunto de demanda mais proxima de ALGUM cliente-semente: o intervalo de demandas e curto
        assert max(dm) - min(dm) <= todas[-1] - todas[0]
        est = Estado(px, cws, dmax)
        tam = sorted(len(r) for r in est.rotas)
        rem = ax.destruir_rota_pequena(est, q, rng)
        assert len(rem) == tam[0]
        est = Estado(px, cws, dmax)
        assert len(ax.destruir_aleatoria_leve(est, q, rng)) == max(2, round(0.4 * q))
        est = Estado(px, cws, dmax)
        assert len(ax.destruir_aleatoria_forte(est, q, rng)) == min(n, max(2, round(2.0 * q)))
        est = Estado(px, cws, dmax)
        assert len(ax.destruir_pior_leve(est, q, rng)) == max(2, round(0.4 * q))
        est = Estado(px, cws, dmax)
        assert len(ax.destruir_relacionada_forte(est, q, rng)) == min(n, max(2, round(1.7 * q)))
    # demanda: o conjunto removido e exatamente os q de menor |dem - dem_semente| (referencia independente)
    for rep in range(20):
        q = rng.randint(5, 20)
        est = Estado(px, cws, dmax)
        r2 = random.Random(1000 + rep)
        rem = set(ax.destruir_demanda(est, q, random.Random(1000 + rep)))
        est2 = Estado(px, cws, dmax)
        todos = [c for r in est2.rotas for c in r]
        sem = random.Random(1000 + rep).choice(todos)
        dist = sorted(abs(est2.dem[c] - est2.dem[sem]) for c in todos)[q - 1]
        assert all(abs(est2.dem[c] - est2.dem[sem]) <= dist for c in rem)
        assert sem in rem
    # radial: os removidos sao os q mais proximos do semente
    for rep in range(20):
        q = rng.randint(5, 20)
        est = Estado(px, cws, dmax)
        rem = set(ax.destruir_radial(est, q, random.Random(2000 + rep)))
        est2 = Estado(px, cws, dmax)
        todos = [c for r in est2.rotas for c in r]
        sem = random.Random(2000 + rep).choice(todos)
        lim = sorted(px._D[sem][c] for c in todos)[q - 1]
        assert all(px._D[sem][c] <= lim + 1e-12 for c in rem) and sem in rem
    print("(3) destruicoes novas: radial = q mais proximos, sequencia = cadeias contiguas <= q, demanda = q de demanda mais proxima, rota_pequena = rota minima, leve/forte")

    # (4) arrependimento-k incremental == ingenuo
    n_ok = 0
    for conj, nome in cen:
        px, cws, dmax = cenario(conj, nome)
        qmin, qmax = qrange(px.n_customers)
        for k in (2, 3, 4):
            for mr in (None, len(cws) + 3):
                for rep in range(6):
                    q = rng.randint(qmin, qmax)
                    est_a = Estado(px, cws, dmax)
                    rem = alns.destruir_aleatoria(est_a, q, random.Random(3000 + rep))
                    est_b = Estado(px, cws, dmax)
                    alns.destruir_aleatoria(est_b, q, random.Random(3000 + rep))
                    ra = ax.reparar_arrependimento_k(est_a, list(rem), rng, k, mr)
                    rb = ax.reparar_arrependimento_k_ingenuo(est_b, list(rem), rng, k, mr)
                    assert ra == rb and est_a.rotas == est_b.rotas, ("regret-k incremental difere do ingenuo", conj, k, mr, rep)
                    if k == 2:
                        est_c = Estado(px, cws, dmax)
                        alns.destruir_aleatoria(est_c, q, random.Random(3000 + rep))
                        reparar_arrependimento(est_c, list(rem), rng, 2, mr)
                        assert est_c.rotas == est_a.rotas, "regret-2 de alns.py difere do generalizado"
                    n_ok += 1
    print(f"(4) arrependimento-k incremental coincide com a referencia ingenua em {n_ok} casos (k = 2, 3, 4; com e sem limite de rotas)")

    # (5) posicao perturbada
    px, cws, dmax = cenario("SYN_TUNE", nome_syn)
    mudou = tot = 0
    for r in cws:
        nos = _nos(px.depot_id, list(r))
        ar = _arestas(px._D, nos)
        for c in rng.sample(list(px.customers), 6):
            if c in r:
                continue
            a = melhor_posicao(px, nos, ar, c)
            b = ax._melhor_posicao_pert(px, nos, ar, c, rng, 0.0, 0.0)
            assert a == b, (a, b)
            ru = ax._melhor_posicao_pert(px, nos, ar, c, rng, ax.ETA_RUIDO * dmax, 0.0)
            tot += 1
            mudou += (ru[1] != a[1])
    assert 0 < mudou < tot
    print(f"(5) posicao perturbada sem ruido == melhor_posicao; o ruido muda a posicao em {mudou} de {tot} casos")

    # (6) contabilidade
    est = Estado(px, cws, dmax)
    px.chamadas = 0
    ax.destruir_demanda(est, 12, rng)
    assert px.chamadas == 0
    for fn in (ax.destruir_radial, ax.destruir_sequencia):
        est = Estado(px, cws, dmax)
        ncli = sum(len(r) for r in est.rotas)
        px.chamadas = 0
        fn(est, 12, rng)
        assert px.chamadas == ncli, (fn.__name__, px.chamadas, ncli)
    est1, est2 = Estado(px, cws, dmax), Estado(px, cws, dmax)
    rem = alns.destruir_aleatoria(est1, 15, random.Random(5))
    alns.destruir_aleatoria(est2, 15, random.Random(5))
    ordem = sorted(rem)
    px.chamadas = 0
    ax.inserir_sequencial(est1, ordem, rng, None)
    c1 = px.chamadas
    px.chamadas = 0
    for c in ordem:                                   # o laco de reparar_barato sem o embaralhamento
        op = alns._opcoes(est2, c)
        mi = min(op, key=lambda i: op[i][0]) if op else None
        cn = px.dist(est2.dep, c) * 2.0
        if mi is None or cn < op[mi][0]:
            est2.abrir(c)
        else:
            est2.inserir(mi, op[mi][1], c)
    assert px.chamadas == c1 and est1.rotas == est2.rotas
    print("(6) contabilidade: demanda 0, radial e sequencia n avaliacoes, insercao sequencial identica ao laco de reparar_barato")

    # (7) o laco com cada conjunto; P7 reproduz a corrida padrao
    for nome_pool, (D, R) in ax.POOLS.items():
        px, cws, dmax = cenario("SYN_TUNE", nome_syn)
        pd_, md = construir_politica_alns("RP", len(D), random.Random(1))
        pr_, mr = construir_politica_alns("RP", len(R), random.Random(2))
        px.chamadas = 0
        orc = 120 * px.n_customers ** 2
        best, info = rodar_alns(px, [list(r) for r in cws], random.Random(3), orc, pd_, md, pr_, mr, lambda: px.chamadas, destruir=D, reparar=R)
        viavel, _ = px.is_feasible(best)
        assert viavel and info["iteracoes"] > 10 and len(info["uso_d"]) == len(D) and len(info["uso_r"]) == len(R)
        assert px.solution_cost(best) <= px.solution_cost(cws) + 1e-9
        print(f"(7) {nome_pool}: {info['iteracoes']} iteracoes, inviaveis {info['inviaveis']}, custo {px.solution_cost(cws):.1f} -> {px.solution_cost(best):.1f}")
    resultados = []
    for tabelas in (None, ax.POOLS["P7"]):
        px, cws, dmax = cenario("SYN_TUNE", nome_syn)
        pd_, md = construir_politica_alns("RP", 4, random.Random(1))
        pr_, mr = construir_politica_alns("RP", 3, random.Random(2))
        px.chamadas = 0
        kw = {} if tabelas is None else {"destruir": tabelas[0], "reparar": tabelas[1]}
        best, info = rodar_alns(px, [list(r) for r in cws], random.Random(3), 80 * px.n_customers ** 2, pd_, md, pr_, mr, lambda: px.chamadas, **kw)
        resultados.append((px.solution_cost(best), info["iteracoes"], px.chamadas, info["uso_d"], info["uso_r"]))
    assert resultados[0] == resultados[1], "P7 nao reproduz a corrida padrao"
    print("(7) a tabela P7 reproduz exatamente a corrida padrao")

    # (8) mutacao: o verificador tem de pegar reparos defeituosos
    px, cws, dmax = cenario("SYN_TUNE", nome_syn)

    def perde(est, rem, rng_, mr):
        return reparar_barato(est, rem[1:], rng_, mr)

    def estoura(est, rem, rng_, mr):
        est.abrir(rem[0])
        for c in rem[1:]:
            est.rotas[0].append(c)
            est._refaz(0)
        return True

    detectados = 0
    for defeituoso in (perde, estoura):
        est = Estado(px, cws, dmax)
        antes = sorted(c for r in est.rotas for c in r)
        rem = ax.destruir_radial(est, 20, rng)
        defeituoso(est, rem, rng, None)
        try:
            conferir(px, est, antes, defeituoso.__name__)
        except AssertionError:
            detectados += 1
    assert detectados == 2
    print("(8) mutacao: 2 de 2 reparos defeituosos detectados")
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
