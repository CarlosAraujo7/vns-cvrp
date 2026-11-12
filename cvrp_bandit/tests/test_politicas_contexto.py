#!/usr/bin/env python3
"""
Validacao das politicas com estado (src/politicas_contexto.py) e do gancho de contexto do laco do ALNS -- Estudo 18 (diario, Entrada 127).
  (1) IDENTIDADES EXATAS: CtxUCB com um so contexto == UCB1Masc; RPTilt com p0 uniforme == RopkePisingerPolitica (escolhas e pesos);
      RPTilt com rho = 0 == EstaticaPolitica; FasesMist com fases iguais == EstaticaPolitica; LogLinMist com a = b = 0 == softmax(theta).
  (2) LinUCB: a inversa mantida por Sherman-Morrison coincide com a inversa direta de lam I + sum x x'; indice == formula direta.
  (3) Q-learning: atualizacao TD conferida numa trilha calculada a mao.
  (4) CONTROLE DE CAPACIDADE (ambiente sintetico, o melhor braco troca na metade do orcamento): as politicas com estado devem seguir a
      troca e superar o UCB1 sem estado; sem isso, um resultado nulo no ALNS nao seria informativo.
  (5) GANCHO: set_context e chamado uma vez por iteracao, antes da selecao, com frac em [0,1] nao decrescente, estag zerada apos nova melhor,
      ult em {0,1,2,3}; e o gancho e INERTE: uma politica identica com um set_context que nada faz produz EXATAMENTE a mesma execucao.
Uso: python tests/test_politicas_contexto.py     (do diretorio cvrp_bandit, venv de direcao1_local)
"""
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "estudo_aos"))
from aos_vnd import UCB1Masc, EstaticaPolitica  # noqa: E402
from alns_loop import RopkePisingerPolitica, rodar_alns, construir_politica_alns  # noqa: E402
from alns import DESTRUIR, REPARAR  # noqa: E402
import politicas_contexto as pc  # noqa: E402
from politicas_contexto import (CtxUCBPolitica, LinUCBPolitica, QLearnPolitica, RPTiltPolitica, FasesMistPolitica,  # noqa: E402
                                LogLinMistPolitica, caracteristicas, softmax_np)

SIGMAS = [0.0, 9.0, 13.0, 33.0]


def ctx_aleatorio(r, t, T):
    return {"frac": t / T, "estag": r.randint(0, 600), "gap_cur": r.random() * 0.1, "ult": r.randint(0, 3), "q": r.random()}


# ------------------------------------------------------------------------------------------------ (1) identidades
def teste_ctxucb_igual_ucb1():
    for k, C in ((4, 1.0), (3, 2.0), (4, 0.3)):
        a = CtxUCBPolitica(k, random.Random(5), C=C, tau=10 ** 9, n_fases=1)
        b = UCB1Masc(k, C=C, rng=random.Random(5))
        rr, rs = random.Random(99), random.Random(98)
        for t in range(6000):
            a.set_context(ctx_aleatorio(rs, t, 6000))
            allowed = list(range(k)) if t % 7 else [j for j in range(k) if rs.random() < 0.7] or [0]
            ja, jb = a.select_masked(allowed), b.select_masked(allowed)
            assert ja == jb, (k, C, t, ja, jb)
            rew = rr.choice(SIGMAS) / 33.0 * (1 if rr.random() < 0.3 + 0.15 * ja else 0)
            a.update(ja, rew)
            b.update(jb, rew)
    print("(1a) CtxUCB com um so contexto reproduz o UCB1 mascarado em 18.000 passos (3 configuracoes, com mascaras)")


def teste_tilt_igual_rp():
    for k in (4, 3):
        a = RPTiltPolitica(k, random.Random(3), [1.0] * k, segmento=50, rho=0.2)
        b = RopkePisingerPolitica(k, random.Random(3), segmento=50, rho=0.2)
        rr = random.Random(4)
        for t in range(8000):
            ja, jb = a.select_masked(list(range(k))), b.select_masked(list(range(k)))
            assert ja == jb, (k, t)
            rew = rr.choice(SIGMAS) * (1 if rr.random() < 0.3 + 0.2 * ja else 0)
            a.update(ja, rew)
            b.update(jb, rew)
        assert a.w == b.w
    print("(1b) RPTilt com prior uniforme reproduz o RP de Ropke e Pisinger (escolhas e pesos) em 16.000 passos")


def teste_tilt_rho0_igual_estatica():
    for k in (4, 3):
        rs = random.Random(11)
        p0 = [rs.random() + 0.1 for _ in range(k)]
        p0 = [x / sum(p0) for x in p0]
        a = RPTiltPolitica(k, random.Random(8), p0, segmento=100, rho=0.0)
        b = EstaticaPolitica(p0, rng=random.Random(8))
        rr = random.Random(9)
        for t in range(20000):
            ja, jb = a.select_masked(list(range(k))), b.select_masked(list(range(k)))
            assert ja == jb, (k, t)
            a.update(ja, rr.choice(SIGMAS))
            b.update(jb, 0.0)
    print("(1c) RPTilt com rho = 0 reproduz a mistura estatica em 40.000 passos")


def teste_fases_e_loglin():
    th = [0.3, -0.2, 0.5, 0.1]
    a = FasesMistPolitica(4, random.Random(2), [th, th, th])
    b = EstaticaPolitica(softmax_np(th), rng=random.Random(2))
    rs = random.Random(1)
    for t in range(20000):
        a.set_context(ctx_aleatorio(rs, t, 20000))
        assert a.select_masked([0, 1, 2, 3]) == b.select_masked([0, 1, 2, 3]), t
    print("(1d) FasesMist com fases iguais reproduz a mistura estatica em 20.000 passos")
    # fases diferentes: a frequencia empirica em cada fase confere com o softmax daquela fase
    lg = [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]
    f = FasesMistPolitica(4, random.Random(3), lg)
    for fase, frac in enumerate((0.1, 0.5, 0.9)):
        c = [0] * 4
        for _ in range(30000):
            f.set_context({"frac": frac, "estag": 0, "gap_cur": 0.0, "ult": 0, "q": 0.5})
            c[f.select_masked([0, 1, 2, 3])] += 1
        esp = softmax_np(lg[fase])
        assert max(abs(c[j] / 30000 - esp[j]) for j in range(4)) < 0.01, (fase, c, esp)
    print("(1e) FasesMist: a frequencia empirica por fase confere com o softmax da fase")
    ll = LogLinMistPolitica(4, random.Random(4), th, [0.0] * 4, [0.0] * 4)
    ll.set_context({"frac": 0.8, "estag": 300, "gap_cur": 0.0, "ult": 0, "q": 0.5})
    p = ll.probabilidades([0, 1, 2, 3])
    assert max(abs(x - y) for x, y in zip(p, softmax_np(th))) < 1e-12
    ll2 = LogLinMistPolitica(4, random.Random(4), th, [0.0, 3.0, 0.0, 0.0], [0.0] * 4)
    ll2.set_context({"frac": 0.1, "estag": 0, "gap_cur": 0.0, "ult": 0, "q": 0.5})
    p_cedo = ll2.probabilidades([0, 1, 2, 3])[1]
    ll2.set_context({"frac": 0.9, "estag": 0, "gap_cur": 0.0, "ult": 0, "q": 0.5})
    p_tarde = ll2.probabilidades([0, 1, 2, 3])[1]
    assert p_tarde > p_cedo + 0.1
    print(f"(1f) LogLinMist: a = b = 0 reproduz o softmax; a_j > 0 aumenta a probabilidade do operador j no fim ({p_cedo:.3f} -> {p_tarde:.3f})")


# ------------------------------------------------------------------------------------------------ (2) LinUCB
def teste_linucb():
    k, lam = 3, 2.0
    pol = LinUCBPolitica(k, random.Random(1), alpha=0.7, lam=lam)
    rs = random.Random(5)
    X = [[] for _ in range(k)]
    Y = [[] for _ in range(k)]
    for t in range(400):
        c = ctx_aleatorio(rs, t, 400)
        pol.set_context(c)
        j = pol.select_masked([0, 1, 2])
        r = rs.choice([0.0, 0.2, 1.0])
        X[j].append(caracteristicas(c))
        Y[j].append(r)
        pol.update(j, r)
    c = ctx_aleatorio(rs, 200, 400)
    pol.set_context(c)
    x = np.array(caracteristicas(c))
    for j in range(k):
        A = lam * np.eye(pc.D_CAR) + sum((np.outer(v, v) for v in map(np.array, X[j])), np.zeros((pc.D_CAR, pc.D_CAR)))
        b = sum((r * np.array(v) for r, v in zip(Y[j], X[j])), np.zeros(pc.D_CAR))
        assert np.max(np.abs(np.linalg.inv(A) - pol.Ainv[j])) < 1e-9
        direto = float(np.linalg.solve(A, b) @ x) + 0.7 * math.sqrt(float(x @ np.linalg.inv(A) @ x))
        assert abs(direto - pol.indice(j)) < 1e-9
    print("(2) LinUCB: a inversa de Sherman-Morrison e o indice coincidem com a conta direta (3 operadores, 400 passos)")


# ------------------------------------------------------------------------------------------------ (3) Q-learning
def teste_qlearn():
    q = QLearnPolitica(3, random.Random(1), alpha=0.5, gamma=0.9, eps0=0.0, tau=100, n_fases=3, eps_min=0.0)
    c1 = {"frac": 0.0, "estag": 0, "gap_cur": 0.0, "ult": 0, "q": 0.5}
    q.set_context(c1)
    s1 = q._st
    a = q.select_masked([0, 1, 2])           # Q = 0 em tudo: desempate aleatorio
    q.update(a, 1.0)
    c2 = {"frac": 0.5, "estag": 150, "gap_cur": 0.0, "ult": 3, "q": 0.5}
    q.set_context(c2)                        # aplica a atualizacao com s' = estado de c2 (Q(s', .) = 0)
    assert abs(q.Q[s1][a] - 0.5 * (1.0 + 0.9 * 0.0 - 0.0)) < 1e-12
    s2 = q._st
    assert s2 != s1 and s2 == (1, 3, 1)
    b = q.select_masked([0, 1, 2])
    q.update(b, 0.0)
    q.Q[s1] = [0.0, 0.0, 0.0]
    q.Q[s1][1] = 0.8                          # max Q(s1, .) = 0.8
    q.set_context(c1)                         # s' = s1: alvo = 0 + 0.9 * 0.8
    assert abs(q.Q[s2][b] - 0.5 * (0.0 + 0.9 * 0.8 - 0.0)) < 1e-12
    print("(3) Q-learning: a atualizacao TD confere com a conta a mao (dois passos, estados distintos)")


# ------------------------------------------------------------------------------------------------ (4) capacidade
# Dois ambientes sinteticos (4 bracos, recompensa de Bernoulli; o ambiente revela o estado em ctx):
#   "troca de fase"        : o melhor braco (media 0,6 contra 0,2-0,3) muda de 0 para 1 na metade do orcamento;
#   "estagnacao alternante": estag percorre 0..49 em ciclos; o melhor braco e 0 se estag <= 25 e 1 caso contrario.
# Uma unica mudanca abrupta tambem e acompanhada pelo UCB1 sem estado (a media do novo melhor braco sobe rapido e o bonus de
# exploracao o testa); a dependencia ALTERNANTE do estado so e capturada por quem usa o contexto.
AMBIENTES = {
    "troca de fase": (lambda frac, est, j: ([0.6, 0.2, 0.3, 0.3] if frac < 0.5 else [0.2, 0.6, 0.3, 0.3])[j],
                      lambda frac, est: 0 if frac < 0.5 else 1),
    "estagnacao alternante": (lambda frac, est, j: ([0.6, 0.2, 0.3, 0.3] if est <= 25 else [0.2, 0.6, 0.3, 0.3])[j],
                              lambda frac, est: 0 if est <= 25 else 1),
}


def jogar(pol, T, seed, ambiente, escala=1.0):
    media, melhor = AMBIENTES[ambiente]
    rr = random.Random(seed)
    boas, ult = 0, 0
    tem = hasattr(pol, "set_context")
    for t in range(T):
        frac, est = t / T, t % 50
        if tem:
            pol.set_context({"frac": frac, "estag": est, "gap_cur": 0.0, "ult": ult, "q": 0.5})
        j = pol.select_masked([0, 1, 2, 3])
        r = 1.0 if rr.random() < media(frac, est, j) else 0.0
        pol.update(j, r * escala)
        ult = 3 if r > 0 else 0
        boas += (j == melhor(frac, est))
    return boas / T


def teste_capacidade():
    T, S = 3000, 8
    fabricas = {   # (fabrica, escala da recompensa: o RP recebe os escores crus sigma, as demais sigma/sigma1)
        "UCB1 (sem estado)": (lambda s: UCB1Masc(4, C=1.0, rng=random.Random(s)), 1.0),
        "RP (sem estado)": (lambda s: RopkePisingerPolitica(4, random.Random(s), segmento=100, rho=0.1), 33.0),
        "CtxUCB": (lambda s: CtxUCBPolitica(4, random.Random(s), C=1.0, tau=25, n_fases=2), 1.0),
        "LinUCB": (lambda s: LinUCBPolitica(4, random.Random(s), alpha=0.5, lam=1.0), 1.0),
        "QLearn": (lambda s: QLearnPolitica(4, random.Random(s), alpha=0.2, gamma=0.0, eps0=0.3, tau=25, n_fases=2), 1.0),
    }
    res = {}
    for amb in AMBIENTES:
        res[amb] = {nome: float(np.mean([jogar(f(s), T, 1000 + s, amb, esc) for s in range(S)])) for nome, (f, esc) in fabricas.items()}
        print(f"(4) {amb}: fracao de jogadas do melhor braco (T = 3000, 8 repeticoes):", {k: round(v, 3) for k, v in res[amb].items()})
    a, b = res["troca de fase"], res["estagnacao alternante"]
    for nome in ("CtxUCB", "LinUCB", "QLearn"):
        assert a[nome] > 0.7, ("troca de fase", nome, a)
        assert b[nome] > max(b["UCB1 (sem estado)"], b["RP (sem estado)"]) + 0.15, ("estagnacao", nome, b)
    assert max(b["UCB1 (sem estado)"], b["RP (sem estado)"]) < 0.65, b
    print("     as tres politicas com estado seguem a dependencia alternante do estado (>= 0,15 acima das sem estado) e a troca de fase (> 0,7)")


# ------------------------------------------------------------------------------------------------ (5) gancho
class Espiao(RopkePisingerPolitica):
    def __init__(self, k, rng):
        super().__init__(k, rng)
        self.ctxs, self.marca = [], []

    def set_context(self, ctx):
        self.ctxs.append(dict(ctx))

    def select_masked(self, allowed):
        self.marca.append(len(self.ctxs))
        return super().select_masked(allowed)


class Mudo(RopkePisingerPolitica):
    def set_context(self, ctx):
        pass


def teste_gancho():
    from run_aos_study import _instancia
    from run_aos_etapa3 import _proxy
    import instance as _inst
    nome = sorted(i.name for i in _inst.load_all(ROOT / "instances" / "SYN_TUNE"))[10]
    inst, cws = _instancia("SYN_TUNE", nome)
    px = _proxy("SYN_TUNE", nome)
    orc = 150 * inst.n_customers ** 2
    saidas = {}
    for rotulo, cls in (("simples", RopkePisingerPolitica), ("mudo", Mudo), ("espiao", Espiao)):
        pd_ = cls(len(DESTRUIR), random.Random(21))
        pr_ = cls(len(REPARAR), random.Random(22)) if cls is not Espiao else RopkePisingerPolitica(len(REPARAR), random.Random(22))
        px.chamadas = 0
        melhor, info = rodar_alns(px, [list(r) for r in cws], random.Random(23), orc, pd_, "raw", pr_, "raw", lambda: px.chamadas)
        saidas[rotulo] = (px.solution_cost(melhor), info["iteracoes"], tuple(info["uso_d"]), tuple(info["uso_r"]), px.chamadas)
        if cls is Espiao:
            esp = pd_
    assert saidas["simples"] == saidas["mudo"] == saidas["espiao"], saidas
    ctxs = esp.ctxs
    assert len(ctxs) == saidas["simples"][1]
    assert esp.marca == list(range(1, len(ctxs) + 1))                 # set_context antes de cada selecao
    fr = [c["frac"] for c in ctxs]
    assert all(0.0 <= x <= 1.0 for x in fr) and all(b >= a for a, b in zip(fr, fr[1:]))
    assert all(c["ult"] in (0, 1, 2, 3) and 0.0 <= c["q"] <= 1.0 and c["estag"] >= 0 for c in ctxs)
    # estagnacao: zera exatamente quando a iteracao ANTERIOR foi uma nova melhor (ult = 3) e, caso contrario, cresce de 1
    assert ctxs[0]["estag"] == 0 and ctxs[0]["ult"] == 0
    n_melhoras = 0
    for a, b in zip(ctxs, ctxs[1:]):
        assert (b["ult"] == 3) == (b["estag"] == 0)
        if b["ult"] != 3:
            assert b["estag"] == a["estag"] + 1
        n_melhoras += b["ult"] == 3
    assert n_melhoras >= 1 and all(c["gap_cur"] >= -1e-12 for c in ctxs)
    print(f"(5) gancho: {len(ctxs)} iteracoes, set_context antes de cada selecao, frac em [0,1] nao decrescente; "
          f"a execucao com gancho (espiao ou mudo) e identica a sem gancho (custo, iteracoes, usos, avaliacoes)")


def main():
    teste_ctxucb_igual_ucb1()
    teste_tilt_igual_rp()
    teste_tilt_rho0_igual_estatica()
    teste_fases_e_loglin()
    teste_linucb()
    teste_qlearn()
    teste_capacidade()
    teste_gancho()
    print("TESTES DAS POLITICAS COM ESTADO: OK")


if __name__ == "__main__":
    main()
