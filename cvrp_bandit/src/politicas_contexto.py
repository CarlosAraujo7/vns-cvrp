"""
politicas_contexto.py
=====================
Politicas COM ESTADO do Estudo 18 (diario, Entrada 127 -- pre-registro): selecao de operadores do ALNS que enxerga o ESTADO DA BUSCA
(fracao do orcamento gasta, estagnacao, folga da solucao corrente, resultado da iteracao anterior), alem do historico de recompensas.
Todas expoem a interface das politicas mascaradas (select_masked(allowed), update(chosen, reward), n, b_t, usa_custo) e o metodo extra
set_context(ctx), chamado por alns_loop.rodar_alns no inicio de cada iteracao com
    ctx = {frac, estag, gap_cur, ult, q}   (ver a docstring de rodar_alns).
Nenhuma politica consome o gerador de numeros aleatorios do laco: cada uma usa o seu proprio `rng`.

  Politicas ONLINE (aprendem durante a execucao; recompensa sigma/sigma1 em [0,1]):
    CtxUCBPolitica     UCB1 tabular contextual: um UCB1 independente por contexto (fase do orcamento x estagnacao).
                       Com n_fases = 1 e tau = infinito reproduz EXATAMENTE o UCB1 mascarado (aos_vnd.UCB1Masc).
    LinUCBPolitica     LinUCB disjunto (Li et al., 2010): um modelo linear por operador sobre `caracteristicas(ctx)`.
    QLearnPolitica     Q-learning tabular, estado = (fase, resultado anterior, estagnacao), epsilon-greedy decrescente com o orcamento.
    RPTiltPolitica     "RP sobre o ajuste": Ropke e Pisinger com roleta p_j ~ p0_j * w_j, em que p0 e uma mistura ajustada offline
                       (adaptacao SOBRE o ajuste). Com rho = 0 reproduz EXATAMENTE a mistura estatica p0 (EstaticaPolitica); com p0 uniforme
                       (todos 1,0), o RP classico.
  Politicas OFFLINE com estado (parametros ajustados por busca de politica; nao aprendem durante a execucao):
    FasesMistPolitica  mistura que muda por FASE do orcamento (n_fases x k logits); com fases iguais reproduz a mistura estatica.
    LogLinMistPolitica mistura log-linear no estado: p_j ~ exp(theta_j + a_j (frac - 0,5) + b_j (estag_norm - 0,5)).
"""
import math

import numpy as np

from alns_loop import RopkePisingerPolitica

LOG_ESTAG_MAX = math.log1p(1000.0)      # estagnacao normalizada = log1p(estag) / log1p(1000), truncada em 1
GAP_ESCALA = 0.05                       # folga normalizada = gap_cur / 5%, truncada em 1
D_CAR = 7                               # dimensao de caracteristicas(ctx)


def estag_norm(estag):
    return min(1.0, math.log1p(estag) / LOG_ESTAG_MAX)


def caracteristicas(ctx):
    """Vetor de caracteristicas em [0,1] do estado da busca: [1, fracao do orcamento, estagnacao (log), folga da corrente,
    resultado anterior = aceita pior, = melhora a corrente, = nova melhor]; rejeitada = tudo zero nos tres ultimos."""
    u = ctx["ult"]
    return [1.0, ctx["frac"], estag_norm(ctx["estag"]), min(1.0, max(0.0, ctx["gap_cur"]) / GAP_ESCALA),
            1.0 if u == 1 else 0.0, 1.0 if u == 2 else 0.0, 1.0 if u == 3 else 0.0]


def softmax_np(logits):
    """Probabilidades por softmax (a mesma conta da busca estatica do Estudo 16: exp, divisao pela soma, tolist)."""
    w = np.exp(np.asarray(logits, float))
    return (w / w.sum()).tolist()


def _roleta(rng, allowed, pesos):
    r = rng.random() * sum(pesos)
    acum = 0.0
    for j, w in zip(allowed, pesos):
        acum += w
        if r < acum:
            return j
    return allowed[-1]


class _BaseCtx:
    b_t = 0.2
    usa_custo = False

    def __init__(self, k, rng):
        self.k, self.rng = k, rng
        self.n = [0] * k

    def reset_drift(self):          # interface (o laco do ALNS nao a usa)
        pass

    def set_context(self, ctx):
        pass


# ------------------------------------------------------------------------------------------------ online
class CtxUCBPolitica(_BaseCtx):
    """UCB1 tabular contextual. Contexto = (fase = min(n_fases - 1, int(frac n_fases)), estagnacao > tau). Cada contexto tem as suas
    contagens, somas de recompensa (normalizada pelo maximo ja observado, como UCB1Masc) e relogio ln s. Em cada contexto, cada operador
    e testado uma vez (ordem aleatoria) antes de usar o indice media + C sqrt(2 ln s / n)."""

    def __init__(self, k, rng, C=1.0, tau=100, n_fases=3):
        super().__init__(k, rng)
        self.C, self.tau, self.n_fases = C, tau, n_fases
        self._cnt, self._soma, self._s = {}, {}, {}
        self._max_r = 0.0
        self._ctx = (0, 0)

    def set_context(self, ctx):
        self._ctx = (min(self.n_fases - 1, int(ctx["frac"] * self.n_fases)), 0 if ctx["estag"] <= self.tau else 1)

    def _tab(self):
        c = self._ctx
        if c not in self._cnt:
            self._cnt[c], self._soma[c], self._s[c] = [0] * self.k, [0.0] * self.k, 0
        return c

    def select_masked(self, allowed):
        c = self._tab()
        self._s[c] += 1
        cnt, soma = self._cnt[c], self._soma[c]
        nunca = [j for j in allowed if cnt[j] == 0]
        if nunca:
            return self.rng.choice(nunca)
        s = self._s[c]
        scores = {j: soma[j] / cnt[j] + self.C * math.sqrt(2.0 * math.log(s) / cnt[j]) for j in allowed}
        melhor = max(scores.values())
        return self.rng.choice([j for j in allowed if scores[j] == melhor])

    def update(self, chosen, reward):
        if reward > self._max_r:
            self._max_r = reward
        r_norm = reward / self._max_r if self._max_r > 0 else 0.0
        c = self._ctx
        self._soma[c][chosen] += r_norm
        self._cnt[c][chosen] += 1
        self.n[chosen] += 1
        return False


class LinUCBPolitica(_BaseCtx):
    """LinUCB disjunto: para cada operador j, A_j = lam I + sum x x', b_j = sum r x; indice theta_j x + alpha sqrt(x' A_j^-1 x),
    theta_j = A_j^-1 b_j. A inversa e mantida por Sherman-Morrison. Recompensa sigma/sigma1 em [0,1] (sem normalizar)."""

    def __init__(self, k, rng, alpha=0.5, lam=1.0):
        super().__init__(k, rng)
        self.alpha, self.lam = alpha, lam
        self.Ainv = [np.eye(D_CAR) / lam for _ in range(k)]
        self.b = [np.zeros(D_CAR) for _ in range(k)]
        self.x = np.zeros(D_CAR)

    def set_context(self, ctx):
        self.x = np.array(caracteristicas(ctx))

    def indice(self, j):
        x, Ai = self.x, self.Ainv[j]
        return float((Ai @ self.b[j]) @ x) + self.alpha * math.sqrt(max(0.0, float(x @ Ai @ x)))

    def select_masked(self, allowed):
        sc = {j: self.indice(j) for j in allowed}
        melhor = max(sc.values())
        return self.rng.choice([j for j in allowed if sc[j] == melhor])

    def update(self, chosen, reward):
        x, Ai = self.x, self.Ainv[chosen]
        Ax = Ai @ x
        self.Ainv[chosen] = Ai - np.outer(Ax, Ax) / (1.0 + float(x @ Ax))
        self.b[chosen] = self.b[chosen] + reward * x
        self.n[chosen] += 1
        return False


class QLearnPolitica(_BaseCtx):
    """Q-learning tabular. Estado = (fase, resultado da iteracao anterior, estagnacao > tau); acoes = operadores; recompensa sigma/sigma1.
    Q(s,a) <- Q(s,a) + alpha (r + gamma max_a' Q(s',a') - Q(s,a)); o proximo estado s' e conhecido na chamada seguinte de set_context.
    epsilon-greedy com epsilon = max(eps_min, eps0 (1 - frac)); Q inicial 0."""

    def __init__(self, k, rng, alpha=0.1, gamma=0.5, eps0=0.2, tau=100, n_fases=3, eps_min=0.02):
        super().__init__(k, rng)
        self.alpha, self.gamma, self.eps0, self.tau, self.n_fases, self.eps_min = alpha, gamma, eps0, tau, n_fases, eps_min
        self.Q = {}
        self._pend = None
        self._st = (0, 0, 0)
        self._frac = 0.0

    def _q(self, st):
        if st not in self.Q:
            self.Q[st] = [0.0] * self.k
        return self.Q[st]

    def set_context(self, ctx):
        st = (min(self.n_fases - 1, int(ctx["frac"] * self.n_fases)), ctx["ult"], 0 if ctx["estag"] <= self.tau else 1)
        if self._pend is not None:
            s, a, r = self._pend
            alvo = r + self.gamma * max(self._q(st))
            qs = self._q(s)
            qs[a] += self.alpha * (alvo - qs[a])
            self._pend = None
        self._st, self._frac = st, ctx["frac"]

    def select_masked(self, allowed):
        eps = max(self.eps_min, self.eps0 * (1.0 - self._frac))
        q = self._q(self._st)
        if self.rng.random() < eps:
            return self.rng.choice(allowed)
        melhor = max(q[j] for j in allowed)
        return self.rng.choice([j for j in allowed if q[j] == melhor])

    def update(self, chosen, reward):
        self._pend = (self._st, chosen, reward)
        self.n[chosen] += 1
        return False


class RPTiltPolitica(RopkePisingerPolitica):
    """RP sobre o ajuste: roleta com pesos p0_j * w_j (w = pesos adaptativos de Ropke e Pisinger; p0 = mistura ajustada offline, com
    soma k para que p0 uniforme seja o vetor de 1,0). Recompensa 'raw' (escores sigma), como o RP."""

    def __init__(self, k, rng, p0, segmento=100, rho=0.1):
        super().__init__(k, rng, segmento=segmento, rho=rho)
        s = float(sum(p0))
        self.p0 = [float(x) * k / s for x in p0]

    def select_masked(self, allowed):
        pesos = [self.p0[j] * self.w[j] for j in allowed]
        return _roleta(self.rng, list(allowed), pesos)


# ------------------------------------------------------------------------------------------------ offline com estado
class FasesMistPolitica(_BaseCtx):
    """Mistura por FASE do orcamento: logits[f][j], f = min(n_fases - 1, int(frac n_fases)). Ignora as recompensas."""

    def __init__(self, k, rng, logits):
        super().__init__(k, rng)
        self.n_fases = len(logits)
        self.P = [softmax_np(row) for row in logits]
        self._f = 0

    def set_context(self, ctx):
        self._f = min(self.n_fases - 1, int(ctx["frac"] * self.n_fases))

    def select_masked(self, allowed):
        return _roleta(self.rng, list(allowed), [self.P[self._f][j] for j in allowed])

    def update(self, chosen, reward):
        self.n[chosen] += 1
        return False


class LogLinMistPolitica(_BaseCtx):
    """Mistura log-linear: p_j ~ exp(theta_j + a_j (frac - 0,5) + b_j (estag_norm - 0,5)) entre os permitidos. Ignora as recompensas."""

    def __init__(self, k, rng, theta, a, b):
        super().__init__(k, rng)
        self.theta, self.a, self.b = list(map(float, theta)), list(map(float, a)), list(map(float, b))
        self._u1, self._u2 = 0.0, 0.0

    def set_context(self, ctx):
        self._u1, self._u2 = ctx["frac"] - 0.5, estag_norm(ctx["estag"]) - 0.5

    def probabilidades(self, allowed):
        lg = [self.theta[j] + self.a[j] * self._u1 + self.b[j] * self._u2 for j in allowed]
        m = max(lg)
        w = [math.exp(x - m) for x in lg]
        s = sum(w)
        return [x / s for x in w]

    def select_masked(self, allowed):
        return _roleta(self.rng, list(allowed), self.probabilidades(allowed))

    def update(self, chosen, reward):
        self.n[chosen] += 1
        return False
