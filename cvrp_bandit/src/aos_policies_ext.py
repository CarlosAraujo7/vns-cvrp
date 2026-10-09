"""
aos_policies_ext.py
===================
Politicas ADICIONAIS de selecao adaptativa de operadores da Etapa 6 (diario,
Entrada 91 -- pre-registro). Ampliam o conjunto de familias de AOS testadas
alem de UCB1, epsilon-greedy, probability matching e HG-UCB:

  Exp3Politica               EXP3 (Auer et al., 2002), bandit adversarial
  ThompsonPolitica           Thompson sampling Beta-Bernoulli (sucesso = melhora)
  SWUCBPolitica              UCB com janela deslizante (Garivier & Moulines, 2011)
  AdaptivePursuitPolitica    Adaptive Pursuit (Thierens, 2005)
  EficienciaUCBPolitica      UCB sobre o GANHO POR AVALIACAO DE DISTANCIA
                             (razao de somas, contando o custo das FALHAS)

Todas expoem `select_masked(allowed)`, `update(chosen, reward)`, `reset_drift`,
`n`, `q_values`, `unblocked_pool` (a mesma interface das politicas mascaradas
de aos_vnd.py). A recompensa recebida e a melhora relativa (c'-c'')/c' >= 0
em caso de sucesso e 0 em insucesso (modo "rel"), normalizada aqui pelo
MAXIMO ja observado quando a politica precisa de recompensa em [0,1] (como
UCB1 e epsilon-greedy em aos_policies.py).

A politica de eficiencia precisa do custo (avaliacoes de distancia) de cada
aplicacao, inclusive das que falham: o laco de aos_vnd.py chama
`update_custo(chosen, reward, calls)` quando `usa_custo` e verdadeiro.

Hiperparametros: defaults da literatura / pontos medios das grades da Etapa 4;
NAO ajustados (pre-registro da Etapa 6).
"""
import math
import random as _random_mod
from collections import deque


class _Base:
    b_t = 0.2
    usa_custo = False

    def __init__(self, n_operators, rng=None):
        self.k = n_operators
        self.rng = rng if rng is not None else _random_mod.Random()
        self.reset_drift()

    def reset_drift(self):
        self.n = [0] * self.k
        self.s = 0
        self._max_r = 0.0

    def q_values(self):
        return [1.0] * self.k

    def unblocked_pool(self, q=None):
        return list(range(self.k))

    def _norm(self, reward):
        if reward > self._max_r:
            self._max_r = reward
        return reward / self._max_r if self._max_r > 0 else 0.0

    def _roleta(self, allowed, pesos):
        total = sum(pesos)
        r = self.rng.random() * total
        acum = 0.0
        for j, w in zip(allowed, pesos):
            acum += w
            if r < acum:
                return j
        return allowed[-1]

    def update(self, chosen, reward):
        self.n[chosen] += 1
        return False


class Exp3Politica(_Base):
    """EXP3: p_j = (1-g) w_j/sum(w) + g/k (sobre os permitidos); apos a
    aplicacao, w_j <- w_j exp(g * (r/p_j) / k), r em [0,1] (normalizada pelo
    maximo observado). Pesos em escala logaritmica (estabilidade numerica)."""

    def __init__(self, n_operators, gamma=0.1, rng=None):
        self.gamma = gamma
        super().__init__(n_operators, rng)

    def reset_drift(self):
        super().reset_drift()
        self._lw = [0.0] * self.k
        self._p_last = 1.0
        self._k_eff = self.k

    def probs(self, allowed):
        m = max(self._lw[j] for j in allowed)
        w = [math.exp(self._lw[j] - m) for j in allowed]
        s = sum(w)
        k_eff = len(allowed)
        return [(1.0 - self.gamma) * x / s + self.gamma / k_eff for x in w]

    def select_masked(self, allowed):
        self.s += 1
        allowed = list(allowed)
        p = self.probs(allowed)
        j = self._roleta(allowed, p)
        self._p_last = p[allowed.index(j)]
        self._k_eff = len(allowed)
        return j

    def update(self, chosen, reward):
        r = self._norm(reward)
        self._lw[chosen] += self.gamma * (r / self._p_last) / self._k_eff
        self.n[chosen] += 1
        return False


class ThompsonPolitica(_Base):
    """Thompson sampling Beta-Bernoulli: sucesso = a aplicacao melhorou a
    solucao (recompensa > 0). Sorteia theta_j ~ Beta(1+s_j, 1+f_j) e escolhe o maior."""

    def reset_drift(self):
        super().reset_drift()
        self._suc = [0] * self.k
        self._fal = [0] * self.k

    def select_masked(self, allowed):
        self.s += 1
        draws = {j: self.rng.betavariate(1 + self._suc[j], 1 + self._fal[j]) for j in allowed}
        melhor = max(draws.values())
        return self.rng.choice([j for j in allowed if draws[j] == melhor])

    def update(self, chosen, reward):
        if reward > 0:
            self._suc[chosen] += 1
        else:
            self._fal[chosen] += 1
        self.n[chosen] += 1
        return False


class SWUCBPolitica(_Base):
    """UCB com janela deslizante (SW-UCB): media e contagem sobre as ultimas
    `window` aplicacoes; indice = media_norm + sqrt(xi ln(min(t, window)) / N_window(j))."""

    def __init__(self, n_operators, window=200, xi=0.6, rng=None):
        self.window = window
        self.xi = xi
        super().__init__(n_operators, rng)

    def reset_drift(self):
        super().reset_drift()
        self._hist = deque()
        self._soma = [0.0] * self.k
        self._cnt = [0] * self.k

    def select_masked(self, allowed):
        self.s += 1
        sem = [j for j in allowed if self._cnt[j] == 0]
        if sem:
            return self.rng.choice(sem)
        t = min(self.s, self.window)
        m = self._max_r if self._max_r > 0 else 1.0
        score = {j: (self._soma[j] / self._cnt[j]) / m + math.sqrt(self.xi * math.log(max(t, 2)) / self._cnt[j])
                 for j in allowed}
        melhor = max(score.values())
        return self.rng.choice([j for j in allowed if score[j] == melhor])

    def update(self, chosen, reward):
        if reward > self._max_r:
            self._max_r = reward
        self._hist.append((chosen, reward))
        self._soma[chosen] += reward
        self._cnt[chosen] += 1
        if len(self._hist) > self.window:
            a, rw = self._hist.popleft()
            self._soma[a] = max(0.0, self._soma[a] - rw)
            self._cnt[a] -= 1
        self.n[chosen] += 1
        return False


class AdaptivePursuitPolitica(_Base):
    """Adaptive Pursuit: Q_j media movel exponencial da recompensa normalizada
    (so do operador aplicado); a probabilidade do operador de maior Q persegue
    p_max = 1-(k-1)p_min, as demais perseguem p_min, ambas com taxa beta."""

    def __init__(self, n_operators, alpha=0.1, beta=0.1, p_min=0.05, rng=None):
        self.alpha, self.beta, self.p_min = alpha, beta, p_min
        super().__init__(n_operators, rng)

    def reset_drift(self):
        super().reset_drift()
        self._Q = [0.0] * self.k
        self._P = [1.0 / self.k] * self.k

    def select_masked(self, allowed):
        self.s += 1
        allowed = list(allowed)
        return self._roleta(allowed, [self._P[j] for j in allowed])

    def update(self, chosen, reward):
        r = self._norm(reward)
        self._Q[chosen] += self.alpha * (r - self._Q[chosen])
        melhor = max(self._Q)
        best = self.rng.choice([j for j in range(self.k) if self._Q[j] == melhor])
        p_max = 1.0 - (self.k - 1) * self.p_min
        for j in range(self.k):
            alvo = p_max if j == best else self.p_min
            self._P[j] += self.beta * (alvo - self._P[j])
        self.n[chosen] += 1
        return False


class EficienciaUCBPolitica(_Base):
    """UCB sobre o ganho relativo por avaliacao de distancia: rho_j = G_j / E_j
    (razao de somas; E_j inclui o custo das aplicacoes que FALHAM, que a
    recompensa sensivel a custo das Etapas 3-4 nao captura). Indice =
    rho_j/max_i rho_i + C sqrt(2 ln s / n_j); cada operador e testado uma vez."""
    usa_custo = True

    def __init__(self, n_operators, C=1.0, rng=None):
        self.C = C
        super().__init__(n_operators, rng)

    def reset_drift(self):
        super().reset_drift()
        self._G = [0.0] * self.k
        self._E = [0.0] * self.k

    def select_masked(self, allowed):
        self.s += 1
        nunca = [j for j in allowed if self.n[j] == 0]
        if nunca:
            return self.rng.choice(nunca)
        rho = {j: (self._G[j] / self._E[j] if self._E[j] > 0 else 0.0) for j in allowed}
        rmax = max(rho.values())
        score = {j: (rho[j] / rmax if rmax > 0 else 0.0) + self.C * math.sqrt(2.0 * math.log(self.s) / self.n[j])
                 for j in allowed}
        melhor = max(score.values())
        return self.rng.choice([j for j in allowed if score[j] == melhor])

    def update_custo(self, chosen, reward, calls):
        self._G[chosen] += reward
        self._E[chosen] += max(calls, 1)
        self.n[chosen] += 1
        return False

    def update(self, chosen, reward):          # sem custo disponivel: trata como 1 avaliacao
        return self.update_custo(chosen, reward, 1)
