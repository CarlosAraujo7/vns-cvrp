#!/usr/bin/env python3
"""
ablation_bandits.py
=====================
Duas classes de bandit SIMPLIFICADAS, usadas SÓ pelo Diagnóstico #3
(tests/diag3_ablation_ucb_softmax.py, Entrada 8 do diário, Achado Real
#3) -- NÃO fazem parte da implementação validada (src/bandit.py) e não
são usadas em nenhuma validação "oficial" da Tarefa #178. Propositalmente
mantidas FORA de src/ para não confundir com o HG-UCB real.

Cada uma isola UM dos dois mecanismos que compõem o "híbrido" do artigo
(UCB1, Eq.5, vs. gradiente de preferência softmax, Eq.11), mantendo o
resto (cold-start uniforme, wrapper ε-greedy Eq.14) idêntico ao
HGUCBBandit real, para que qualquer diferença de resultado seja atribuível
à peça isolada, não a uma combinação de mudanças (mesma disciplina de
tests/compare_phat_variants.py).

Ambas implementam a MESMA interface mínima que bandit_vns() (vns.py)
espera de um objeto `bandit`: .select(), .update(chosen, reward), .n
(lista), .q_values(), .unblocked_pool(q), .b_t, .reset_drift(). Isso
permite passá-las diretamente como `bandit=` para bandit_vns() sem
nenhuma modificação em vns.py.
"""
import math
import random as _random_mod


class UCB1OnlyBandit:
    """UCB1 CLÁSSICO -- média empírica bruta de recompensa por operador
    (SEM softmax(H), SEM gradiente de preferência) + termo de exploração
    UCB1 padrão (Eq.5, mas com a média empírica no lugar de p_hat). Mesmo
    cold-start e mesmo wrapper ε-greedy (Eq.14) do HGUCBBandit -- só o
    termo de EXPLOITATION muda.

    b_t/pool: sem análogo natural sem softmax(H) -- pool sempre = todos os
    k operadores (filtro b_t desabilitado nesta ablação, documentado
    explicitamente na Entrada 8 do diário). q_values() retorna um
    sentinela [1.0]*k só para satisfazer a interface que bandit_vns()
    espera -- a checagem de parada "max(q_values())<b_t" em vns.py nunca
    dispara aqui por construção (max sentinela=1.0 >= qualquer b_t
    razoável); só o teto max_inner_iter para o loop interno nesta
    variante, resultado esperado e aceitável para um diagnóstico isolado
    de um único mecanismo."""

    def __init__(self, n_operators, C=2.0, epsilon=0.1, b_t=0.2, rng=None):
        self.k = n_operators
        self.C = C
        self.epsilon = epsilon
        self.b_t = b_t
        self.rng = rng if rng is not None else _random_mod.Random()
        self.reset_drift()

    def reset_drift(self):
        self.n = [0] * self.k
        self._reward_sum = [0.0] * self.k
        self.s = 0

    def q_values(self):
        return [1.0] * self.k  # sentinela -- ver docstring da classe

    def unblocked_pool(self, q=None):
        return list(range(self.k))  # filtro b_t desabilitado nesta ablação

    def _media_empirica(self, j):
        return self._reward_sum[j] / self.n[j] if self.n[j] > 0 else 0.0

    def select(self):
        self.s += 1
        never_tried = [j for j in range(self.k) if self.n[j] == 0]
        if never_tried:
            return self.rng.choice(never_tried)
        pool = list(range(self.k))
        if self.rng.random() < self.epsilon:
            return self.rng.choice(pool)

        def ucb_score(j):
            explore = self.C * math.sqrt(2.0 * math.log(self.s) / self.n[j])
            return self._media_empirica(j) + explore

        return max(pool, key=ucb_score)

    def update(self, chosen, reward):
        self.n[chosen] += 1
        self._reward_sum[chosen] += reward
        return False  # sem ADWIN nesta ablação -- isola só o eixo UCB1 vs. softmax


class SoftmaxOnlyBandit:
    """GRADIENT BANDIT CLÁSSICO (Sutton & Barto 2018) -- amostra o
    operador PROPORCIONALMENTE a softmax(H) (Eq.11, mesmo sinal corrigido
    usado em HGUCBBandit -- ver bandit.py), SEM nenhum termo de
    exploração UCB1. Mesmo cold-start, mesmo wrapper ε-greedy, e MESMO
    filtro b_t/pool (renormalizado sobre o pool) do HGUCBBandit -- como
    H/softmax continuam existindo aqui, o filtro b_t tem o mesmo sentido
    que na classe original. Só o termo de SELEÇÃO muda: amostragem
    proporcional em vez de argmax(p_hat + explore_UCB1)."""

    def __init__(self, n_operators, alpha=5.0, epsilon=0.1, b_t=0.2, rng=None):
        self.k = n_operators
        self.alpha = alpha
        self.epsilon = epsilon
        self.b_t = b_t
        self.rng = rng if rng is not None else _random_mod.Random()
        self.reset_drift()

    def reset_drift(self):
        self.H = [0.0] * self.k
        self.n = [0] * self.k
        self.s = 0
        self._reward_sum = 0.0
        self._reward_count = 0

    def q_values(self):
        m = max(self.H)
        exps = [math.exp(h - m) for h in self.H]
        total = sum(exps)
        return [e / total for e in exps]

    def unblocked_pool(self, q=None):
        if q is None:
            q = self.q_values()
        pool = [j for j in range(self.k) if q[j] >= self.b_t]
        return pool if pool else list(range(self.k))

    def select(self):
        self.s += 1
        never_tried = [j for j in range(self.k) if self.n[j] == 0]
        if never_tried:
            return self.rng.choice(never_tried)
        q = self.q_values()
        pool = self.unblocked_pool(q)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(pool)
        # amostragem proporcional a softmax(H), RENORMALIZADA sobre o pool
        # (gradient bandit clássico -- sem termo de exploração UCB1)
        pesos = [q[j] for j in pool]
        total = sum(pesos)
        r = self.rng.random() * total
        acumulado = 0.0
        for j, w in zip(pool, pesos):
            acumulado += w
            if r <= acumulado:
                return j
        return pool[-1]  # salvaguarda de arredondamento de ponto flutuante

    def update(self, chosen, reward):
        r_bar = self._reward_sum / self._reward_count if self._reward_count else 0.0
        pi = self.q_values()
        for a in range(self.k):
            if a == chosen:
                self.H[a] += self.alpha * (reward - r_bar) * (1.0 - pi[a])
            else:
                self.H[a] -= self.alpha * (reward - r_bar) * pi[a]
        self.n[chosen] += 1
        self._reward_sum += reward
        self._reward_count += 1
        return False  # sem ADWIN nesta ablação -- isola só o eixo UCB1 vs. softmax
