"""
aos_policies.py
===============
Políticas CLÁSSICAS de seleção adaptativa de operadores (AOS) para o
estudo controlado da Direção A (diário, Entradas 62 e 64 -- pré-registro).
Todas expõem a MESMA interface de `HGUCBBandit` usada por `bandit_vns()`
(`select`, `update`, `q_values`, `unblocked_pool`, `n`, `b_t`,
`reset_drift`), então entram como `bandit=` sem nenhuma mudança no loop.

Contrato com `bandit_vns` (verificado lendo vns.py):
- `q_values()` devolve o sentinela [1.0]*k -- só para a checagem de parada
  "max(q) < b_t" nunca disparar (mesma convenção de UCB1OnlyBandit em
  tests/ablation_bandits.py). O loop interno então roda SEMPRE até
  `max_inner_iter`, ou seja, exatamente `steps_max * max_inner_iter`
  aplicações de operador -- é isso que dá o orçamento em operações.
- `update()` devolve sempre False (nenhuma destas políticas tem detector de
  drift), então `reset_drift()` só roda na construção.

Recompensa: estas políticas esperam receber, via os hooks
`reward_positive_fn` / `reward_negative_fn` de `bandit_vns`, a melhora
RELATIVA (c' - c'')/c' >= 0 em caso de sucesso e 0 em caso de insucesso
(ver `reward_relativa` / `reward_zero` abaixo) -- NÃO a Eq. 12/13 do
HG-UCB, para não herdar a assimetria de escala do achado #9.

Hiperparâmetros: defaults da literatura, NÃO ajustados nas instâncias de
teste (ver ameaças à validade no pré-registro, Entrada 64).

Também define `gvns_budgeted`: GVNS clássico (ordem fixa, VND) cortado por
número de aplicações de operador, para comparar com as demais no mesmo
orçamento.
"""
import math
import random as _random_mod

from vns import shake


def reward_relativa(sol_global, sol_current, unblocked_count, exec_count):
    """Melhora relativa (c' - c'')/c' (>= 0). Assinatura compatível com
    `reward_positive_fn` de bandit_vns (os dois últimos args são ignorados)."""
    if sol_global <= 0:
        return 0.0
    return max(0.0, (sol_global - sol_current) / sol_global)


def reward_zero(q_prev_chosen, unblocked_count, exec_count):
    """Insucesso -> recompensa 0. Assinatura compatível com `reward_negative_fn`."""
    return 0.0


class _PoliticaBase:
    b_t = 0.2

    def __init__(self, n_operators, rng=None):
        self.k = n_operators
        self.rng = rng if rng is not None else _random_mod.Random()
        self.reset_drift()

    def reset_drift(self):
        self.n = [0] * self.k
        self.s = 0

    def q_values(self):
        return [1.0] * self.k  # sentinela -- ver docstring do módulo

    def unblocked_pool(self, q=None):
        return list(range(self.k))

    def update(self, chosen, reward):
        self.n[chosen] += 1
        return False


class _MediaNormalizada(_PoliticaBase):
    """Base para políticas que usam média amostral de recompensa
    normalizada pelo MÁXIMO já observado (mantém a recompensa em [0,1],
    como UCB1 exige -- evita um espantalho por escala; ver pré-registro)."""

    def reset_drift(self):
        super().reset_drift()
        self._soma = [0.0] * self.k
        self._max_r = 0.0

    def _media(self, j):
        return self._soma[j] / self.n[j] if self.n[j] > 0 else 0.0

    def update(self, chosen, reward):
        if reward > self._max_r:
            self._max_r = reward
        r_norm = reward / self._max_r if self._max_r > 0 else 0.0
        self._soma[chosen] += r_norm
        self.n[chosen] += 1
        return False


class UniformePolitica(_PoliticaBase):
    """Controle não-adaptativo: operador sorteado uniformemente."""

    def select(self):
        self.s += 1
        return self.rng.randrange(self.k)


class UCB1Politica(_MediaNormalizada):
    """UCB1 puro: testa cada operador uma vez (ordem aleatória), depois
    argmax(média_normalizada + C*sqrt(2 ln s / n)), C=2,0 (mesmo C do
    HG-UCB, Tabela do artigo). Sem ε, sem softmax, sem ADWIN."""

    def __init__(self, n_operators, C=2.0, rng=None):
        self.C = C
        super().__init__(n_operators, rng)

    def select(self):
        self.s += 1
        nunca = [j for j in range(self.k) if self.n[j] == 0]
        if nunca:
            return self.rng.choice(nunca)
        scores = [self._media(j) + self.C * math.sqrt(2.0 * math.log(self.s) / self.n[j])
                  for j in range(self.k)]
        melhor = max(scores)
        return self.rng.choice([j for j in range(self.k) if scores[j] == melhor])


class EpsGreedyPolitica(_MediaNormalizada):
    """ε-greedy com média amostral (ε=0,1): testa cada operador uma vez;
    depois, com prob. ε sorteia uniformemente, senão o de maior média
    (empates sorteados)."""

    def __init__(self, n_operators, epsilon=0.1, rng=None):
        self.epsilon = epsilon
        super().__init__(n_operators, rng)

    def select(self):
        self.s += 1
        nunca = [j for j in range(self.k) if self.n[j] == 0]
        if nunca:
            return self.rng.choice(nunca)
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.k)
        medias = [self._media(j) for j in range(self.k)]
        melhor = max(medias)
        return self.rng.choice([j for j in range(self.k) if medias[j] == melhor])


class ProbMatchingPolitica(_PoliticaBase):
    """Probability matching (Thierens 2005): qualidade Q_a atualizada por
    média móvel exponencial só para o operador escolhido
    (Q <- Q + γ(r - Q), γ=0,1); probabilidade
    p_a = p_min + (1 - k*p_min) * Q_a / ΣQ (p_min=0,05); uniforme se ΣQ=0.
    Invariante a escala da recompensa por construção (só entra a razão)."""

    def __init__(self, n_operators, p_min=0.05, gamma=0.1, rng=None):
        self.p_min = p_min
        self.gamma = gamma
        super().__init__(n_operators, rng)

    def reset_drift(self):
        super().reset_drift()
        self._Q = [0.0] * self.k

    def _probs(self):
        soma = sum(self._Q)
        if soma <= 0:
            return [1.0 / self.k] * self.k
        return [self.p_min + (1.0 - self.k * self.p_min) * q / soma for q in self._Q]

    def select(self):
        self.s += 1
        probs = self._probs()
        r = self.rng.random()
        acumulado = 0.0
        for j, p in enumerate(probs):
            acumulado += p
            if r < acumulado:
                return j
        return self.k - 1

    def update(self, chosen, reward):
        self._Q[chosen] += self.gamma * (reward - self._Q[chosen])
        self.n[chosen] += 1
        return False


def gvns_budgeted(instance, initial_routes, rng, budget_ops, neighborhoods):
    """GVNS clássico (Shake + VND de ordem fixa, aceita só se estritamente
    melhor que o melhor até aqui -- mesma regra de `vns.gvns`) cortado ao
    esgotar `budget_ops` aplicações de operador (cada chamada a `best_fn`
    conta 1; o shake NÃO conta, igual nas políticas de bandit). Se o
    orçamento acaba no meio de um VND, o ponto parcial (sempre viável e
    nunca pior que o shake) ainda é candidato a melhor. Devolve o melhor
    ponto visto. `neighborhoods` deve ser a lista (nome, best_fn, random_fn)
    -- pode vir com best_fn contador, ver estudo_aos/run_aos_study.py."""
    x = [list(r) for r in initial_routes]
    best_cost = instance.solution_cost(x)
    usado = 0
    while usado < budget_ops:
        routes = shake(instance, x, rng, neighborhoods)
        l = 0
        while l < len(neighborhoods) and usado < budget_ops:
            usado += 1
            _, best_fn, _ = neighborhoods[l]
            improved, novo = best_fn(instance, routes)
            if improved:
                routes = novo
                l = 0
            else:
                l += 1
        custo = instance.solution_cost(routes)
        if custo < best_cost - 1e-9:
            x = routes
            best_cost = custo
    return x
