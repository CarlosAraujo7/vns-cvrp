"""
aos_vnd.py
==========
Loop generico da Etapa 2 da Direcao A (diario, Entrada 71 -- pre-registro).
Um unico laco parametrizado por tres fatores, para decompor por que o GVNS
classico supera o loop do Algoritmo 3 (2A) e para testar selecao adaptativa
DENTRO do VND (2B):

  politica   : quem escolhe o proximo operador (FO, U, ucb1, eps, pm, hgucb)
  aceitacao  : "best"   -> x <- r se cost(r) < melhor global (GVNS classico)
               "shaken" -> x <- r se cost(r) < cost(x') (loop do Algoritmo 3)
  termino    : "LO"  -> para o laco interno quando os k operadores falharam
                        desde a ultima melhora (otimo local, como o VND)
               "F40" -> para apos `max_inner` aplicacoes fixas (Algoritmo 3)

Orcamento: `budget` aplicacoes de operador (chamadas a best_fn), cortado em
B exato mesmo no meio do laco interno. O shake nao conta (igual Etapa 1).

Controles de regressao exata contra a Etapa 1 (ver pre-registro):
  FO/best/LO    == aos_policies.gvns_budgeted      (braco `gvns_classico`)
  U/shaken/F40  == bandit_vns + UniformePolitica   (braco `uniforme`)

As politicas aqui sao versoes MASCARADAS (select_masked(allowed)) das
politicas de aos_policies.py / HGUCBBandit; as classes originais NAO sao
alteradas -- as mascaradas as estendem por heranca.
"""
import math

from vns import shake
from bandit import HGUCBBandit, reward_positive_normalized, reward_negative
from aos_policies import (UniformePolitica, UCB1Politica, EpsGreedyPolitica,
                          ProbMatchingPolitica, reward_relativa)
from aos_policies_ext import (Exp3Politica, ThompsonPolitica, SWUCBPolitica, AdaptivePursuitPolitica,
                              EficienciaUCBPolitica)


class OrdemFixaPolitica:
    """VND classico: menor indice entre os permitidos."""
    b_t = 0.2

    def __init__(self, n_operators, rng=None):
        self.k = n_operators
        self.n = [0] * n_operators

    def select_masked(self, allowed):
        return min(allowed)

    def update(self, chosen, reward):
        self.n[chosen] += 1
        return False

    def reset_drift(self):
        pass


class OrdemFixaListaPolitica:
    """Ordem fixa dada por uma lista literal `ordem` (Etapa 3: ordem por
    custo). Percorre a lista ciclicamente a partir de um ponteiro: volta ao
    inicio numa melhora (reward > 0) e avanca uma posicao numa falha. No
    termino LO (so operadores nao falhos) e no R (todos permitidos) o
    comportamento e o do VND clássico com essa ordem."""
    b_t = 0.2

    def __init__(self, ordem, rng=None):
        self.ordem = list(ordem)
        self.k = len(self.ordem)
        self.n = [0] * self.k
        self.ptr = 0

    def select_masked(self, allowed):
        s = set(allowed)
        for off in range(self.k):
            j = self.ordem[(self.ptr + off) % self.k]
            if j in s:
                return j
        raise RuntimeError("nenhum operador permitido")

    def update(self, chosen, reward):
        self.n[chosen] += 1
        self.ptr = 0 if reward > 0 else (self.ptr + 1) % self.k
        return False

    def reset_drift(self):
        pass


class EstaticaPolitica:
    """Mistura ESTATICA de operadores (Etapa 4): amostra o operador com
    probabilidade proporcional a `pesos[j]` entre os permitidos, sem
    aprendizagem. Os pesos sao ajustados OFFLINE (diario, Entrada 83)."""
    b_t = 0.2

    def __init__(self, pesos, rng=None):
        import random as _r
        self.pesos = list(pesos)
        self.k = len(self.pesos)
        self.n = [0] * self.k
        self.rng = rng if rng is not None else _r.Random()

    def select_masked(self, allowed):
        w = [self.pesos[j] for j in allowed]
        r = self.rng.random() * sum(w)
        acum = 0.0
        for j, wj in zip(allowed, w):
            acum += wj
            if r < acum:
                return j
        return allowed[-1]

    def update(self, chosen, reward):
        self.n[chosen] += 1
        return False

    def reset_drift(self):
        pass


class UniformeMasc(UniformePolitica):
    def select_masked(self, allowed):
        self.s += 1
        return self.rng.choice(allowed)


class UCB1Masc(UCB1Politica):
    def select_masked(self, allowed):
        self.s += 1
        nunca = [j for j in allowed if self.n[j] == 0]
        if nunca:
            return self.rng.choice(nunca)
        scores = {j: self._media(j) + self.C * math.sqrt(2.0 * math.log(self.s) / self.n[j])
                  for j in allowed}
        melhor = max(scores.values())
        return self.rng.choice([j for j in allowed if scores[j] == melhor])


class EpsGreedyMasc(EpsGreedyPolitica):
    def select_masked(self, allowed):
        self.s += 1
        nunca = [j for j in allowed if self.n[j] == 0]
        if nunca:
            return self.rng.choice(nunca)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(allowed)
        medias = {j: self._media(j) for j in allowed}
        melhor = max(medias.values())
        return self.rng.choice([j for j in allowed if medias[j] == melhor])


class ProbMatchingMasc(ProbMatchingPolitica):
    def select_masked(self, allowed):
        self.s += 1
        probs = self._probs()
        pesos = [probs[j] for j in allowed]
        r = self.rng.random() * sum(pesos)
        acumulado = 0.0
        for j, w in zip(allowed, pesos):
            acumulado += w
            if r < acumulado:
                return j
        return allowed[-1]


class HGUCBMasc(HGUCBBandit):
    """HG-UCB com selecao restrita aos operadores permitidos. Mesma logica de
    HGUCBBandit.select (cold-start, filtro b_t, epsilon-greedy, UCB1 com
    q_values), so que sobre `allowed`."""

    def select_masked(self, allowed):
        self.s += 1
        nunca = [j for j in allowed if self.n[j] == 0]
        if nunca:
            return self.rng.choice(nunca)
        q = self.q_values()
        pool = [j for j in self.unblocked_pool(q) if j in allowed] or list(allowed)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(pool)
        p_hat = self.p_hat(q)

        def ucb_score(j):
            return p_hat[j] + self.C * math.sqrt(2.0 * math.log(self.s) / self.n[j])

        return max(pool, key=ucb_score)


def construir_politica(nome, k, rng, custo=False, ordem=None, params=None):
    """Devolve (politica, modo_recompensa). `custo=True` troca o modo de
    recompensa para a variante sensivel a custo da Etapa 3 (diario, Entrada
    77): 'rel' -> 'rel_custo', 'hgucb' -> 'hgucb_custo'. `ordem` so para FOC."""
    if nome == "FO":
        return OrdemFixaPolitica(k, rng), "rel"
    if nome == "FOC":
        return OrdemFixaListaPolitica(ordem, rng), "rel"
    if nome == "U":
        return UniformeMasc(k, rng=rng), "rel"
    params = params or {}
    if nome == "static":
        return EstaticaPolitica(params["pesos"], rng=rng), "rel"
    if nome == "ucb1":
        return UCB1Masc(k, C=params.get("C", 2.0), rng=rng), "rel_custo" if custo else "rel"
    if nome == "eps":
        return EpsGreedyMasc(k, rng=rng), "rel_custo" if custo else "rel"
    if nome == "pm":
        return (ProbMatchingMasc(k, p_min=params.get("p_min", 0.05), gamma=params.get("gamma", 0.1), rng=rng),
                "rel_custo" if custo else "rel")
    if nome == "hgucb":
        return (HGUCBMasc(n_operators=k, rng=rng, p_hat_mode="q_values", alpha=params.get("alpha", 0.5),
                          C=params.get("C", 2.0)),
                "hgucb_custo" if custo else "hgucb")
    # Etapa 6 (diario, Entrada 91): familias adicionais, sempre com recompensa "rel" (nao ha variante cost-aware)
    if nome == "exp3":
        return Exp3Politica(k, gamma=params.get("gamma", 0.1), rng=rng), "rel"
    if nome == "ts":
        return ThompsonPolitica(k, rng=rng), "rel"
    if nome == "swucb":
        return SWUCBPolitica(k, window=params.get("window", 200), xi=params.get("xi", 0.6), rng=rng), "rel"
    if nome == "ap":
        return (AdaptivePursuitPolitica(k, alpha=params.get("alpha", 0.1), beta=params.get("beta", 0.1),
                                        p_min=params.get("p_min", 0.05), rng=rng), "rel")
    if nome == "eff":
        return EficienciaUCBPolitica(k, C=params.get("C", 1.0), rng=rng), "rel"
    raise ValueError(nome)


def vnd_generico(instance, initial_routes, rng, budget, neighborhoods, politica,
                 aceitacao, termino, modo_recompensa="rel", max_inner=40,
                 neighborhoods_shake=None, medida=None, paciencia=5, on_aplicacao=None):
    """Retorna (best_routes, info) com info = {shakes, usado, paciencia_fim}.

    Parametros ADITIVOS da Etapa 3 (os defaults preservam EXATAMENTE o
    comportamento da Etapa 2):
      neighborhoods_shake : lista usada so pelo shake (default = a mesma de
                            `neighborhoods`). Na Etapa 3 e a lista dos 5
                            operadores originais.
      medida              : callable sem argumentos que devolve o gasto
                            acumulado (avaliacoes de distancia, via
                            InstanciaContadora). Se None, o orcamento `budget`
                            e em APLICACOES de operador (Etapas 1-2). A
                            checagem e antes de cada aplicacao; a aplicacao
                            em curso se completa.
      termino             : "LO" | "F40" | "R" (R = para apos `paciencia`
                            falhas consecutivas, amostragem com reposicao).
      modo_recompensa     : "rel" | "hgucb" | "rel_custo" | "hgucb_custo"
                            (os dois ultimos multiplicam a recompensa de
                            sucesso por f = clip(calls_ref/calls_app, 0.1, 10)).
    """
    assert aceitacao in ("best", "shaken") and termino in ("LO", "F40", "R")
    k = len(neighborhoods)
    nb_shake = neighborhoods if neighborhoods_shake is None else neighborhoods_shake
    x = [list(r) for r in initial_routes]
    best_cost = instance.solution_cost(x)
    best_x = x
    usado = 0
    shakes = 0
    pending_drift = False
    fim_paciencia = 0
    soma_calls, n_calls = 0, 0

    def gasto():
        return usado if medida is None else medida()

    while gasto() < budget:
        x_linha = shake(instance, x, rng, nb_shake)
        shakes += 1
        custo_linha = instance.solution_cost(x_linha)
        r, custo_r = x_linha, custo_linha
        falhos = set()
        interno = 0
        consec = 0
        while gasto() < budget:
            if termino == "LO" and len(falhos) == k:
                break
            if termino == "F40" and interno >= max_inner:
                break
            if termino == "R" and consec >= paciencia:
                fim_paciencia += 1
                break
            permitidos = [j for j in range(k) if j not in falhos] if termino == "LO" else list(range(k))
            if pending_drift:
                politica.reset_drift()
                j = rng.choice(permitidos)
                pending_drift = False
            else:
                j = politica.select_masked(permitidos)
            if modo_recompensa in ("hgucb", "hgucb_custo"):
                q_antes = politica.q_values()
                unblocked = len(politica.unblocked_pool(q_antes))
                exec_count = politica.n[j] + 1
                q_prev = q_antes[j]
            m0 = medida() if medida is not None else 0
            usado += 1
            interno += 1
            _, best_fn, _ = neighborhoods[j]
            melhorou, novo = best_fn(instance, r)
            fator = 1.0
            if modo_recompensa in ("rel_custo", "hgucb_custo"):
                calls_app = medida() - m0
                soma_calls += calls_app
                n_calls += 1
                if calls_app > 0:
                    fator = min(10.0, max(0.1, (soma_calls / n_calls) / calls_app))
            melhora = False
            if melhorou:
                c_novo = instance.solution_cost(novo)
                if c_novo < custo_r - 1e-9:
                    melhora = True
            if on_aplicacao is not None:
                on_aplicacao(j, melhora, (custo_r - c_novo) / custo_r if melhora else 0.0)
            if modo_recompensa in ("rel", "rel_custo"):
                recompensa = reward_relativa(custo_r, c_novo, 0, 0) * fator if melhora else 0.0
            else:
                recompensa = (reward_positive_normalized(custo_r, c_novo, unblocked, exec_count) * fator
                              if melhora else reward_negative(q_prev, unblocked, exec_count))
            if melhora:
                r, custo_r = novo, c_novo
                falhos = set()
                consec = 0
            else:
                falhos.add(j)
                consec += 1
            if getattr(politica, "usa_custo", False):
                # Etapa 6: politica de eficiencia -- precisa do custo da aplicacao, inclusive das que falham
                terminou = politica.update_custo(j, recompensa, (medida() - m0) if medida is not None else 1)
            else:
                terminou = politica.update(j, recompensa)
            if terminou:
                pending_drift = True

        if aceitacao == "best":
            aceita = custo_r < best_cost - 1e-9
        else:
            aceita = custo_r < custo_linha - 1e-9
        if aceita:
            x = r
            if custo_r < best_cost - 1e-9:
                best_cost, best_x = custo_r, r
    return best_x, {"shakes": shakes, "usado": usado, "paciencia_fim": fim_paciencia}
