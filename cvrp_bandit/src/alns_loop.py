"""
alns_loop.py
============
Laco do ALNS (Estudo 9) e a politica CLASSICA de Ropke & Pisinger (2006): roleta sobre pesos atualizados por
segmentos de 100 iteracoes com os escores sigma1=33 (nova melhor solucao), sigma2=9 (melhora a corrente),
sigma3=13 (aceita pior) e reacao rho=0.1. As demais politicas (U, UCB1, PM, TS, EXP3, HG-UCB, eficiencia) vem de
aos_vnd.construir_politica e recebem a recompensa normalizada sigma/sigma1.

Destruicao e reparo sao escolhidos por duas instancias INDEPENDENTES da mesma politica (como no ALNS original).
Aceitacao: recozimento simulado com temperatura decrescente em funcao da FRACAO DO ORCAMENTO gasto
(T = T0 * (Tf/T0)^fracao), igual para todos os bracos.
"""
import math

from alns import Estado, DESTRUIR, REPARAR
from bandit import reward_positive_normalized, reward_negative

SIGMA1, SIGMA2, SIGMA3 = 33.0, 9.0, 13.0


class RopkePisingerPolitica:
    """Pesos adaptativos de Ropke & Pisinger (2006): roleta proporcional ao peso; a cada `segmento` atualizacoes,
    w_j <- (1-rho) w_j + rho * pi_j / theta_j (se theta_j > 0)."""
    b_t = 0.2
    usa_custo = False

    def __init__(self, k, rng, segmento=100, rho=0.1):
        self.k, self.rng, self.segmento, self.rho = k, rng, segmento, rho
        self.w = [1.0] * k
        self.pi = [0.0] * k
        self.theta = [0] * k
        self.n = [0] * k
        self.cont = 0

    def select_masked(self, allowed):
        tot = sum(self.w[j] for j in allowed)
        r = self.rng.random() * tot
        acum = 0.0
        for j in allowed:
            acum += self.w[j]
            if r < acum:
                return j
        return allowed[-1]

    def update(self, chosen, reward):
        self.n[chosen] += 1
        self.pi[chosen] += reward
        self.theta[chosen] += 1
        self.cont += 1
        if self.cont % self.segmento == 0:
            for j in range(self.k):
                if self.theta[j] > 0:
                    self.w[j] = (1 - self.rho) * self.w[j] + self.rho * self.pi[j] / self.theta[j]
                self.pi[j], self.theta[j] = 0.0, 0
        return False

    def reset_drift(self):
        pass


def construir_politica_alns(nome, k, rng, params=None):
    """(politica, modo) -- modo: 'raw' (escores sigma crus, so RP), 'score' (sigma/sigma1) ou 'hgucb'."""
    from aos_vnd import construir_politica
    params = params or {}
    if nome == "RP":
        return RopkePisingerPolitica(k, rng, segmento=params.get("segmento", 100), rho=params.get("rho", 0.1)), "raw"
    pol, modo = construir_politica(nome, k, rng, custo=False, params=params)
    return pol, ("hgucb" if modo == "hgucb" else "score")


def qrange(n):
    """Faixa do numero de clientes removidos por iteracao."""
    return max(5, math.ceil(0.1 * n)), min(math.ceil(0.3 * n), 60)


def rodar_alns(inst, rotas0, rng, orcamento, pol_d, modo_d, pol_r, modo_r, medida, w0=0.05, tf_fator=1e-3,
               max_rotas=None, on_iteracao=None):
    """Executa o ALNS ate `orcamento` avaliacoes de distancia (checado antes de cada iteracao).
    Retorna (melhor_solucao, info)."""
    n = inst.n_customers
    qmin, qmax = qrange(n)
    D = inst._D
    dmax = max(max(row) for row in D) or 1.0
    cur = [list(r) for r in rotas0 if r]
    cur_c = inst.solution_cost(cur)
    best, best_c = [list(r) for r in cur], cur_c
    T0 = w0 * cur_c / math.log(2.0)
    kd, kr = len(DESTRUIR), len(REPARAR)
    nd, nr = [0] * kd, [0] * kr
    sucd, sucr = [0.0] * kd, [0.0] * kr
    calld, callr = [0] * kd, [0] * kr
    it = 0
    inviaveis = 0
    while medida() < orcamento:
        it += 1
        frac = min(1.0, medida() / orcamento)
        T = T0 * (tf_fator ** frac)
        q = rng.randint(qmin, qmax)
        jd = pol_d.select_masked(list(range(kd)))
        jr = pol_r.select_masked(list(range(kr)))
        est = Estado(inst, cur, dmax)
        m0 = medida()
        removidos = DESTRUIR[jd][1](est, q, rng) if jd != 3 else DESTRUIR[jd][1](est, q, rng, qmax=qmax)
        m1 = medida()
        ok = REPARAR[jr][1](est, removidos, rng, max_rotas)
        m2 = medida()
        # Emenda de contabilidade (diario, Entrada 107): cada iteracao paga n + R avaliacoes de distancia pelo custo de
        # avaliar a solucao candidata (recomputo do custo), que antes era gratuito; sem isso, operadores quase gratuitos
        # em avaliacoes (destruir uma rota pequena) rodavam milhoes de iteracoes dentro do mesmo orcamento.
        inst.chamadas += n + len(est.rotas)
        nd[jd] += 1
        nr[jr] += 1
        calld[jd] += m1 - m0
        callr[jr] += m2 - m1
        sigma, c_new, cur_antes = 0.0, None, cur_c
        if ok:
            cand = est.rotas
            viavel, _ = inst.is_feasible(cand)
            if viavel:
                c_new = inst.solution_cost(cand)
                if c_new < best_c - 1e-9:
                    sigma = SIGMA1
                    best, best_c = [list(r) for r in cand], c_new
                    cur, cur_c = [list(r) for r in cand], c_new
                elif c_new < cur_c - 1e-9:
                    sigma = SIGMA2
                    cur, cur_c = [list(r) for r in cand], c_new
                elif c_new - cur_c < -T * math.log(max(rng.random(), 1e-12)):
                    sigma = SIGMA3
                    cur, cur_c = [list(r) for r in cand], c_new
            else:
                inviaveis += 1
        else:
            inviaveis += 1
        if sigma > 0:
            sucd[jd] += sigma
            sucr[jr] += sigma
        for pol, modo, j, calls in ((pol_d, modo_d, jd, m1 - m0), (pol_r, modo_r, jr, m2 - m1)):
            if modo == "raw":
                rec = sigma
            elif modo == "score":
                rec = sigma / SIGMA1
            else:   # hgucb: sucesso = melhora a corrente (nova melhor ou melhora)
                if sigma in (SIGMA1, SIGMA2):
                    rec = reward_positive_normalized(cur_antes, c_new, len(pol.unblocked_pool(pol.q_values())), pol.n[j] + 1)
                else:
                    rec = reward_negative(pol.q_values()[j], len(pol.unblocked_pool(pol.q_values())), pol.n[j] + 1)
            if getattr(pol, "usa_custo", False):
                pol.update_custo(j, rec, max(calls, 1))
            else:
                pol.update(j, rec)
        if on_iteracao is not None:
            on_iteracao(it, jd, jr, sigma)
    return best, {"iteracoes": it, "inviaveis": inviaveis, "uso_d": nd, "uso_r": nr, "score_d": sucd,
                  "score_r": sucr, "calls_d": calld, "calls_r": callr}
