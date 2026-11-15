"""
alns_loop_rrt.py
================
Laco do ALNS do Estudo 22 (diario, Entrada 138 -- pre-registro): copia EXATA de `alns_loop.rodar_alns` com UM parametro a mais, a regra de aceitacao. Existe como
modulo separado para que o laco dos Estudos 9 a 21 (em uso pelas filas de execucao) nao seja tocado.

  aceitacao = "sa"   recozimento simulado (o do Estudo 9): aceita o candidato pior com probabilidade exp(-(c_new - c_cur) / T), T = T0 * tf^frac. Com este valor o
                     laco e IDENTICO ao de alns_loop.py (mesmos numeros aleatorios, mesmas decisoes): tests/test_alns_loop_rrt.py confere execucao a execucao.
  aceitacao = "rrt"  record-to-record travel (Dueck e Scheuer 1990; Schrimpf et al. 2000, o ruin-and-recreate original): aceita o candidato se
                     c_new < best * (1 + delta0 * (1 - frac)), com frac a fracao do orcamento gasta (o desvio cai linearmente a zero); nao sorteia numero aleatorio.
A pontuacao dos operadores e a mesma: sigma1 (nova melhor), sigma2 (melhora a corrente), sigma3 (candidato pior aceito).
"""
import math

from alns import Estado, DESTRUIR, REPARAR
from alns_loop import SIGMA1, SIGMA2, SIGMA3, qrange
from bandit import reward_positive_normalized, reward_negative


def rodar_alns_aceitacao(inst, rotas0, rng, orcamento, pol_d, modo_d, pol_r, modo_r, medida, w0=0.05, tf_fator=1e-3, max_rotas=None, on_iteracao=None,
                         destruir=None, reparar=None, estado_cls=None, aceitacao="sa", rrt_delta0=0.0):
    assert aceitacao in ("sa", "rrt")
    DESTRUIR_, REPARAR_ = (DESTRUIR if destruir is None else destruir), (REPARAR if reparar is None else reparar)
    n = inst.n_customers
    qmin, qmax = qrange(n)
    D = inst._D
    dmax = max(max(row) for row in D) or 1.0
    cur = [list(r) for r in rotas0 if r]
    cur_c = inst.solution_cost(cur)
    best, best_c = [list(r) for r in cur], cur_c
    T0 = w0 * cur_c / math.log(2.0)
    kd, kr = len(DESTRUIR_), len(REPARAR_)
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
        est = (Estado if estado_cls is None else estado_cls)(inst, cur, dmax)
        m0 = medida()
        removidos = DESTRUIR_[jd][1](est, q, rng) if jd != 3 else DESTRUIR_[jd][1](est, q, rng, qmax=qmax)
        m1 = medida()
        ok = REPARAR_[jr][1](est, removidos, rng, max_rotas)
        m2 = medida()
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
                elif (c_new - cur_c < -T * math.log(max(rng.random(), 1e-12))) if aceitacao == "sa" \
                        else (c_new < best_c * (1.0 + rrt_delta0 * (1.0 - frac))):
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
    return best, {"iteracoes": it, "inviaveis": inviaveis, "uso_d": nd, "uso_r": nr, "score_d": sucd, "score_r": sucr, "calls_d": calld, "calls_r": callr}
