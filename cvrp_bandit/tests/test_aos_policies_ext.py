#!/usr/bin/env python3
"""
test_aos_policies_ext.py
========================
Testes das politicas da Etapa 6 (src/aos_policies_ext.py) em um bandit
sintetico de 6 bracos com probabilidade de sucesso conhecida e ganho/custo
conhecidos. Verifica: (i) selecao sempre dentro dos permitidos; (ii) cada
politica concentra a escolha no melhor braco acima do nivel uniforme
(criterio de que ela de fato aprende; EXP3 e a mais lenta); (iii) probabilidades
validas (EXP3, Adaptive Pursuit); (iv) a politica de eficiencia identifica o
braco de maior ganho POR CUSTO mesmo quando outro tem maior ganho por
aplicacao. Uso: python tests/test_aos_policies_ext.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aos_policies_ext import (Exp3Politica, ThompsonPolitica, SWUCBPolitica, AdaptivePursuitPolitica,  # noqa: E402
                              EficienciaUCBPolitica)

K = 6
P_SUCESSO = [0.05, 0.05, 0.40, 0.05, 0.05, 0.05]       # braco 2 e o melhor (ganho medio e taxa)
STEPS = 4000


def simular(pol, rng, custo_por_braco=None, ganho=None):
    usos = [0] * K
    for _ in range(STEPS):
        permitidos = list(range(K))
        j = pol.select_masked(permitidos)
        assert j in permitidos
        ok = rng.random() < P_SUCESSO[j]
        g = (ganho[j] if ganho else 0.01) if ok else 0.0
        usos[j] += 1
        if getattr(pol, "usa_custo", False):
            pol.update_custo(j, g, custo_por_braco[j] if custo_por_braco else 1)
        else:
            pol.update(j, g)
    return usos


def main():
    falhas = []
    for nome, fab, minimo in (
            ("Exp3", lambda r: Exp3Politica(K, gamma=0.1, rng=r), 0.25),
            ("Thompson", lambda r: ThompsonPolitica(K, rng=r), 0.60),
            ("SW-UCB", lambda r: SWUCBPolitica(K, rng=r), 0.40),
            ("AdaptivePursuit", lambda r: AdaptivePursuitPolitica(K, rng=r), 0.40),
            ("Eficiencia", lambda r: EficienciaUCBPolitica(K, C=1.0, rng=r), 0.40)):
        fr = []
        for seed in range(5):
            rng = random.Random(seed)
            usos = simular(fab(random.Random(100 + seed)), rng)
            fr.append(usos[2] / sum(usos))
        m = sum(fr) / len(fr)
        ok = m >= minimo
        print(f"{nome:16s} fracao no melhor braco: {m:.2f} (uniforme = {1/K:.2f}; minimo exigido {minimo:.2f}) "
              f"{'OK' if ok else 'FALHOU'}")
        if not ok:
            falhas.append(nome)
    # mascara: so operadores 1 e 3 permitidos
    for fab in (lambda r: Exp3Politica(K, rng=r), lambda r: ThompsonPolitica(K, rng=r),
                lambda r: SWUCBPolitica(K, rng=r), lambda r: AdaptivePursuitPolitica(K, rng=r),
                lambda r: EficienciaUCBPolitica(K, rng=r)):
        pol = fab(random.Random(7))
        for _ in range(300):
            j = pol.select_masked([1, 3])
            assert j in (1, 3)
            if getattr(pol, "usa_custo", False):
                pol.update_custo(j, 0.01 if j == 1 else 0.0, 10)
            else:
                pol.update(j, 0.01 if j == 1 else 0.0)
    print("selecao mascarada: OK")
    # probabilidades validas
    e = Exp3Politica(K, rng=random.Random(1))
    for _ in range(200):
        j = e.select_masked(list(range(K)))
        e.update(j, 0.01 if j == 2 else 0.0)
    p = e.probs(list(range(K)))
    assert abs(sum(p) - 1) < 1e-9 and min(p) >= e.gamma / K - 1e-12
    a = AdaptivePursuitPolitica(K, rng=random.Random(1))
    for _ in range(200):
        j = a.select_masked(list(range(K)))
        a.update(j, 0.01 if j == 2 else 0.0)
    assert abs(sum(a._P) - 1) < 1e-9 and min(a._P) >= a.p_min - 1e-12
    print("probabilidades validas (EXP3, Adaptive Pursuit): OK")
    # eficiencia: braco 0 tem MAIOR ganho por aplicacao mas custa 50x mais; braco 4 ganha por custo
    P_bkp = list(P_SUCESSO)
    P_SUCESSO[:] = [0.30, 0.10, 0.10, 0.10, 0.20, 0.10]
    custos = [5000, 100, 100, 100, 60, 100]
    ganhos = [0.02, 0.01, 0.01, 0.01, 0.01, 0.01]
    fr = []
    for seed in range(5):
        usos = simular(EficienciaUCBPolitica(K, C=1.0, rng=random.Random(200 + seed)), random.Random(seed), custos, ganhos)
        fr.append(usos[4] / sum(usos))
    m = sum(fr) / len(fr)
    P_SUCESSO[:] = P_bkp
    print(f"Eficiencia: fracao no braco de maior ganho por custo: {m:.2f} {'OK' if m > 0.35 else 'FALHOU'}")
    if m <= 0.35:
        falhas.append("Eficiencia-custo")
    # contextual: o operador 3 so tem sucesso alto logo apos um SUCESSO do operador 1 (dependencia sequencial)
    from aos_policies_ext import ContextualThompsonPolitica
    for nome, kw in (("Contextual TS", dict(eficiencia=False)), ("Contextual TS eficiencia", dict(eficiencia=True))):
        fr_ctx, fr_base = [], []
        for seed in range(5):
            for contextual in (True, False):
                rng = random.Random(seed)
                pol = (ContextualThompsonPolitica(K, rng=random.Random(300 + seed), **kw) if contextual
                       else ThompsonPolitica(K, rng=random.Random(300 + seed)))
                prev, apos, tres = (-1, False), 0, 0
                for _ in range(6000):
                    j = pol.select_masked(list(range(K)))
                    p = 0.70 if (j == 3 and prev == (1, True)) else (0.05 if j == 3 else 0.15)
                    ok = rng.random() < p
                    if prev == (1, True):
                        apos += 1
                        tres += j == 3
                    if getattr(pol, "usa_custo", False):
                        pol.update_custo(j, 0.01 if ok else 0.0, 100)
                    else:
                        pol.update(j, 0.01 if ok else 0.0)
                    prev = (j, ok)
                (fr_ctx if contextual else fr_base).append(tres / max(apos, 1))
        mc, mb = sum(fr_ctx) / len(fr_ctx), sum(fr_base) / len(fr_base)
        ok = mc > 0.40 and mc > mb + 0.15
        print(f"{nome:26s} P(operador 3 | apos sucesso do 1): contextual {mc:.2f} vs nao contextual {mb:.2f} "
              f"{'OK' if ok else 'FALHOU'}")
        if not ok:
            falhas.append(nome)
    print("TESTES:", "OK" if not falhas else f"FALHAS {falhas}")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
