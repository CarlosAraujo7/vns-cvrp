#!/usr/bin/env python3
"""
diag9_reward_rescale.py
==========================
Diagnóstico #9, aprovado pelo pesquisador (via `AskUserQuestion`, respondendo
à Entrada 16 do diário, achado real #9): "Testar reescalar a recompensa
positiva".

Motivação (Entrada 16, achado real #9): o tracing detalhado de
`diag8_bandit_trace_v2.py` (A-n32-k5, seed=1000, alpha=5,0, normalizado)
revelou que a nossa PRÓPRIA normalização da Entrada 12 (dividir rs por
sol_current, Eq. 12) deixou a recompensa POSITIVA uniformemente pequena
(média 0,0146) frente à recompensa NEGATIVA (Eq. 13, nunca normalizada,
média -1,6734) -- uma assimetria de ~115x que não existia da mesma forma
antes da Entrada 12. Com só 8,9% das iterações sendo sucesso e r_bar
arrastado bem negativo pela maioria de falhas, o termo (reward - r_bar) de
um acerto (Eq. 11) é dominado pelo "menos r_bar", não pela magnitude real
da melhora -- tornando o sinal de sucesso quase BINÁRIO na prática.

Testa `reward_positive_normalized_scaled()` (bandit.py, NOVA, wrapper
aditivo sobre reward_positive_normalized() que multiplica o resultado por
um fator fixo `scale`) via o mesmo hook `reward_positive_fn` de
bandit_vns() já usado desde a Entrada 11 -- nenhuma mudança de vns.py ou
HGUCBBandit necessária, só functools.partial() para fixar `scale` por
configuração.

Grade testada: os dois alphas candidatos completos da Entrada 13 (5,0 e
0,5) cruzados com 5 fatores de reescala (1,0 -- reproduz o baseline atual
byte-a-byte, sem NENHUMA mudança, incluído para comparação DIRETA na mesma
rodada e como checagem de regressão extra; 10,0; 50,0; 100,0 -- a sugestão
original do pesquisador, análoga à convenção de "porcentagem" de erro_pct;
200,0 -- ponto ACIMA da razão medida de ~115x, para checar se a relação é
monótona ou tem um ótimo interior), na mesma amostra ampliada de 8
instâncias dos diagnósticos 5-8 (5 de A, B-n41-k6, B-n63-k10, X-n101-k25),
5 seeds, SEED_BASE=1000, progressive_x_linha=True.

Mede, por configuração: gap% (métrica principal), % de iterações com
softmax quase one-hot (max_q>0.99 -- checagem de SEGURANÇA contra
reintroduzir o colapso catastrófico do achado real #4, já que aumentar a
magnitude da recompensa positiva é exatamente o tipo de mudança que
poderia fazer isso), taxa de drift do ADWIN (consistência com a Entrada 14),
e a proporção de magnitude |recompensa negativa média| / |recompensa
positiva média| observada (para confirmar que a reescala de fato reduz a
assimetria de ~115x medida no achado real #9 na direção esperada -- a
pergunta não é só "isso muda o gap", é "isso muda o gap E a assimetria na
direção mecânica esperada").

Uso: python diag9_reward_rescale.py
"""
import functools
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from instance import load_instance  # noqa: E402
from construction import clarke_wright_savings, repair_route_count  # noqa: E402
from vns import bandit_vns  # noqa: E402
from bandit import HGUCBBandit, reward_positive_normalized_scaled  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STEPS_MAX = 20
N_SEEDS = 5
SEED_BASE = 1000

CASOS = [
    ("A-n32-k5", "A", 0.1892),
    ("A-n34-k5", "A", 0.1404),
    ("A-n38-k5", "A", 0.6835),
    ("A-n55-k9", "A", 1.2720),
    ("A-n64-k9", "A", 4.0859),
    ("B-n41-k6", "B", 0.9000),
    ("B-n63-k10", "B", 4.0804),
    ("X-n101-k25", "X", 3.5419),
]

ALPHAS = [5.0, 0.5]
SCALES = [1.0, 10.0, 50.0, 100.0, 200.0]

CONFIGURACOES = [
    (
        f"alpha={alpha:.1f} / scale={scale:.0f}x" + ("(baseline)" if scale == 1.0 else ""),
        alpha, scale,
    )
    for alpha in ALPHAS
    for scale in SCALES
]


def rodar_uma_semente(inst, cws, alpha, scale, seed):
    rng = random.Random(seed)
    bandit_obj = HGUCBBandit(n_operators=5, rng=rng, p_hat_mode="q_values", alpha=alpha)
    reward_fn = functools.partial(reward_positive_normalized_scaled, scale=scale)

    contador = {"colapsadas": 0, "drift": 0, "total": 0,
                "pos_soma": 0.0, "pos_n": 0, "neg_soma": 0.0, "neg_n": 0}

    def on_inner(step, inner_iter, escolhido, melhorou, reward, drift, _b=bandit_obj, _c=contador):
        _c["total"] += 1
        if max(_b.q_values()) > 0.99:
            _c["colapsadas"] += 1
        if drift:
            _c["drift"] += 1
        if reward > 0:
            _c["pos_soma"] += reward
            _c["pos_n"] += 1
        else:
            _c["neg_soma"] += reward
            _c["neg_n"] += 1

    resultado = bandit_vns(
        inst, [list(r) for r in cws], rng, steps_max=STEPS_MAX,
        bandit=bandit_obj, max_inner_iter=40, progressive_x_linha=True,
        reward_positive_fn=reward_fn, on_inner=on_inner,
    )
    return resultado, contador


def main():
    print("=== Diagnóstico 9: reescala da recompensa positiva normalizada (achado real #9) ===\n")
    t_inicio = time.perf_counter()

    resumo_gap = {rotulo: {} for rotulo, _, _ in CONFIGURACOES}
    resumo_extra = {
        rotulo: {"colapsadas": 0, "drift": 0, "total": 0,
                  "pos_soma": 0.0, "pos_n": 0, "neg_soma": 0.0, "neg_n": 0}
        for rotulo, _, _ in CONFIGURACOES
    }

    for name, set_name, alvo_artigo in CASOS:
        path = ROOT / "instances" / set_name / f"{name}.vrp"
        inst = load_instance(path)
        cws = repair_route_count(inst, clarke_wright_savings(inst))

        print(f"--- {name} (conjunto {set_name}, artigo HG-UCB={alvo_artigo:.2f}%) ---")
        for rotulo, alpha, scale in CONFIGURACOES:
            gaps = []
            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                resultado, c = rodar_uma_semente(inst, cws, alpha, scale, seed)
                viavel, motivo = inst.is_feasible(resultado)
                if not viavel:
                    print(f"    ERRO: {rotulo} seed={seed} inviável: {motivo}")
                    sys.exit(1)
                custo = inst.solution_cost(resultado)
                gap = 100 * (custo - inst.optimum) / inst.optimum
                gaps.append(gap)
                for k in resumo_extra[rotulo]:
                    resumo_extra[rotulo][k] += c[k]

            media = sum(gaps) / len(gaps)
            resumo_gap[rotulo].setdefault(set_name, []).append(media)
            print(f"  {rotulo:28s}  gap_medio={media:6.2f}%")
        print()

    dt_total = time.perf_counter() - t_inicio
    print(f"Tempo total de execução: {dt_total:.1f}s\n")

    print("=== Resumo GERAL (média sobre as 8 instâncias, todos os 3 conjuntos) ===")
    for rotulo, alpha, scale in CONFIGURACOES:
        todos_gaps = [g for lst in resumo_gap[rotulo].values() for g in lst]
        media_geral = sum(todos_gaps) / len(todos_gaps)
        e = resumo_extra[rotulo]
        pct_colapso = 100.0 * e["colapsadas"] / e["total"] if e["total"] else 0.0
        pct_drift = 100.0 * e["drift"] / e["total"] if e["total"] else 0.0
        media_pos = e["pos_soma"] / e["pos_n"] if e["pos_n"] else 0.0
        media_neg = e["neg_soma"] / e["neg_n"] if e["neg_n"] else 0.0
        razao = abs(media_neg) / media_pos if media_pos else float("inf")
        print(f"  {rotulo:28s}  gap_medio={media_geral:6.2f}%  colapso={pct_colapso:5.1f}%  "
              f"drift={pct_drift:5.2f}%  R+_media={media_pos:9.4f}  R-_media={media_neg:9.4f}  "
              f"|R-|/R+={razao:7.2f}x  (n+={e['pos_n']} n-={e['neg_n']})")

    print("\n=== Resumo POR CONJUNTO (A vs. B vs. X) ===")
    for rotulo, alpha, scale in CONFIGURACOES:
        linha = f"  {rotulo:28s}"
        for set_name in ("A", "B", "X"):
            lst = resumo_gap[rotulo].get(set_name, [])
            if lst:
                linha += f"  {set_name}={sum(lst)/len(lst):5.2f}%"
        print(linha)


if __name__ == "__main__":
    main()
