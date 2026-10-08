#!/usr/bin/env python3
"""
ablation_randomness_bandits.py
================================
Subclasses de `HGUCBBandit` (src/bandit.py) usadas SÓ pela ablação dos 3
mecanismos de aleatoriedade do HG-UCB -- Direção 2 (Tarefa #179), diário
Entrada 28, 06/09/2026. NÃO fazem parte da implementação validada e não
são usadas em nenhuma validação "oficial" da Tarefa #178. Propositalmente
mantidas FORA de src/, mesmo racional de tests/ablation_bandits.py (a
OUTRA ablação deste projeto, UCB1-only vs. softmax-only, Entrada 8): estes
3 mecanismos são inequivocamente parte do algoritmo do artigo (não são
leituras alternativas em disputa, ao contrário de p_hat_mode/
adwin_feed_mode, que por isso continuam como flags de primeira classe em
HGUCBBandit) -- isolá-los aqui evita qualquer risco de uma flag de
ablação ser deixada ligada por engano numa validação real.

Cada classe HERDA de HGUCBBandit e sobrescreve SÓ `select()` (mesmo padrão
de `TracingHGUCBBandit`, tests/diag1_bandit_trace.py) -- `update()`,
`q_values()`, `p_hat()`, `unblocked_pool()`, `reset_drift()` continuam
exatamente os herdados, sem reimplementação, então qualquer correção
futura em HGUCBBandit propaga automaticamente para estas subclasses
(exceto na própria lógica de seleção que cada uma ablata de propósito).

Filosofia de ablação adotada (Gemini, 2.5 Flash, consulta neutra, Entrada
28): "B" -- remover a intervenção inteiramente e deixar o PRÓXIMO
mecanismo da cadeia de precedência (cold-start > ε-greedy > UCB1) decidir
sozinho, em vez de "A" (substituir a aleatoriedade por uma escolha
determinística que preserva a função do mecanismo removido).

CUIDADO COM UMA ARMADILHA (documentada aqui para não ser redescoberta):
uma forma ÓBVIA de "consertar" a divisão por zero que aparece ao remover
o cold-start é dar pontuação infinita a operadores nunca tentados (n[j]=0)
no termo de exploração do UCB1 -- convenção comum em implementações
didáticas do algoritmo. ISSO NÃO ABLA NADA: garante os mesmos 5 operadores
testados em sequência antes de qualquer exploitation, só troca o bloco
`if never_tried: ...` explícito por uma pontuação que força exatamente o
mesmo resultado por outro caminho. A correção adotada abaixo (piso
`max(n[j], 1)` só no DENOMINADOR do termo de exploração) é diferente: não
garante nada, só evita o crash -- um operador nunca tentado ganha um termo
de exploração FINITO (não infinito) que cresce com log(s) como qualquer
outro, competindo em pé de igualdade com os já testados via a própria
fórmula do UCB1, sem nenhuma regra especial por cima dela. Como
p_hat_mode="q_values" (o modo de produção) já é uniforme (1/k) na
inicialização por construção (H começa em zero -> softmax(H) uniforme),
não há necessidade de nenhuma "inicialização otimista" adicional no termo
de exploitation -- só o termo de exploração precisava do piso.
"""
import math

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bandit import HGUCBBandit  # noqa: E402


class NoColdStartBandit(HGUCBBandit):
    """Ablação do mecanismo (1): cold-start removido. Ver docstring do
    módulo para a justificativa do piso `max(n[j], 1)` e da alternativa
    rejeitada (pontuação infinita). ε-greedy e UCB1 (com o piso) idênticos
    a HGUCBBandit.select() -- só o bloco `if never_tried: ...` (linhas
    383-385 de bandit.py) foi removido."""

    def select(self):
        self.s += 1
        q = self.q_values()
        pool = self.unblocked_pool(q)

        if self.rng.random() < self.epsilon:
            return self.rng.choice(pool)

        p_hat = self.p_hat(q)

        def ucb_score(j):
            explore = self.C * math.sqrt(2.0 * math.log(self.s) / max(self.n[j], 1))
            return p_hat[j] + explore

        return max(pool, key=ucb_score)


class NoEpsilonGreedyBandit(HGUCBBandit):
    """Ablação do mecanismo (3): wrapper ε-greedy removido -- uma vez que
    o cold-start termina, a seleção é SEMPRE o argmax do UCB1 (Eq. 5),
    nunca o sorteio uniforme de Eq. 14. Cold-start e UCB1 idênticos a
    HGUCBBandit.select() -- só o bloco `if self.rng.random() < epsilon:
    ...` (linhas 395-396 de bandit.py) foi removido. Sem risco de divisão
    por zero aqui: cold-start (mantido) já garante n[j]>=1 para todo
    operador do pool antes de este ramo ser alcançado -- não precisa do
    piso usado em NoColdStartBandit."""

    def select(self):
        self.s += 1

        never_tried = [j for j in range(self.k) if self.n[j] == 0]
        if never_tried:
            return self.rng.choice(never_tried)

        q = self.q_values()
        pool = self.unblocked_pool(q)
        p_hat = self.p_hat(q)

        def ucb_score(j):
            explore = self.C * math.sqrt(2.0 * math.log(self.s) / self.n[j])
            return p_hat[j] + explore

        return max(pool, key=ucb_score)


class NoColdStartNoEpsilonBandit(HGUCBBandit):
    """Ablação COMBINADA dos mecanismos (1)+(3): cold-start E ε-greedy
    removidos juntos -- braço "sem_nenhum" da Entrada 28 (drift-reset,
    mecanismo (2), é desligado à parte via `drift_reset_enabled=False`
    em bandit_vns(), não tem representação nesta classe). A seleção é
    SEMPRE o argmax do UCB1 com o piso de NoColdStartBandit.

    NOTA MECÂNICA registrada de antemão (Entrada 28): na primeiríssima
    chamada (s=1, todo n[j]=0), o piso deixa o termo de exploração em
    C*sqrt(2*log(1)/1)=0 para todos -- e p_hat[j]=1/k=0,2 também é igual
    para todos (H começa em zero). Empate exato entre os 5 operadores;
    `max(pool, key=ucb_score)` do Python resolve empates a favor do
    PRIMEIRO elemento na ordem do pool (índice 0 -- 2-opt, ver
    neighborhoods.NEIGHBORHOODS), deterministicamente, toda vez que este
    braço reinicia do zero (início da execução, ou logo após um
    drift-reset se este mecanismo não estiver desligado nesta mesma
    execução). Não é um bug desta classe -- é a consequência fiel e
    esperada de remover os dois únicos mecanismos que poderiam desfazer
    esse empate (cold-start escolheria uniformemente entre os 5;
    ε-greedy, com prob. 0,1, também poderia)."""

    def select(self):
        self.s += 1
        q = self.q_values()
        pool = self.unblocked_pool(q)
        p_hat = self.p_hat(q)

        def ucb_score(j):
            explore = self.C * math.sqrt(2.0 * math.log(self.s) / max(self.n[j], 1))
            return p_hat[j] + explore

        return max(pool, key=ucb_score)
