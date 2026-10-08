#!/usr/bin/env python3
"""
bandit.py
=========
HG-UCB (Hybrid Gradient UCB): o bandit multi-armado que escolhe qual
vizinhança tentar a cada iteração interna do Bandit VNS (Algoritmo 3 de
Kalatzantonakis et al. 2023), combinando:

  - a REGRA DE ATUALIZAÇÃO de preferência do Gradient Bandit (Eq. 11,
    Sutton & Barto 2018) -- softmax das preferências H vira o "q value"
    de cada operador;
  - a REGRA DE SELEÇÃO do UCB1 (Eq. 5) -- usa esse q value no lugar da
    média empírica de recompensa que o UCB1 "puro" usaria;
  - cold-start uniforme entre operadores nunca tentados (Algoritmo 4,
    linhas 2-3);
  - wrapper ε-greedy por cima de tudo isso (Eq. 14, ε=0.1 fixo em todos
    os experimentos do artigo);
  - filtro por limiar b_t=0.2: operador com q_value abaixo do limiar é
    excluído do pool de seleção -- só PARA a iteração externa do GVNS
    corrente (reseta a cada novo Shake, ver Entrada 2 do diário: "the
    operator is removed from the pool until the next iteration starts");
  - detecção de concept drift via ADWIN (Bifet & Gavalda 2007), usando a
    implementação de referência da biblioteca `river` (não reimplementada
    à mão -- ADWIN é um algoritmo de streaming não-trivial o bastante
    para preferir uma implementação madura e testada por terceiros a uma
    própria, mesmo princípio de usar OR-Tools/CP-SAT em vez de um
    bin-packing solver artesanal em construction.py).

CORREÇÃO CONFIRMADA NO PRÓPRIO ARTIGO PUBLICADO (não um erro de extração
do pdftotext -- verificado por inspeção visual direta da página 17 do
PDF): a Equação 11, como IMPRESSA no artigo, mostra
    H_{t+1}(a) = H_t(a) + α(R_t - R̄_t)π_t(a),   para todo a ≠ A_t
com sinal de MAIS para os operadores não escolhidos. Isso contradiz (a) a
frase seguinte do PRÓPRIO texto do artigo -- "If R_t is greater than R̄_t
then H(At) increases while non-selected actions preference value
DECREASES" -- e (b) a fórmula canônica de Sutton & Barto (2018, Eq. 2.12),
citada explicitamente pelo artigo como a fonte desta equação, que tem
sinal de MENOS nesse termo. Interpretado como erro tipográfico real do
artigo publicado (a "correção 2" desta linha de pesquisa, na mesma linha
do que já foi encontrado no formato de fórmula das Eq. 12-13, Entrada 2 --
mas desta vez confirmado por inspeção visual da própria imagem do PDF, não
por comparação com resumo de terceiros). Implementado aqui com o sinal de
MENOS, conforme Sutton & Barto e conforme a prosa do próprio artigo.

Outra decisão documentada (gap real do pseudocódigo, não erro): o artigo
diz que R̄_t é "a média de recompensa até e incluindo o tempo t" -- lido
literalmente, isso faria a primeira recompensa nunca mover H nenhum (R̄_1
seria igual a R_1). A formulação padrão de Sutton & Barto (a fonte citada)
usa a média das recompensas ANTERIORES a t (excluindo R_t) como baseline
de comparação -- interpretação adotada aqui.

DUAS OUTRAS decisões de interpretação, tomadas em 01/09/2026 ao escrever
bandit_vns() (vns.py) e re-lidas com cuidado junto ao pseudocódigo do
Algoritmo 3/4 (paper.txt):

(1) "n" escalar (Algoritmo 3, linha 6: "n <- n+1", incrementado no TOPO de
CADA iteração interna, antes de qualquer seleção) NÃO é o mesmo "n" do
Algoritmo 4 (que lá é um VETOR por-operador, n_j). O "n" escalar da linha 6
corresponde ao "s" (timestep) que o Algoritmo 4 recebe como argumento
SEPARADO ("Input: C, s, n, p̂") e usa em log(s) na fórmula do UCB1 (linha 6
do Alg. 4) -- sobreposição de símbolo entre os dois blocos de pseudocódigo,
não um erro nosso. Por isso self.s é incrementado aqui no TOPO de select()
(chamado uma vez por iteração interna sem drift, espelhando a linha 6 antes
da própria chamada "UCB(C,step,n,p̂)" da linha 12) -- e NÃO mais dentro de
update() (onde estava antes, incorretamente, incrementando log(s) com um
atraso de uma iteração inteira).

(2) "p̂" -- usado tanto na linha 13 do Alg. 3 ("p̂ <- Sliding_avg(p,w)",
sobre o stream GLOBAL p) quanto na linha 6 do Alg. 4 ("p̂_j", COM índice por
operador) -- tem leitura ambígua: um "sliding window de W=5 recompensas por
OPERADOR" separado dos q_values (softmax de H) é uma leitura possível, mas
exigiria um TERCEIRO mecanismo de estado não mencionado em nenhum outro
lugar do artigo. A leitura adotada aqui -- "p̂" = "q_values" = softmax(H) --
é a mais economicamente sustentada pelo texto: (a) a própria Seção 5.3
descreve HG-UCB como a combinação de exatamente DOIS mecanismos (update de
preferência do Gradient Bandit + fórmula de seleção do UCB1), não três; (b)
o nome da própria variável no pseudocódigo -- "UpdateQ(q_values, reward)",
linha 23 -- bate com "q_values" como o artefato central e único mantido
entre iterações; (c) Seção 5.4 chama estas mesmas grandezas de "Qvalues" e
descreve-as como normalizadas por softmax E usadas tanto para alimentar o
ADWIN quanto para o filtro bt -- exatamente o papel que q_values() já cumpre
aqui. Registrada como decisão CONSCIENTE, não uma leitura óbvia -- a lacuna
real do pseudocódigo é genuína (ver Entrada do diário desta data).

VERIFICAÇÃO INDEPENDENTE (01/09/2026, Gemini 2.5 Flash via MCP, primeiro uso
desta ferramenta neste projeto -- pedido explícito do pesquisador para
verificar decisões de interpretação): apresentamos as duas leituras acima
SEM revelar qual havíamos escolhido. O Gemini favoreceu a leitura (B) --
sliding-window por-operador separada de H -- com um argumento textual real
que não tínhamos ponderado o suficiente: "p̂" e "q_values" são variáveis
SINTATICAMENTE DISTINTAS no pseudocódigo (linha 1, "initialize(n,p,p̂)", não
menciona q_values/H; linha 13 usa Sliding_avg(), linha 23 usa UpdateQ() --
dois mecanismos de atualização com nomes diferentes). Isso é um sinal
textual genuíno, registrado aqui honestamente -- NÃO descartado. Mantivemos
a leitura (A) mesmo assim, por três razões: (i) a leitura (B) do próprio
Gemini exige a MESMA quantidade de "leitura caridosa" de uma linha do
pseudocódigo tida como imprecisa -- ele próprio reconhece que a linha 20
("p.append(reward)", sem índice de operador) precisaria realmente ser
"p[l].append(reward)" para (B) funcionar, ou seja, nenhuma das duas leituras
é uma cópia 100% literal do texto; (ii) a frase em prosa da Seção 5.3 (HG-UCB
= EXATAMENTE dois mecanismos combinados) é uma afirmação AUTORAL direta
sobre a identidade do próprio algoritmo, não uma linha de pseudocódigo
solta -- pesa mais na balança; (iii) o escopo de implementação adicional
exigido pela leitura (B) fica CONTIDO a um único ponto (o termo de
exploitation da fórmula UCB1 dentro de select()). Dado o caráter isolado
dessa peça, o pesquisador pediu (01/09/2026, via AskUserQuestion) para
IMPLEMENTAR E COMPARAR as duas leituras ANTES de escolher qual entra na
validação da Tarefa #178, em vez de assumir (A) e só reconsiderar depois.
Implementado como `p_hat_mode` no construtor de HGUCBBandit: "q_values"
(leitura A, default) vs. "sliding_window" (leitura B do Gemini -- método
`p_hat()`, atualizado em `update()`). O filtro b_t e o pool desbloqueado
continuam SEMPRE sobre q_values()=softmax(H) nos dois modos -- só o termo
de exploitation do UCB1 muda -- para isolar exatamente a peça em disputa
e não misturar duas mudanças numa comparação só. Resultado do experimento
comparativo: ver `tests/compare_phat_variants.py` e Entrada 6 do diário.
Ambiguidade mantida como aberta e documentada, não fechada por autoridade
de uma segunda opinião de IA; decisão de qual leitura usar segue sendo
nossa, com a IA como insumo adicional, não como árbitro.

(4) ALIMENTAÇÃO DO ADWIN -- achado real #8 (diário, Entrada 14, 02/09/2026,
encontrado por auditoria manual linha-a-linha pedida pelo pesquisador após
o achado real #7): a Seção 5.4 do artigo diz "The resulting reward signals
from all operators per iteration (Qvalues) are first normalized with the
softmax function... Qvalues are then stacked in a sequence and further
processed using ADWIN to detect concept drift" -- ou seja, "Qvalues"
(=q_values()=softmax(H), a MESMA grandeza que já alimenta corretamente o
filtro b_t) é o que deveria alimentar o ADWIN, não a recompensa bruta da
Eq. 12/13. A decisão (2) acima, escrita em 01/09/2026, já CITAVA esta
mesma frase do artigo como evidência para a leitura "q_values" de p̂, e
afirmava que isso era "exatamente o papel que q_values() já cumpre aqui"
-- afirmação que a leitura direta de update() (antes desta correção)
desmentia: a linha real era `self.adwin.update(reward)`, sempre a
recompensa bruta, nunca q_values(). Não era uma leitura nova do artigo,
era uma DISCREPÂNCIA entre o que já documentávamos ter decidido e o que o
código de fato fazia.

Verificação empírica ANTES de mudar o default (diário, Entrada 14): com a
alimentação antiga (recompensa bruta), o ADWIN dispara numa taxa nada
trivial e fortemente sensível a alpha -- 0,69% das iterações internas
(≈5,6 resets completos por execução) com alpha=5,0 normalizado, contra
0,04% (≈0,35 resets) com alpha=0,5 normalizado, quase 16x menos. Cada
disparo é um APAGAMENTO COMPLETO de H/n/s (reset_drift()) -- uma pista
plausível (não confirmada) para parte da vantagem empírica já observada de
alpha=0,5 sobre alpha=5,0 (Entrada 13).

Implementado `adwin_feed_mode` no construtor (mesmo padrão de
`p_hat_mode`): "reward" (default -- preserva EXATAMENTE o comportamento já
validado em todas as validações completas até aqui, retrocompatibilidade
total) vs. "q_value" (nova alternativa, testada em `tests/diag7_adwin_feed.py`
-- alimenta o ADWIN com pi[chosen], o q_value/softmax(H) do operador
ESCOLHIDO nesta iteração, calculado ANTES desta atualização de H -- a
leitura mais direta e defensável de "Qvalues" como um streaming escalar
por iteração, já que ADWIN é um detector de deriva de um stream UNIVARIADO
por natureza; stackar os k valores por operador todos juntos misturaria
identidades de operadores diferentes numa única sequência temporal, o que
não faria sentido para detectar deriva do AMBIENTE).

RESULTADO DO EXPERIMENTO COMPARATIVO (diário, Entrada 15, `tests/diag7_adwin_feed.py`,
mesma amostra de 8 instâncias, cruzando os dois alphas candidatos da Entrada
13): "q_value" de fato reduz a taxa de drift como esperado (alpha=5,0:
0,69%→0,50%; alpha=0,5: 0,04%→0,01%), CONFIRMANDO que a mudança tem o efeito
mecânico previsto -- mas o gap% resultante PIORA discretamente nos dois
alphas (alpha=5,0: 3,64%→3,70%; alpha=0,5: 3,39%→3,50%), não melhora.
RESULTADO NULO/DESFAVORÁVEL para a hipótese de que esta era (parte d)a causa
da lacuna residual da Entrada 13 -- reportado honestamente como tal (não
maquiado), conforme `LEI_GERAL_PESQUISA.md`. "reward" (o comportamento
ORIGINAL, default) continua sendo o melhor candidato conhecido apesar de
ser menos fiel à Seção 5.4 -- mantido como default; "q_value" preservado no
código só para documentar o que foi tentado e permitir reprodução do
experimento, não recomendado para uso.

(5) "solg" ("the best global", usado em rs = |solg - solc| na Eq. 12):
também tem leitura ambígua -- (A) solg = f(x') (o resultado do Shake desta
iteração externa, FIXO durante todo o while(true) interno -- ver linha 3);
(B) solg = o melhor custo já encontrado em toda a execução (um incumbente
global rastreado à parte de x). Decisão: usamos (A) para a FÓRMULA da
recompensa (reward_positive recebe f(x') como sol_global), mas TAMBÉM
rastreamos separadamente, em bandit_vns() (vns.py), um best_cost/best_x
monotonicamente não-decrescente -- usado como VALOR DE RETORNO da função
(para não devolver uma solução pior que a melhor já vista durante a busca,
e para ficar comparável ao retorno de gvns()) mas NÃO usado na fórmula da
recompensa. Razão para (A) na fórmula: a prosa diz "solc is always smaller
than solg for minimization problems" DENTRO da própria descrição do caso de
feedback POSITIVO -- ou seja, essa afirmação só precisa valer quando
reward_positive() é de fato chamada, o que só acontece quando f(x'')<f(x')
JÁ foi confirmado (linha 16) -- sob a leitura (A) isso é verdade por
CONSTRUÇÃO, sem exigir nenhuma suposição adicional sobre o comportamento do
algoritmo. Sob a leitura (B), a mesma afirmação exigiria que x NUNCA se
afaste do incumbente global entre Shakes -- não garantido, já que Shake
aceita piora (docstring de shake() em vns.py) e não há mecanismo de "voltar
ao melhor" no nível externo do Algoritmo 3. VERIFICAÇÃO INDEPENDENTE
(mesma consulta ao Gemini): o Gemini favoreceu inicialmente a leitura (B)
com base no SIGNIFICADO LITERAL da frase "best global" (um x' pós-Shake,
possivelmente piorado, é um nome estranho para "o melhor global") -- um
ponto justo, registrado aqui -- mas sua própria análise também apontou que
NENHUMA das duas leituras torna "solc sempre menor que solg" literalmente
verdadeira em TODAS as iterações, o que sustenta que a frase da prosa só
faz sentido quando escopada ao caso de invocação de reward_positive (como
em (A)), não como uma afirmação global sobre todas as iterações internas
(o Gemini pareceu ler "always" como abrangendo também as iterações SEM
melhora, o que a própria estrutura da Seção 5.4 -- "In the case of positive
feedback..." -- não sustenta). Ambiguidade real, decisão registrada com
justificativa própria, não apenas por deferência à segunda opinião.
"""
import math
import random as _random_mod

from river import drift


class HGUCBBandit:
    """Um bandit HG-UCB por CHAMADA de bandit_vns() -- ver vns.py. Mantém
    todo o estado (H, contadores n[j], timestep s, ADWIN) ao longo de TODA
    a execução (todos os passos externos do GVNS, não só um), sendo
    reiniciado (reset_drift()) SOMENTE quando o ADWIN detecta concept drift
    -- não a cada novo Shake.

    NÃO existe um reset "por passo externo" separado: o filtro b_t
    (unblocked_pool(), "operador removido do pool até a próxima iteração
    começar", Seção 5.4) não precisa de nenhum estado adicional para isso,
    porque já é recalculado do zero a cada chamada a partir do q_values()
    CORRENTE (softmax de H) -- um operador cujo H se recupera (por updates
    de Eq. 11 favoráveis a ele, inclusive quando outro operador é o
    escolhido) volta a aparecer no pool automaticamente, sem precisar de
    nenhum sinalizador "por-iteração" guardado à parte. (Uma versão anterior
    desta docstring mencionava um método reset_step() que nunca chegou a
    ser implementado -- essa menção estava desatualizada/incorreta e foi
    removida em 01/09/2026; não há necessidade funcional dele dado como
    unblocked_pool() foi implementado.)

    DEFAULT de `alpha` -- HISTÓRICO E DECISÃO (05/09/2026, Entrada 26 do
    diário): até aqui este default era 5.0 (Tabela 8, valor literal do
    artigo). A partir das Entradas 6/12 (alpha=0.5 empiricamente melhor) e,
    principalmente, da Entrada 25 (teste de significância pareado nas 74
    instâncias completas do benchmark: alpha=0,5 vence alpha=5,0 de forma
    estatisticamente robusta -- Wilcoxon p=0,00389, teste t pareado
    p=0,00298, os dois concordam, n=148 pares), o pesquisador decidiu
    alinhar o default DESTA CLASSE ao valor já validado, para que qualquer
    uso futuro de bandit_vns() sem `bandit=` explícito (ver fallback em
    vns.py) não herde silenciosamente o valor do artigo já superado por
    evidência. Nenhum resultado já sincronizado depende deste default: todo
    script de diagnóstico/validação que já produziu um número reportado
    sempre passou `alpha=` explicitamente -- EXCETO diag1_bandit_trace.py,
    diag2_bt_sensitivity.py, diag3_ablation_ucb_softmax.py,
    compare_progressive_variant.py e compare_phat_variants.py, que
    investigavam outros mecanismos (tracing, b_t, ablação UCB/softmax,
    variante progressiva de x', p_hat_mode) sob o alpha então-default
    (5.0) como pano de fundo fixo -- esses 5 scripts foram atualizados no
    mesmo commit desta mudança para passar `alpha=5.0` explicitamente,
    preservando exatamente o comportamento que geraram quando os achados
    correspondentes foram registrados no diário, agora que o default da
    classe mudou."""

    def __init__(self, n_operators, C=2.0, alpha=0.5, adwin_delta=0.02,
                 epsilon=0.1, b_t=0.2, rng=None, p_hat_mode="q_values",
                 sliding_window_w=5, adwin_feed_mode="reward"):
        self.k = n_operators
        self.C = C
        self.alpha = alpha
        self.epsilon = epsilon
        self.b_t = b_t
        self.rng = rng if rng is not None else _random_mod.Random()
        self._adwin_delta = adwin_delta
        if p_hat_mode not in ("q_values", "sliding_window"):
            raise ValueError(f"p_hat_mode inválido: {p_hat_mode!r}")
        self.p_hat_mode = p_hat_mode
        self.sliding_window_w = sliding_window_w
        if adwin_feed_mode not in ("reward", "q_value"):
            raise ValueError(f"adwin_feed_mode inválido: {adwin_feed_mode!r}")
        self.adwin_feed_mode = adwin_feed_mode
        self.reset_drift()

    def reset_drift(self):
        """Reset completo do aprendizado -- disparado por drift do ADWIN
        (Algoritmo 3, linha 8: "reset(n,p)") ou no início de uma chamada
        de bandit_vns(). Reinicia H (preferências), contadores n[j], o
        contador global de timestep s, a média de recompensa acumulada, o
        próprio detector ADWIN (janela nova), e -- se p_hat_mode ==
        "sliding_window" (ver docstring do módulo, decisão (2)/experimento
        01/09/2026) -- a janela deslizante de recompensas POR OPERADOR
        usada como termo de exploitation alternativo do UCB1."""
        self.H = [0.0] * self.k
        self.n = [0] * self.k
        self.s = 0
        self._reward_sum = 0.0
        self._reward_count = 0
        self.adwin = drift.ADWIN(delta=self._adwin_delta)
        self._reward_window = [[] for _ in range(self.k)]

    def q_values(self):
        """Softmax das preferências H (Eq. 10) -- os "q values" citados
        na Seção 5.4 e usados tanto no filtro b_t quanto (no lugar da
        média empírica) na fórmula de seleção UCB1 (Eq. 5)."""
        m = max(self.H)
        exps = [math.exp(h - m) for h in self.H]
        total = sum(exps)
        return [e / total for e in exps]

    def p_hat(self, q=None):
        """Termo de EXPLOITATION usado na fórmula de seleção UCB1 (Eq. 5,
        "p̂_j" -- ver docstring do módulo, decisão (2), incluindo a segunda
        opinião do Gemini e o experimento de comparação de 01/09/2026).
        Duas implementações, escolhidas via p_hat_mode no construtor:

        "q_values" (default, nossa leitura original): retorna q_values()
        diretamente -- softmax(H), a MESMA grandeza usada no filtro b_t.

        "sliding_window" (leitura alternativa que o Gemini defendeu como
        melhor sustentada pelo texto -- "p̂ <- Sliding_avg(p,w)", Alg.3
        linha 13): retorna, para cada operador j, a média das últimas
        `sliding_window_w` recompensas BRUTAS (não normalizadas por
        softmax) recebidas quando j foi o operador escolhido. Operadores
        nunca escolhidos (janela vazia) retornam 0.0 -- não deveria
        acontecer na prática, já que select() só chama isto DEPOIS do
        cold-start garantir n[j]>=1 para todo j (ver select())."""
        if self.p_hat_mode == "q_values":
            return q if q is not None else self.q_values()
        janelas = self._reward_window
        return [
            (sum(w) / len(w)) if w else 0.0
            for w in janelas
        ]

    def unblocked_pool(self, q=None):
        """Operadores com q_value >= b_t -- o pool "desbloqueado" no
        instante da chamada (recalculado do zero a cada vez a partir do
        q_values() corrente -- não há estado "de pool" separado para
        resetar; ver docstring da classe). Nunca retorna pool vazio: se
        TODOS os operadores
        estiverem abaixo do limiar, o próprio Algoritmo 3 já teria
        terminado a iteração interna (linha 24: "no improvement &
        max(q_values) < minq: break") antes de precisar chamar select()
        de novo -- mas por robustez, se ainda assim for chamado nesse
        estado, retorna todos os operadores em vez de uma lista vazia."""
        if q is None:
            q = self.q_values()
        pool = [j for j in range(self.k) if q[j] >= self.b_t]
        return pool if pool else list(range(self.k))

    def select(self):
        """Escolhe o índice do operador para esta iteração interna,
        seguindo, em ordem de precedência: (1) cold-start uniforme entre
        TODOS os operadores nunca tentados, IGNORANDO o filtro b_t
        (Algoritmo 4, linhas 2-3 -- ver nota BUG abaixo); (2) wrapper
        ε-greedy (Eq. 14) -- com prob. ε, sorteio uniforme no pool
        desbloqueado; (3) caso contrário, UCB1 (Eq. 5) usando q_values()
        no lugar da média empírica de recompensa, restrito ao pool
        desbloqueado.

        Incrementa self.s (o "n" ESCALAR do Algoritmo 3, linha 6 -- ver
        docstring do módulo, decisão (1)) como primeiríssima ação, ANTES de
        qualquer seleção -- espelhando "n <- n+1" (linha 6) acontecendo
        incondicionalmente no topo do loop interno, antes de "l <-
        UCB(C,step,n,p̂)" (linha 12) na MESMA iteração. select() só é
        chamada no ramo SEM drift (Algoritmo 3, linha 11) -- na iteração
        de drift, quem incrementava seria irrelevante mesmo, pois
        reset_drift() zera self.s de volta a 0 logo em seguida.

        BUG REAL ENCONTRADO E CORRIGIDO (01/09/2026, via teste de bancada
        antes de qualquer validação em instância real -- exatamente o tipo
        de erro que só aparece rodando, não só lendo): cold-start
        originalmente checava "never_tried" só DENTRO do pool já filtrado
        por b_t (`[j for j in pool if n[j]==0]`). Com alpha=5.0 (o
        hiperparâmetro "melhor" da Tabela 8) e uma única atualização de H
        já grande o bastante, os operadores AINDA NÃO TESTADOS podem ver
        seu q_value cair abaixo de b_t=0.2 SÓ PELO EFEITO Eq.11 do
        operador vizinho ter sido escolhido (o termo "-alpha*(...)*pi[a]"
        para a≠chosen não exige que `a` já tenha sido testado) -- ficando
        birlados PARA SEMPRE fora do pool antes de terem a MÍNIMA chance
        de ser tentados. Confirmado empiricamente: um teste de bancada de
        3 iterações ficou preso escolhendo SEMPRE o operador 0 a partir da
        2a iteração, mesmo com os outros 4 operadores ainda em n=0.
        Corrigido fazendo cold-start escanear TODOS os k operadores
        (ignorando b_t), não só o pool -- garante que TODO operador tenha
        ao menos 1 execução antes do filtro b_t/UCB1 entrarem em ação,
        prática padrão em algoritmos bandit (evita estimativa inicial
        indefinida) e consistente com Algoritmo 4 linha 2 fazendo essa
        checagem como o PRIMEIRO passo, antes de qualquer menção a pool
        filtrado (que só aparece na prosa da Seção 5.4, não dentro do
        Algoritmo 4 em si)."""
        self.s += 1

        never_tried = [j for j in range(self.k) if self.n[j] == 0]
        if never_tried:
            return self.rng.choice(never_tried)

        # filtro b_t / pool desbloqueado SEMPRE via q_values()=softmax(H)
        # (Seção 5.4: "Qvalues... normalized with softmax... used for
        # online filtering") -- independente de p_hat_mode, que só afeta
        # o termo de EXPLOITATION do UCB1 abaixo. Isolar a ambiguidade a
        # essa única peça é o propósito do experimento de 01/09/2026.
        q = self.q_values()
        pool = self.unblocked_pool(q)

        if self.rng.random() < self.epsilon:
            return self.rng.choice(pool)

        p_hat = self.p_hat(q)

        def ucb_score(j):
            explore = self.C * math.sqrt(2.0 * math.log(self.s) / self.n[j])
            return p_hat[j] + explore

        return max(pool, key=ucb_score)

    def update(self, chosen, reward):
        """Registra o resultado da iteração (operador `chosen`, valor de
        `reward` já calculado via reward_positive()/reward_negative()
        abaixo), atualiza H de TODOS os operadores (Eq. 11, com o sinal
        corrigido -- ver docstring do módulo), incrementa o contador
        n[chosen] (Algoritmo 4, linha 8: "n_x <- n_x+1" -- NÃO mexe em
        self.s, que já foi incrementado por select() no início desta MESMA
        iteração, ver sua docstring), alimenta o ADWIN com a recompensa, e
        retorna True se um drift foi detectado NESTA atualização (quem
        chama decide o reset -- bandit_vns() em vns.py chama reset_drift()
        e sorteia uma vizinhança aleatória para a PRÓXIMA iteração quando
        isso acontece, conforme Algoritmo 3 linhas 7-9).

        Alimentação do ADWIN controlada por `adwin_feed_mode` (achado real
        #8, diário Entrada 14 -- ver docstring do módulo, decisão (4)):
        "reward" (default) alimenta com a recompensa bruta desta iteração
        (Eq. 12/13) -- comportamento ORIGINAL, preservado por
        retrocompatibilidade; "q_value" alimenta com pi[chosen] (o q_value/
        softmax(H) do operador ESCOLHIDO, calculado ANTES desta atualização
        de H -- mesma variável `pi` já usada acima para a Eq. 11), a leitura
        mais direta de "Qvalues" (Seção 5.4) como o que deveria alimentar o
        ADWIN."""
        r_bar = self._reward_sum / self._reward_count if self._reward_count else 0.0

        pi = self.q_values()  # π_t ANTES desta atualização (Eq. 11 usa π_t(A_t), o valor corrente)
        for a in range(self.k):
            if a == chosen:
                self.H[a] += self.alpha * (reward - r_bar) * (1.0 - pi[a])
            else:
                self.H[a] -= self.alpha * (reward - r_bar) * pi[a]

        self.n[chosen] += 1
        self._reward_sum += reward
        self._reward_count += 1

        if self.p_hat_mode == "sliding_window":
            janela = self._reward_window[chosen]
            janela.append(reward)
            if len(janela) > self.sliding_window_w:
                janela.pop(0)

        adwin_input = reward if self.adwin_feed_mode == "reward" else pi[chosen]
        self.adwin.update(adwin_input)
        return bool(self.adwin.drift_detected)


def reward_positive(sol_global, sol_current, unblocked_count, exec_count):
    """Equação 12 (feedback positivo -- o operador melhorou a solução):
    reward = rs * Nt(u)^(-1/nt(k)), rs = |sol_g - sol_c|.
    sol_global: custo de x' (o resultado do Shake desta iteração externa,
        FIXO durante todo o loop interno) -- decisão de interpretação para
        "sol_g" ("the best global"); ver docstring do módulo, decisão (3),
        incluindo a segunda opinião do Gemini e por que não seguimos ela
        aqui. NÃO é o melhor custo já visto em toda a execução -- esse é
        rastreado separadamente em bandit_vns() (vns.py) só para o valor
        de retorno, não para esta fórmula.
    sol_current: custo da nova solução encontrada, x'' (sol_c) -- só é
        chamada quando f(x'')<f(x') já vale, então sol_current<sol_global
        sempre nesta função (coerente com "solc is always smaller than
        solg" da Seção 5.4, sob a leitura (3)-(A) adotada).
    unblocked_count: |Nt(u)| -- tamanho do pool desbloqueado usado nesta seleção.
    exec_count: nt(k) -- contagem de execuções do operador escolhido (>=1)."""
    rs = abs(sol_global - sol_current)
    return rs * (unblocked_count ** (-1.0 / exec_count))


def reward_positive_normalized(sol_global, sol_current, unblocked_count, exec_count):
    """Variante EXPERIMENTAL de reward_positive() (Eq. 12) -- NÃO está no
    artigo, é uma proposta NOSSA -- motivada pelo Diagnóstico 5 (diário,
    Entrada 11, dentro da linha de investigação do Achado Real #4/#5,
    Entradas 9-10): com rs=|sol_g-sol_c| em unidades de custo ABSOLUTAS, a
    faixa "segura" de alpha (que evita colapso de softmax(H) -- Entrada 9)
    depende do conjunto de instâncias -- o conjunto X, com custos tipicamente
    maiores, colapsa em alphas bem menores que A/B (Entrada 11). Hipótese
    testada aqui (`tests/diag6_reward_normalization.py`): dividir rs pelo
    custo da solução corrente torna a recompensa uma medida de melhora
    RELATIVA (adimensional -- mesma ideia do "gap%" usado em toda a
    validação deste projeto), o que deveria tornar o efeito de um dado
    alpha muito mais consistente entre instâncias de escalas diferentes,
    inclusive possivelmente permitindo manter alpha=5.0 (Tabela 8 do
    artigo) SEM reescalar o hiperparâmetro -- ao contrário da correção via
    "alpha pequeno" (Entradas 9-10), que altera um valor que o artigo
    reporta como "melhor".

    Eq. 13 (reward_negative) já é adimensional (só usa q_prev∈[0,1] e
    termos de contagem/pool) -- não precisa de uma variante correspondente,
    só reward_positive (Eq. 12) usa uma grandeza de custo em unidades
    absolutas."""
    rs = abs(sol_global - sol_current) / sol_current
    return rs * (unblocked_count ** (-1.0 / exec_count))


def reward_positive_normalized_scaled(sol_global, sol_current, unblocked_count,
                                       exec_count, scale=100.0):
    """Variante EXPERIMENTAL de reward_positive_normalized() -- NÃO está no
    artigo, é uma proposta NOSSA -- motivada pelo achado real #9 (diário,
    Entrada 16): o tracing detalhado (`tests/diag8_bandit_trace_v2.py`)
    revelou que a normalização da Entrada 12 (dividir rs por sol_current)
    tornou a recompensa POSITIVA (Eq. 12) uniformemente pequena (média
    0,0146, faixa 0,0006-0,11 em A-n32-k5) frente à recompensa NEGATIVA
    (Eq. 13, nunca normalizada, sempre em unidades "absolutas" de q_prev/
    pool -- média -1,6734, faixa -2,49 a -0,54) -- uma assimetria de ~115x
    que não existia da mesma forma antes da Entrada 12 (quando a Eq. 12
    estava em unidades de custo absolutas, ora maiores ora menores que a
    Eq. 13, dependendo da instância). Hipótese testada aqui
    (`tests/diag9_reward_rescale.py`): reescalar a recompensa positiva
    normalizada por um fator fixo `scale` (default 100.0 -- mesma
    convenção de "porcentagem" já usada em `erro_pct` em todo o projeto:
    rs é uma fração relativa, x100 a expressa como percentual de melhora)
    para aproximar sua ordem de grandeza da Eq. 13, e verificar se isso
    reduz a dominância do termo "-r_bar" sobre a magnitude real da melhora
    dentro do update de H (Eq. 11) -- ver achado real #9 (Entrada 16) para
    a análise completa de por que essa dominância torna o sinal de sucesso
    quase binário na prática.

    Implementado como wrapper ADITIVO sobre reward_positive_normalized()
    (nunca duplica a fórmula, elimina risco de divergência/typo entre as
    duas) -- `scale` só é diferente do valor original (equivalente a
    scale=1.0) quando o chamador passa explicitamente
    reward_positive_fn=functools.partial(reward_positive_normalized_scaled,
    scale=X) (ver diag9, que varre vários valores de X). Não recebe
    scale=1.0 como default porque nesse caso seria idêntico a já usar
    reward_positive_normalized diretamente -- o objetivo desta função é
    justamente representar o candidato reescalado. reward_positive() e
    reward_positive_normalized() permanecem INTOCADAS; o default de
    bandit_vns() (reward_positive_fn=None -> reward_positive()) também não
    muda -- extensão estritamente aditiva, mesmo padrão de
    reward_positive_fn (Entrada 11) e adwin_feed_mode (Entrada 14).

    RESULTADO DO EXPERIMENTO (diário, Entrada 17, `tests/diag9_reward_rescale.py`,
    mesma amostra de 8 instâncias, 5 seeds, cruzando os dois alphas candidatos
    da Entrada 13 com scale em {1x(=sem reescala), 10x, 50x, 100x, 200x}):
    RESULTADO NEGATIVO, monotônico e dose-dependente -- reescalar PIORA o
    gap% em TODA E QUALQUER escala testada, para os dois alphas, e a piora
    cresce junto com a taxa de colapso do softmax (ex. alpha=5,0: gap
    3,64%(1x)->4,03%(10x)->4,96%(50x)->5,44%(100x)->5,37%(200x), colapso
    12,5%->40,2%->77,5%->91,8%->92,1%). O objetivo mecânico (aproximar
    |R-|/R+ de 1x) É alcançado por volta de scale=100 (1,34x p/ alpha=5,0;
    0,86x p/ alpha=0,5) -- mas alcançá-lo exige AMPLIFICAR a magnitude bruta
    da recompensa positiva, o que desfaz exatamente a propriedade
    ("recompensa positiva uniformemente pequena") que a normalização da
    Entrada 12 usa para evitar o colapso de H em primeiro lugar. Ou seja: a
    assimetria de escala do achado real #9 e a proteção contra colapso do
    achado real #6 estão em tensão DIRETA -- não dá para corrigir uma via
    reescala uniforme sem reabrir a outra. `scale=100.0` (a sugestão
    original) fica preservado como default só para reprodutibilidade do
    experimento -- NÃO é recomendado para uso; nenhum candidato desta
    família substitui os candidatos da Entrada 13 (normalizado, SEM
    reescala, alpha=5,0 ou alpha=0,5)."""
    return scale * reward_positive_normalized(sol_global, sol_current, unblocked_count, exec_count)


def reward_negative(q_prev_chosen, unblocked_count, exec_count):
    """Equação 13 (feedback negativo -- o operador NÃO melhorou):
    regret = -1/2 + (-2 * q_{t-1} * Nt(u)^(-1/nt(k))).
    q_prev_chosen: q_{t-1} -- o q_value do operador escolhido ANTES desta
    atualização (softmax de H antes de update() ser chamado)."""
    return -0.5 + (-2.0 * q_prev_chosen * (unblocked_count ** (-1.0 / exec_count)))
