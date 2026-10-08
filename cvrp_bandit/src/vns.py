#!/usr/bin/env python3
"""
vns.py
======
VND (Algoritmo 2) e GVNS (Algoritmo 1) de Kalatzantonakis et al. (2023) --
baseline SEM bandit (ordem de vizinhança fixa). O bandit entra depois,
como um módulo à parte que decide a ORDEM/ESCOLHA de vizinhança; esta
implementação usa a ordem fixa de NEIGHBORHOODS (ver neighborhoods.py).

Lacuna real encontrada no Algoritmo 1 do artigo (documentada em
diario_pesquisa.md, Entrada 3, não escondida em silêncio): o pseudocódigo
da Fig. "Algorithm 1" (pág. 8) mostra Shake(x,kmax) -> VND(x',kmax) -> mede
tempo -> repete, mas NÃO mostra explicitamente um passo "aceita x'' como
novo x somente se f(x'') < f(x)". Isso contraria a definição clássica de
GVNS (Hansen & Mladenović, citada pelos próprios autores) e, se levado ao
pé da letra (aceitar x'' sempre), o algoritmo seria um passeio aleatório
sem nunca travar uma melhora -- incompatível com os resultados de 2-6% de
gap reportados nas Tabelas 10-12. Adotamos a regra de aceitação PADRÃO da
literatura de VNS (aceita x'' só se estritamente melhor; caso contrário
mantém x e faz o próximo Shake a partir do x anterior) -- é a única leitura
que produz um algoritmo de otimização coerente, mas é uma interpretação
nossa de uma lacuna do pseudocódigo publicado, não uma cópia literal dele.
"""
import time

from neighborhoods import NEIGHBORHOODS
from bandit import HGUCBBandit, reward_positive, reward_negative


def vnd(instance, routes, neighborhoods=None):
    """Algoritmo 2: Variable Neighborhood Descent.

    Cicla pelas vizinhanças em ordem fixa; ao achar melhora em N_l, aplica
    e volta para N_1; para quando passa por TODAS sem nenhuma melhora.
    """
    if neighborhoods is None:
        neighborhoods = NEIGHBORHOODS
    l = 0
    while l < len(neighborhoods):
        _, best_fn, _ = neighborhoods[l]
        improved, new_routes = best_fn(instance, routes)
        if improved:
            routes = new_routes
            l = 0
        else:
            l += 1
    return routes


def shake(instance, routes, rng, neighborhoods=None):
    """Shake do Algoritmo 1, linha 5: sorteia UM índice de vizinhança
    uniformemente e aplica UM movimento aleatório dela. Aceita o resultado
    mesmo se piorar (desde que viável) -- é a diversificação, não a busca
    por melhora. Se a vizinhança sorteada não tiver nenhum movimento válido
    (ex.: operador inter-rota com só 1 rota), tenta as outras antes de
    desistir (evita ficar "preso" sem perturbar nunca)."""
    if neighborhoods is None:
        neighborhoods = NEIGHBORHOODS
    ordem = list(range(len(neighborhoods)))
    rng.shuffle(ordem)
    for idx in ordem:
        _, _, random_fn = neighborhoods[idx]
        novo = random_fn(instance, routes, rng)
        if novo is not None:
            return novo
    return routes  # nenhuma vizinhança tinha movimento válido -- sem perturbação


def gvns(instance, initial_routes, rng, steps_max=20, time_max=None,
         neighborhoods=None, on_step=None):
    """Algoritmo 1: General Variable Neighborhood Search.

    steps_max: número de ciclos shake+VND (equivalente ao 'GVNS steps' das
    Tabelas 6-8 do artigo -- eles testaram 5, 10 e 20).
    time_max: teto de tempo de parede em segundos (None = sem teto, só
    steps_max decide).
    on_step: callback opcional (step, custo_atual) -- usado para logging/
    depuração, não altera o algoritmo.
    """
    x = [list(r) for r in initial_routes]
    best_cost = instance.solution_cost(x)
    t0 = time.perf_counter()
    step = 0
    while step < steps_max:
        if time_max is not None and (time.perf_counter() - t0) >= time_max:
            break
        x_shaken = shake(instance, x, rng, neighborhoods)
        x_local = vnd(instance, x_shaken, neighborhoods)
        cost_local = instance.solution_cost(x_local)
        if cost_local < best_cost - 1e-9:
            x = x_local
            best_cost = cost_local
        step += 1
        if on_step is not None:
            on_step(step, best_cost)
    return x


def bandit_vns(instance, initial_routes, rng, steps_max=20, time_max=None,
                neighborhoods=None, bandit=None, max_inner_iter=None,
                on_step=None, on_inner=None, progressive_x_linha=False,
                reward_positive_fn=None, drift_reset_enabled=True,
                reward_negative_fn=None):
    """Algoritmo 3: Bandit VNS (Kalatzantonakis et al. 2023) -- versão COM
    bandit HG-UCB escolhendo a vizinhança a cada iteração interna, ao invés
    da ordem fixa usada por vnd()/gvns(). Reusa shake() (idêntico ao GVNS
    clássico -- Algoritmo 3 linha 3 é o MESMO Shake do Algoritmo 1) e os
    best_fn já validados de neighborhoods.NEIGHBORHOODS.

    ESTRUTURA (fiel ao pseudocódigo -- outer loop = passos GVNS, inner loop
    = tentativas de operador dentro de um mesmo passo):

      outer: x' <- Shake(x, kmax)                              [linha 3]
      inner (repete até parar):
        se drift pendente (do update() da iteração ANTERIOR): reseta o
          bandit e sorteia vizinhança uniforme                 [linhas 7-9]
        senão: escolhe via bandit.select() (cold-start > ε-greedy > UCB1)
                                                                 [linhas 10-13]
        x'' <- best_fn(vizinhança escolhida) aplicado a x' (FIXO durante
          todo o inner loop -- não a x, não ao x'' de uma iteração anterior)
                                                                 [linha 15]
        se f(x'') < f(x'): x <- x'' (aceita); melhora=True      [linhas 16-19]
        reward <- Eq.12 (positivo) ou Eq.13 (negativo)          [linha 20]
        bandit.update(escolhido, reward) -> pode sinalizar drift[linhas 21-23]
        para (com CAVEAT -- ver abaixo) se sem melhora E max(q)<b_t
                                                                 [linhas 24-26]
      step += 1                                                 [linha 29]

    Ver bandit.py (docstring do módulo) para as decisões de interpretação
    de "n"/s, "p̂"/q_values, e "solg" -- incluindo a segunda opinião do
    Gemini sobre cada uma. Duas decisões ADICIONAIS, específicas desta
    função, tomadas em 01/09/2026:

    (1) x' fica FIXO durante TODO o loop interno (não é atualizado a cada
    x'' aceito) -- leitura literal da linha 15 ("With x' from Shake..."),
    mesmo que isso signifique que múltiplas iterações internas do MESMO
    passo externo competem contra a MESMA referência, não contra um x'
    "andante". Ver bandit.py para o raciocínio completo (inclusive por que
    isso reconcilia "solc sempre menor que solg" da Seção 5.4).

    (2) "x" (o incumbente persistente entre passos externos) NÃO é
    necessariamente monotonicamente não-crescente -- Shake aceita piora
    (docstring de shake() acima), e x só é atualizado quando bate x' (não
    o melhor global). Isso pode fazer "x" divergir do melhor já encontrado.
    Por isso rastreamos best_cost/best_x SEPARADAMENTE (mesmo padrão de
    gvns()) só para o VALOR DE RETORNO desta função -- garante que
    bandit_vns() nunca devolve algo pior do que já foi visto, e fica
    comparável ao retorno de gvns() para a Tarefa #178 (comparação com as
    colunas HG-UCB das Tabelas 10-12/13). Esse rastreamento NÃO influencia
    a busca em si (x continua sendo perturbado/comparado exatamente como
    o Algoritmo 3 descreve) nem a fórmula de recompensa (que usa f(x'),
    não best_cost -- ver bandit.py decisão (3)).

    ACHADO REAL (confirmado matematicamente + segunda opinião do Gemini,
    01/09/2026, aprovado pelo pesquisador via AskUserQuestion): com k=5
    operadores e b_t=0.2 (hiperparâmetros reportados como "melhores" no
    artigo, Tabelas 2 e 7/8), "max(q_values) < b_t" é MATEMATICAMENTE
    IMPOSSÍVEL -- softmax de 5 valores sempre soma 1, então o máximo nunca
    fica abaixo de 1/5=0.2 (confirmado com 200 mil amostras aleatórias de
    H). Ou seja, a condição de parada do loop interno tal como descrita
    NUNCA dispara -- nem mesmo quando há melhora (que também não é, por si
    só, uma condição de parada no pseudocódigo). Decisão aprovada: manter
    b_t=0.2 EXATAMENTE como reportado (fidelidade ao valor publicado, não
    escondendo/alterando um hiperparâmetro que o artigo apresenta como
    testado e escolhido) e adicionar `max_inner_iter` como um TETO DE
    SEGURANÇA de engenharia -- não uma reinterpretação do algoritmo, só uma
    garantia de terminação. Na prática, dado que a condição original nunca
    dispara, este teto se torna o mecanismo de parada EFETIVO do loop
    interno na maioria dos casos -- não é um caso extremo raro, é o
    comportamento típico esperado. O valor de `max_inner_iter` (default
    8*kmax) é portanto um parâmetro adicional, LIVRE, introduzido por nós
    (não está no artigo) -- registrado explicitamente como tal, a ser
    calibrado/questionado na validação empírica da Tarefa #178 se os
    resultados não forem competitivos.

    RESULTADO DA CALIBRAÇÃO EMPÍRICA DE max_inner_iter (diagnóstico 10,
    tests/diag10_max_inner_iter.py, 02/09/2026, aprovado pelo pesquisador
    via AskUserQuestion, recomendado independentemente por mim e pelo
    Gemini): testado max_inner_iter em {20, 40 (default acima), 80, 160}
    x os dois alphas candidatos da Entrada 13 (5,0 -- valor do artigo;
    0,5 -- candidato empírico melhor), com reward_positive_normalized
    (recomendada atual, sem reescala -- achados reais #6/#10), na mesma
    amostra de 8 instâncias/5 seeds dos diagnósticos 5-9. Resultado
    completo, com tabela por instância: diário de pesquisa, Entrada 19.

    (a) CONFIRMAÇÃO ESTRUTURAL (fecha a dedução acima com CONTAGEM DIRETA
    de execução via on_inner, não só dedução matemática): o loop interno
    SEMPRE roda até o teto -- inner_iter_medio_por_passo foi EXATAMENTE
    igual ao max_inner_iter configurado, nas 8 configurações testadas
    (320 execuções), sem exceção. Tempo de execução escala quase
    linearmente com max_inner_iter (dobrar o teto ≈ dobra o tempo) --
    consequência direta de (a), já que mais iterações internas por passo
    custam proporcionalmente mais.

    (b) ACHADO REAL #11 (MISTO -- sensibilidade a max_inner_iter depende
    do alpha): para alpha=0,5 (candidato melhor da Entrada 13), variar
    max_inner_iter NÃO produz ganho real -- o baseline (40) já é o
    melhor dos 4 valores na média de 8 instâncias (3,39%); comparando
    cada instância individualmente entre baseline e mii=160, 3 pioram e
    só 3 melhoram (2 empatam) -- sem direção consistente. NENHUMA
    justificativa para mudar o default de 40 para este candidato.

    Para alpha=5,0 (valor do artigo, historicamente PIOR por causa do
    colapso do softmax -- achados #4/#6), max_inner_iter=160 produz uma
    melhora real: gap médio cai de 3,64% (baseline) para 3,25% (≈11%
    relativo), e a comparação direta baseline-vs-160 é UNÂNIME nas 8
    instâncias (nenhuma piora, 6 melhoram, 2 empatam) -- mesmo com o
    ponto intermediário (mii=80) sendo ruidoso/não-monotônico em várias
    instâncias. Esse resultado (3,25%) SUPERA o melhor já visto para
    alpha=0,5 em qualquer mii testado (3,36%) -- reabre, não fecha, a
    comparação entre os dois alphas candidatos da Entrada 13. Mas vem
    com custo computacional ≈3,6x maior E sem dado de %colapso por
    configuração (este diagnóstico só mediu gap%, iterações internas e
    tempo) -- não sabemos, com os dados desta entrada, se o colapso do
    achado #6 persiste sob mii=160 para alpha=5,0 e é só compensado pelo
    volume maior de tentativas, ou se de fato diminui. Nenhuma mudança
    de default foi aplicada unilateralmente; max_inner_iter=None (-> 40)
    continua sendo o comportamento padrão desta função.

    (c) ACHADO REAL #12 (03/09/2026, diagnóstico 11,
    tests/diag11_colapso_max_inner_iter.py -- resposta direta à pergunta
    em aberto de (b)): o colapso do softmax para alpha=5,0 DE FATO
    DIMINUI com max_inner_iter maior -- não é só compensado pelo volume
    de tentativas. %colapso agregado (8 instâncias): 21,4% (mii=20) /
    12,5% (mii=40, bate exatamente com os achados #6/#10) / 4,4%
    (mii=80) / 1,5% (mii=160). Reconstruído o %colapso POR INSTÂNCIA
    (via diferenciação do acumulado, verificado por execução em Python
    independente) e confirmado que a queda é MONOTÔNICA em TODAS as 8
    instâncias, sem uma única exceção -- o resultado mais limpo (menos
    ruidoso) desta sub-investigação inteira (Entradas 16-20). alpha=0,5
    mantém %colapso ≈0,0% em todos os mii (nunca teve o problema).
    Fortalece a robustez do achado real #11: a superioridade de gap% de
    alpha=5,0/mii=160 vem ACOMPANHADA de colapso substancialmente menor
    (não de colapso escondido/compensado) -- reduz o risco de que seja
    um artefato. Hipótese mecanística (NÃO testada diretamente, só
    plausível): um mii curto pode truncar um passo externo NO MEIO do
    processo de autocorreção já observado no achado #9 (Entrada 16, uma
    preferência que colapsa e depois se autocorrige ao longo de ~15
    iterações) -- mii maior dá mais margem para essa autocorreção
    acontecer dentro do mesmo passo. Ver diário de pesquisa, Entrada 20,
    para a tabela completa e a nota metodológica sobre a reconstrução
    por-instância.

    reward_negative_fn: hook ADITIVO (04/10/2026, Entrada 64 do diário --
    estudo AOS) -- análogo a `reward_positive_fn` para o caso SEM melhora.
    None (default) usa exatamente `reward_negative` de bandit.py (Eq. 13),
    idêntico a todas as validações anteriores. Existe para as políticas
    clássicas de AOS do estudo (aos_policies.py), cuja recompensa de
    insucesso é 0 e não a Eq. 13 específica do HG-UCB. Assinatura:
    fn(q_prev_escolhido, unblocked_count, exec_count) -> float.

    on_inner: callback opcional (step, inner_iter, escolhido, melhorou,
    reward, drift) -- chamado a cada iteração interna, para diagnóstico
    (ex.: medir com que frequência max_inner_iter é de fato atingido).

    ACHADO REAL #2 (01/09/2026, via execução da Tarefa #178 -- não só
    leitura): a validação completa contra as Tabelas 10-12 mostrou erro
    MUITO acima do artigo com a leitura (1) acima (x' fixo) -- Set A:
    5,76% (média-de-5) vs. 1,61% do artigo (+4,15 p.p., consistente em
    quase todas as 26 instâncias, não ruído). Hipótese levantada A PARTIR
    do próprio resultado (não antes): com x' fixo, TODAS as ~35-40
    iterações internas pós-cold-start de um mesmo passo competem contra o
    MESMO ponto de referência -- um operador tentado tarde no passo está
    sendo comparado contra o ponto ORIGINAL pós-Shake, não contra o que "x"
    já virou depois de uma melhora anterior NO MESMO passo. Isso impede
    QUALQUER composição de melhoras de operadores diferentes dentro de um
    mesmo passo (ex.: 2-opt melhora, depois Relocate melhora ainda mais EM
    CIMA disso) -- exatamente o que vnd()/gvns() fazem e bandit_vns() não
    faria sob a leitura (1). Also explica a taxa de "melhora" muito alta
    (78-98%, Entrada 5): operadores tardios continuam "conseguindo" bater
    um alvo que não avança, sem que isso signifique progresso real
    acumulado. `progressive_x_linha=True` implementa a leitura alternativa:
    x' AVANÇA para x'' a cada melhora aceita dentro do mesmo passo externo
    (ou seja, a partir da 2a iteração interna em diante, best_fn é aplicado
    ao x' JÁ atualizado, não ao ponto original do Shake) -- tornando o
    inner loop uma busca local progressiva de verdade (mais parecida com
    VND, só que a ORDEM/escolha de qual operador tentar a cada passo é
    decidida pelo bandit, não fixa). A fórmula de recompensa (Eq. 12) muda
    de significado junto: rs passa a medir o ganho INCREMENTAL desta
    iteração especificamente (delta desta melhora), não a distância até o
    ponto original do Shake -- leitura mais alinhada ao "credit assignment"
    de uma ação individual (Seção 5.4) do que comparar contra um alvo cada
    vez mais desatualizado. NÃO decidido unilateralmente: implementado como
    variante experimental para comparação lado a lado (ver
    tests/compare_progressive_variant.py), com o resultado levado ao
    pesquisador antes de substituir o default.

    reward_positive_fn: hook ADITIVO (02/09/2026, Entrada 11 do diário) --
    permite injetar uma função alternativa para o caso de feedback
    POSITIVO (Eq. 12) SEM alterar o comportamento default desta função:
    None (default) usa exatamente `reward_positive` de bandit.py, idêntico
    a todas as validações já feitas e sincronizadas (fixo/progressivo,
    Tarefa #178) -- nenhum resultado anterior é afetado por esta adição.
    Usado por `tests/diag6_reward_normalization.py` para testar
    `reward_positive_normalized` (rs relativo, não em unidades de custo
    absolutas) sem duplicar a lógica do loop interno num script à parte
    (que arriscaria divergir sutilmente do bandit_vns() real). O caso de
    feedback NEGATIVO (Eq. 13, `reward_negative`) não tem hook equivalente
    porque já é adimensional -- ver docstring do módulo bandit.py.

    drift_reset_enabled: hook ADITIVO (06/09/2026, Entrada 28 do diário --
    Direção 2/Tarefa #179, ablação do mecanismo de drift-reset via ADWIN)
    -- True (default) preserva EXATAMENTE o comportamento já validado em
    toda a Tarefa #178: quando `bandit.update()` sinaliza drift, a PRÓXIMA
    iteração reseta o bandit e sorteia uma vizinhança aleatória (Algoritmo
    3, linhas 7-9). False faz esta função IGNORAR o sinal de drift
    inteiramente -- `bandit.update()` continua rodando o ADWIN e
    retornando o sinal bruto normalmente (nenhuma mudança em bandit.py),
    só a REAÇÃO a esse sinal (`pending_drift = True`) é desativada. Por
    isso `on_inner` continua recebendo o valor real de `drift` mesmo com
    `drift_reset_enabled=False` -- permite medir quantas vezes o ADWIN
    TERIA disparado mesmo na variante que não reage, sem instrumentação
    extra. Ver `tests/ablation_randomness_bandits.py` e
    `tests/diag15_ablacao_randomness.py` para as outras duas ablações
    (cold-start, ε-greedy) e o desenho experimental completo.
    """
    if neighborhoods is None:
        neighborhoods = NEIGHBORHOODS
    if bandit is None:
        bandit = HGUCBBandit(n_operators=len(neighborhoods), rng=rng)
    if max_inner_iter is None:
        max_inner_iter = 8 * len(neighborhoods)
    if reward_positive_fn is None:
        reward_positive_fn = reward_positive
    if reward_negative_fn is None:
        reward_negative_fn = reward_negative

    x = [list(r) for r in initial_routes]
    custo_x_linha_ref = instance.solution_cost(x)  # só usado se o inner loop nunca rodar (steps_max=0)
    best_x = x
    best_cost = custo_x_linha_ref
    t0 = time.perf_counter()
    step = 0
    pending_drift = False

    while step < steps_max:
        if time_max is not None and (time.perf_counter() - t0) >= time_max:
            break

        x_linha = shake(instance, x, rng, neighborhoods)
        custo_x_linha = instance.solution_cost(x_linha)

        inner_iter = 0
        while True:
            inner_iter += 1

            if pending_drift:
                bandit.reset_drift()
                escolhido = rng.randrange(len(neighborhoods))
                pending_drift = False
            else:
                escolhido = bandit.select()

            # snapshot PRÉ-update -- usado tanto na seleção quanto na
            # recompensa (q_{t-1}, Nt(u), nt(k) -- ver bandit.py)
            q_antes = bandit.q_values()
            unblocked_count = len(bandit.unblocked_pool(q_antes))
            exec_count = bandit.n[escolhido] + 1
            q_prev_escolhido = q_antes[escolhido]

            _, best_fn, _ = neighborhoods[escolhido]
            melhorou_na_vizinhanca, x_dupla_linha = best_fn(instance, x_linha)

            improvement = False
            if melhorou_na_vizinhanca:
                custo_x_dupla = instance.solution_cost(x_dupla_linha)
                if custo_x_dupla < custo_x_linha - 1e-9:
                    x = x_dupla_linha
                    improvement = True
                    if custo_x_dupla < best_cost - 1e-9:
                        best_cost = custo_x_dupla
                        best_x = x_dupla_linha

            if improvement:
                reward = reward_positive_fn(custo_x_linha, custo_x_dupla, unblocked_count, exec_count)
            else:
                reward = reward_negative_fn(q_prev_escolhido, unblocked_count, exec_count)

            if improvement and progressive_x_linha:
                # ACHADO REAL #2 (ver docstring) -- x' avança para o x''
                # recém-aceito, para que a PRÓXIMA iteração interna deste
                # MESMO passo externo continue a busca A PARTIR da melhora
                # já encontrada, em vez de competir de novo contra o ponto
                # original (pré-melhora) do Shake.
                x_linha = x_dupla_linha
                custo_x_linha = custo_x_dupla

            drift = bandit.update(escolhido, reward)
            if drift and drift_reset_enabled:
                pending_drift = True

            if on_inner is not None:
                on_inner(step + 1, inner_iter, escolhido, improvement, reward, drift)

            sem_melhora_e_q_baixo = (not improvement) and max(bandit.q_values()) < bandit.b_t
            if sem_melhora_e_q_baixo:
                break
            if inner_iter >= max_inner_iter:
                break  # teto de segurança -- ver docstring, "ACHADO REAL"

        step += 1
        if on_step is not None:
            on_step(step, best_cost)

    return best_x
