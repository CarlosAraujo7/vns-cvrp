# Resumo do estado atual da pesquisa — Tarefa #178

*Compilado em 04/09/2026, a partir de releitura verbatim de todo o `diario_pesquisa_novo.md` (Entradas 1-20), não de memória/paráfrase de sessões anteriores. Cobre o estado da investigação até este ponto; a Entrada 21 (diagnóstico em andamento no momento em que este resumo foi escrito) será registrada separadamente quando concluída.*

## Objetivo da Tarefa #178

Reproduzir de boa-fé o bandit **HG-UCB** (Kalatzantonakis, Sifaleras & Samaras, 2023, *Expert Systems with Applications*, DOI 10.1016/j.eswa.2022.118812) acoplado a uma VNS para CVRP, validar essa reprodução contra os números publicados (Tabelas 7-13 do artigo, benchmark CVRPLIB: conjuntos A/B/CMT/X, 74 instâncias), e decidir se/como ele se torna a base da contribuição própria deste projeto para o CEC 2027 (Tarefas #179-180, ainda não iniciadas).

## 1. Baseline GVNS clássico (Entradas 1-3)

Antes de qualquer bandit, foi implementado e validado um GVNS clássico (sem aprendizado de operador) contra a coluna "VNS" do artigo. Essa etapa já produziu achados reais próprios: a necessidade de tratar A/B (exigem contagem EXATA de veículos) diferente de X/CMT (só exigem um limite inferior) — um bug de suposição herdada corrigido nesta sessão —, e um bug real no operador Relocate(2,0), encontrado por derivação manual linha a linha contra o pseudocódigo do artigo (o mesmo método de auditoria manual reaplicado depois na Entrada 14). O baseline final validado ficou muito próximo da coluna "VNS" do artigo (Entrada 13: nosso GVNS clássico = 3,35% de gap médio em A/B/X, contra 3,39% do artigo — diferença de -0,04pp, dentro do ruído esperado).

## 2. Escolha da direção de contribuição (Entrada 4)

O pesquisador escolheu perseguir as 3 direções de contribuição propostas, não uma única.

## 3. Implementação do bandit e a saga de validação inicial (Entradas 5-8)

* **Entrada 5:** primeira implementação do Bandit VNS. Achado: a condição de parada do loop interno do artigo (`max(q_values) < b_t`) é matematicamente impossível de disparar com k=5 operadores e b_t=0,2 (softmax de 5 valores nunca fica abaixo de 1/5). Isso levou à introdução de `max_inner_iter` — um parâmetro de segurança NOSSO, fora do artigo, default 8×kmax=40 — que só voltaria a ser calibrado 15 entradas depois (Entradas 19-20, ver seção 7).
* **Entrada 6:** comparação de modos de leitura de p̂ — resultado empatado.
* **Entrada 7:** primeira validação completa (74 instâncias) com "x' fixo" — achado real #2.
* **Entrada 8:** validação completa com "x' progressivo" (correção sobre a Entrada 7) — **achado real #3, o problema central de toda a investigação seguinte:** o bandit, do jeito que estava, piorava a busca em vez de melhorá-la (o oposto do que o artigo relata). Gemini sugeriu 3 diagnósticos para investigar a causa.

## 4. Diagnosticando o achado #3 (Entradas 9-11)

* **Entrada 9:** os 3 diagnósticos do Gemini executados — **achado real #4:** identificado o mecanismo — "colapso do softmax": com `alpha=5,0` (valor do artigo) e a recompensa da Eq. 12 em unidades absolutas de custo, o softmax(H) satura para quase one-hot (>0,99) em praticamente toda iteração (99,6% das vezes), neutralizando os dois mecanismos de exploração do algoritmo (filtro b_t e ε-greedy).
* **Entrada 10:** varredura de alpha — `alpha=0,001` chega a superar nosso próprio GVNS clássico.
* **Entrada 11:** varredura fina + amostra ampliada (A/B/X) — nuance importante: o conjunto X colapsa em valores de alpha bem mais baixos que A/B, sugerindo que a causa raiz é um passo mais profunda que "só" o valor do hiperparâmetro: falta de normalização da recompensa.

## 5. A correção decisiva (Entradas 12-13)

* **Entrada 12 — achado real #6, o mais importante desta investigação:** implementada `reward_positive_normalized()` (proposta NOSSA, fora do artigo) — divide a recompensa da Eq. 12 pelo custo da solução corrente, tornando-a relativa/adimensional. Com essa única mudança, **`alpha=5,0` — o valor LITERAL da Tabela 8 do artigo, sem alterar nenhum hiperparâmetro** — deixa de colapsar catastroficamente (99,6%→12,5% de iterações colapsadas) e passa a ter desempenho consistente entre conjuntos. `alpha=0,5` (não-literal) fica marginalmente melhor ainda (3,39% vs. 3,64% de gap médio na amostra de 8 instâncias) — os **dois candidatos** que dominam toda a investigação desde então.
* **Entrada 13 — achado real #7:** os dois candidatos validados nas 74 instâncias completas. A distância para o nosso próprio GVNS clássico cai monotonicamente ao longo de toda a investigação: **+1,89pp (fixo) → +0,93pp (progressivo) → +0,16pp (normalizado, α=5,0) → +0,06pp (normalizado, α=0,5)** — uma redução de mais de 90%. Honestamente: nenhum dos dois candidatos SUPERA o GVNS clássico ou o "VNS" do artigo — ficam 0,02-0,16pp acima —, então o efeito positivo específico do HG-UCB sobre VNS puro (+0,29pp no artigo) ainda não foi reproduzido, só o efeito catastroficamente negativo foi corrigido.

## 6. Investigação da lacuna residual (Entradas 14-18) — quatro frentes, majoritariamente resultado nulo/negativo

* **Entrada 14 — achado real #8:** auditoria manual linha a linha revelou uma discrepância entre a Seção 5.4 do artigo (que diz que o ADWIN deveria ser alimentado com "Qvalues") e o código real (alimentado com a recompensa bruta) — e essa discrepância já estava documentada erroneamente no nosso PRÓPRIO docstring anterior como "resolvida".
* **Entrada 15 — diag7, RESULTADO NULO/DESFAVORÁVEL:** testada a alimentação do ADWIN com Qvalues (mais fiel ao texto) — reduz a taxa de drift como esperado, mas PIORA o gap%, não melhora. `adwin_feed_mode="reward"` (original) mantido.
* **Entrada 16 — achado real #9:** tracing detalhado revelou que a PRÓPRIA correção da Entrada 12 criou uma nova assimetria: a recompensa negativa (Eq. 13, inalterada) ficou ≈115x maior em magnitude típica que a positiva (agora normalizada).
* **Entrada 17 — diag9, achado real #10, RESULTADO NEGATIVO:** tentado reescalar a recompensa positiva para fechar essa assimetria — piora o gap em TODA escala testada, para os dois alphas, sem exceção nas médias agregadas. Causa mecânica clara: reescalar de volta para cima desfaz a própria propriedade que evita o colapso (Entrada 12). **Achado #9 e achado #6 estão em tensão direta** — não dá para corrigir um via reescala multiplicativa uniforme sem reabrir o outro.
* **Entrada 18:** segunda opinião do Gemini recomendou ENCERRAR a investigação da lacuna residual (ganho marginal esperado já pequeno). Como último cheque, sua hipótese (Eq. 13 também precisaria de normalização por custo) foi verificada contra o PDF original e **REFUTADA** — a Eq. 13 nunca teve termo de custo, confirmado pela terceira vez desde a Entrada 2.

Resultado desta fase: 4 frentes de investigação sobre a lacuna de 0,02-0,16pp, sem identificar nenhuma causa adicional isolável — consistente com a lacuna estar próxima do limite de ruído de uma reprodução de boa-fé com 5 seeds (nenhum teste de significância formal foi aplicado em nenhum ponto desta investigação — ressalva do próprio Gemini, registrada mas não endereçada ainda).

## 7. A frente atual: calibração de max_inner_iter (Entradas 19-20)

* **Entrada 19 — diag10, achado real #11, MISTO:** primeira calibração real do parâmetro `max_inner_iter` (nunca variado desde a Entrada 5). Confirmado estruturalmente, por contagem direta, que o loop interno SEMPRE roda até o teto configurado, qualquer que seja o valor (320/320 execuções). Substantivamente: para `alpha=0,5`, não há ganho em subir o teto (o default, 40, já é o melhor). Para `alpha=5,0`, subir o teto para 160 (4x) produz o **menor gap médio de toda a investigação (3,25%)** — superando até o melhor resultado de alpha=0,5 — mas a 3,6x o custo computacional, e sem saber ainda se o colapso do softmax (achado #6) continua presente e é só compensado, ou se de fato diminui.
* **Entrada 20 — diag11, achado real #12, o resultado mais limpo desta sub-investigação:** medido %colapso diretamente. **O colapso REALMENTE diminui com max_inner_iter maior — não é compensado.** Cai de 21,4% (mii=20) para 1,5% (mii=160), de forma monotônica nas 8 instâncias testadas, sem exceção. Isso dá uma explicação mecanística plausível (ainda não testada diretamente) para o resultado da Entrada 19: um teto maior dá mais margem para a autocorreção do operador dominante (já observada na Entrada 16) ocorrer dentro do mesmo passo.

## 8. Trabalho em andamento nesta sessão (ainda não é uma Entrada formal do diário)

Com a aprovação do pesquisador, está em execução o **diagnóstico 12**: expande a amostra de 8 para 16 instâncias (mantendo as 8 originais + 8 novas, incluindo a primeira instância CMT desta sub-investigação), nas mesmas 8 configurações (2 alphas × 4 valores de mii), para reduzir ruído antes de decidir os próximos passos sobre a Entrada 20. **Nota de transparência operacional:** o processo em segundo plano foi interrompido por um evento externo do ambiente (não um erro do código) faltando 1 de 16 instâncias; está sendo concluído agora por uma retomada direcionada só da instância faltante, com verificação de determinismo (as linhas já calculadas antes da interrupção foram reproduzidas exatamente). Resultado final e Entrada 21 do diário virão logo em seguida, com apresentação ao pesquisador via `AskUserQuestion`.

## 9. Estado atual e decisões pendentes do pesquisador

Nenhum default foi alterado unilateralmente em nenhum ponto desta investigação — disciplina mantida em 20 entradas consecutivas. Os dois candidatos seguem sendo, desde a Entrada 13:

* **`alpha=0,5` / `max_inner_iter=40` (default atual):** validado nas 74 instâncias completas (+0,06pp vs. GVNS clássico), nunca apresentou colapso de softmax em nenhum diagnóstico, custo computacional baixo.
* **`alpha=5,0` / `max_inner_iter=160` (valor literal do artigo, candidato reaberto pela Entrada 19-20):** melhor gap% já visto na amostra de 8 instâncias (3,25%), agora com explicação mecanística (menos colapso, não mais volume compensando), mas **ainda não validado nas 74 instâncias completas** e a ~3,6x o custo computacional.

Decisões que continuam explicitamente do pesquisador, não desta análise: qual candidato (se algum) adotar como novo default; se/quando validar o vencedor da Entrada 20 (ou o resultado do diagnóstico 12) na bateria completa de 74 instâncias; e se a Tarefa #178 deve ser considerada suficientemente investigada para avançar às Tarefas #179-180, ou se ainda vale continuar refinando antes disso — decisão já adiada por 5 entradas consecutivas (16 a 20) à medida que cada nova frente de investigação abriu uma pergunta nova genuína.

## 10. Disciplina metodológica seguida (para referência)

Todo achado reportado (positivo, nulo ou negativo) veio de execução real, nunca só de leitura de código ou raciocínio teórico. Toda correção foi implementada como hook aditivo/retrocompatível, com regressão verificada por reprodução de valor exato antes de prosseguir. Toda mudança de default foi leva ao pesquisador via `AskUserQuestion`, nunca aplicada unilateralmente. O Gemini foi usado como segunda opinião independente em pontos-chave (Entradas 9, 13→14, 16→18, 19→20), sempre apresentado o fato de forma neutra antes de revelar qualquer conclusão prévia — e suas sugestões foram testadas contra a fonte primária ou empiricamente, não aceitas por autoridade (a hipótese da Entrada 18 foi refutada; a hipótese da Entrada 20 foi confirmada). A fonte primária (o PDF do artigo) foi reconferida diretamente múltiplas vezes ao longo da investigação, tratando inclusive a própria documentação anterior deste projeto como hipótese a reverificar, não fato estabelecido (achado real #8 nasceu exatamente de encontrar uma discrepância na nossa própria nota anterior).
