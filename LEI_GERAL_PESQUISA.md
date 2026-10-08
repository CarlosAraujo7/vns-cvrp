# Lei Geral para Fundamentar Pesquisa Rumo a um Periódico de Elite

**Versão 1.0 — 21/08/2026**

**Origem:** este documento não é teoria abstrata. Cada regra abaixo foi extraída de um projeto de pesquisa real — uma learnheurística (Shapley + ML + GRASP/ILNS) para o Problema de Dispersão Equitativa (EDP), documentada entrada a entrada em `diario_pesquisa.md` ao longo de 28 entradas e cerca de 9 dias de trabalho intensivo (13-21/08/2026). Cada item de jurisprudência (Parte II) tem pelo menos um caso concreto, com números reais, por trás — nada aqui é princípio genérico sem precedente. É um documento novo e autônomo: não substitui nem edita `GUIA_PESQUISA_GERAL.md` (o guia estrutural já existente do projeto de origem, mantido intocado); funciona como camada complementar a ele.

## Como este documento se relaciona com os demais (leia antes do resto)

Se o seu projeto já tem, ou vier a ter, os documentos abaixo, use-os assim:

- **Guia específico do veículo-alvo** (ex.: `GUIA_PESQUISA_EJOR.md`) — a instância aplicada: números de referência, periódico, checklist calibrado para o seu campo.
- **Guia estrutural genérico** (ex.: `GUIA_PESQUISA_GERAL.md`) — o manual de "o que precisa existir": anatomia do artigo, padrão de rigor empírico, padrão de redação e estilo, fundamentação de literatura, arquitetura de arquivamento de 3 níveis, checklist de prontidão para submissão, protocolo operacional para agentes de IA. Responde "o que precisa existir num artigo rigoroso" e "como organizar isso".
- **Este documento (`LEI_GERAL_PESQUISA.md`)** — não repete o manual estrutural. Responde a uma pergunta diferente: **quais erros se repetem quando se persegue rigor de verdade, e como pegá-los antes que cheguem ao manuscrito?** É jurisprudência operacional — precedente real, não princípio abstrato — e traz um protocolo específico para quando um agente (humano ou IA) assume um projeto de pesquisa **já em andamento**, iniciado por outra sessão ou outra pessoa.

Se você só tem este documento — por exemplo, começando um projeto novo, sem os outros dois ainda — ele funciona sozinho: a Parte I e a Parte V cobrem o mínimo estrutural necessário. Mas ele foi pensado para ser lido **depois** de `estado_atual.md` do projeto em questão, se existir: este documento ensina padrões; `estado_atual.md` diz o que já é verdade agora.

## A cláusula mais importante — leia isto duas vezes

Chamar isto de "lei" é um risco deliberado, e este parágrafo é o contrapeso. No próprio projeto de origem, na Entrada 2 do diário, um guia quase idêntico a este foi aplicado como bloqueador absoluto — travando o sequenciamento do projeto até 100% de conformidade com um item específico. O pesquisador humano corrigiu isso explicitamente na Entrada 3: um guia é direcionamento forte, não uma trava que impede qualquer outro avanço até satisfação completa. Foi exatamente essa correção que motivou a criação do guia generalizado do projeto de origem — o mesmo espírito se aplica, com ainda mais força, a este documento. Um agente que lê "lei" e conclui "logo, bloqueante" já cometeu o primeiro erro que este documento existe para prevenir.

Portanto: siga as regras abaixo por padrão. Quando um item for deliberadamente adiado, pulado, ou você discordar dele para o caso concreto à sua frente, **registre isso** — no diário de pesquisa do projeto (Parte III) — em vez de (a) ignorá-lo silenciosamente ou (b) tratá-lo como impedimento de tudo o mais. Priorização é julgamento de quem pesquisa, não deste documento. Na dúvida entre seguir a letra ou usar o julgamento, pergunte a quem pesquisa — não presuma a leitura mais literal.

---

## Parte I — Princípios inegociáveis

Diferente do resto deste documento (normas fortes, com desvio permitido e registrado), os itens abaixo não têm exceção justificável por conveniência de prazo ou escopo. Violá-los não é um atraso corrigível — é o tipo de erro que só aparece depois de publicado, quando já é tarde.

1. **Nenhuma citação entra no código ou no manuscrito sem verificação independente contra fonte primária** — nunca aceite uma citação de um agente de pesquisa, de um script antigo, ou até de você mesmo numa sessão anterior, sem reconferir autor/ano/veículo/DOI diretamente na fonte. Ver Parte II.3.
2. **Nenhum número "final" existe sem rastro até o script/arquivo exato que o produziu.** Se você não consegue apontar a origem de um número em 30 segundos, ele não está pronto para citação.
3. **Um resultado nulo, negativo, ou inconveniente para a narrativa é reportado com o mesmo rigor que um resultado favorável.** Não se relança o experimento em busca de significância; não se esconde a parte desfavorável atrás da favorável. Ver Parte II.5.
4. **Nunca declarar reprodutibilidade ou determinismo sem ter, de fato, executado de novo e comparado.** "Deveria ser determinístico" não é evidência. Ver Parte II.9.
5. **Nenhuma exclusão de dado é silenciosa.** Toda exclusão (instância, seed, outlier, período, sujeito) tem uma linha registrada explicando o motivo, com o dado bruto preservado fora do cálculo agregado, mas disponível.
6. **Decisões de enquadramento, escopo ou direção estratégica do artigo são de quem pesquisa, não do agente.** Um agente pode e deve recomendar, com posição clara e justificada — mas a decisão final sobre o que o artigo afirma, para qual periódico mirar, ou se um resultado nulo se torna a tese central, pertence a quem assina o trabalho. Ver Parte II.5, II.8 e Parte IV.
7. **Nada que representa esforço de pesquisa é apagado sem preservação prévia.** Nota estruturada — data, motivo, conteúdo relevante — antes de qualquer exclusão, nunca depois.

---

## Parte II — Jurisprudência: os padrões que se repetem

Cada item traz a regra, seguida de um ou mais casos reais (com números) do projeto de origem, e uma nota de "como detectar" quando aplicável. As referências a "Entrada N" apontam para o diário do projeto de origem, citadas aqui como evidência de que a regra não é hipotética — não como algo que você precise consultar para usar este documento.

### II.1 — O padrão "config drift": um mecanismo muda, um consumidor não acompanha

**A regra:** toda vez que uma função central muda de assinatura, um valor de referência é centralizado, ou um protocolo de execução é atualizado, procure ativamente por todos os lugares que ainda dependem do comportamento antigo — não assuma que a mudança "só" afeta onde você a fez. Trate qualquer script fora do caminho de execução ativamente exercitado como suspeito até validado por execução real.

**Casos reais (o mesmo padrão, pelo menos 5 vezes, no mesmo projeto):** uma função de busca passou a retornar 6 valores em vez de 5; um harness de teste continuou desempacotando 5, e o erro ficou silenciosamente engolido por um `try/except` genérico — só apareceu numa auditoria dedicada, bem depois. Um script de ablação teve exatamente o mesmo bug de desempacotamento em 12 pontos de chamada diferentes, porque nunca tinha sido de fato executado ponta a ponta desde que a assinatura mudara em outro lugar. O mesmo script de ablação tinha um dicionário de valores de referência hardcoded cobrindo só instâncias pequenas — sob o escopo real (instâncias maiores), toda consulta teria retornado silenciosamente vazio/NaN, sem erro nenhum. Um módulo de análise de resultados tinha a lista de métodos comparados hardcoded como só 2 de 5 métodos de comparação oficiais — três deles, introduzidos numa fase posterior do projeto, nunca apareceram em nenhuma tabela comparativa formal, sem que ninguém percebesse até uma reconferência direta dos números publicados contra o dado bruto. O mesmo módulo usava uma constante de corte de tempo antiga depois que o protocolo real passou a escalonar o tempo por tamanho de instância — analisando as instâncias maiores a uma fração pequena do orçamento que de fato usaram.

**Como detectar:** ao mudar uma assinatura, um formato de dado, ou um valor centralizado, faça uma busca textual por todos os consumidores antes de considerar a mudança terminada — não confie em lembrar todos os lugares que usam aquilo. Trate qualquer script que não faz parte do pipeline principal, rodado com frequência, como tendo alta probabilidade de estar dessincronizado — a única forma de saber é executá-lo de verdade.

### II.2 — Validar por execução real, nunca só por leitura de código

**A regra:** revisão de código cuidadosa sistematicamente não pega bugs de interação entre duas funções que, isoladamente, parecem corretas. Antes de confiar em qualquer mudança — especialmente qualquer coisa que toque persistência, agregação ou migração de dado — execute de verdade contra dado real (ou uma cópia dele) e compare o resultado.

**Caso real (o mais sério do projeto):** uma migração de formato de progresso (de arquivo único compartilhado para um arquivo por tarefa) parecia correta numa primeira leitura: "migre só as combinações completas, é a opção mais conservadora." Só ao rodar o ciclo completo — migração seguida de reagregação — contra uma cópia de dados reais do projeto é que ficou evidente que combinações *parciais*, existentes de verdade num arquivo legado, desapareceriam silenciosamente, porque a etapa de agregação tratava o novo formato como única fonte da verdade e reescrevia tudo do zero a cada chamada. A lógica "conservadora" escondia perda de dado exatamente na interação entre duas funções que, cada uma isoladamente, estava certa.

**Caso relacionado:** uma hipótese de correção testada sob a mesma disciplina — "adicionar um piso a um hiperparâmetro do modelo deveria evitar degenerescência em instâncias pequenas" — era plausível e apoiada por literatura adjacente. Testada de fato contra 6 combinações reais de instância/seed, piorou o resultado nas 6 (num caso, quase pela metade). A hipótese foi revertida, e o próprio teste — não a hipótese original — virou o achado documentado.

**Como detectar:** antes de marcar qualquer correção como concluída, pergunte: "eu de fato rodei isso contra um caso não trivial e comparei o resultado, ou só li o código de novo e me convenci?" Se a resposta for a segunda, ainda não terminou.

### II.3 — Toda citação trazida por pesquisa delegada é uma alegação não verificada até prova em contrário

**A regra:** um agente de pesquisa, ou uma busca sua feita rápido demais, pode trazer autor/ano/veículo corretos com um detalhe errado — e o detalhe errado costuma ser exatamente o que ninguém confere: o DOI. Confirme cada citação nova contra pelo menos uma fonte primária (página do editor, DOI resolvido de fato, DBLP, arXiv, registro RePEc/IDEAS) antes que ela entre em código ou manuscrito — mesmo dentro do mesmo projeto que já fez essa verificação antes, mesmo quando a citação "parece" familiar.

**Casos reais:** ao longo do projeto de origem, pelo menos 5 citações distintas tiveram um erro real pego exatamente dessa forma — não um evento isolado. Entre elas: autorias trocadas em três baselines portados, incompatíveis com o algoritmo de fato implementado; uma referência a um trabalho "em preparação" que nunca foi localizável e precisou ser substituída; e o caso mais instrutivo — uma citação para a prova de complexidade computacional do problema central do artigo, com autor, ano, periódico, volume e páginas corretos, mas com um **DOI que resolvia para um artigo completamente diferente e não relacionado** (um artigo sobre avaliação de risco de crédito, em vez do artigo sobre o problema em questão). O erro só foi pego porque o DOI foi de fato resolvido e o resultado lido, não apenas copiado.

**Regra derivada:** quando um detalhe específico — tipicamente o DOI, às vezes a faixa de página — não puder ser confirmado por nenhuma fonte consultada, diga isso explicitamente no próprio texto/documento ("DOI não confirmado — não usar até confirmar via [fonte]") em vez de arredondar para "confirmado" por conveniência. Isso já evitou, no projeto de origem, que uma citação com detalhe genuinamente pendente fosse tratada como fechada antes da hora.

### II.4 — Amostra pequena mente: desconfie de tendência antes que o poder estatístico chegue

**A regra:** um resultado direcional obtido com poucas unidades de amostra não é um resultado fraco — é um resultado que ainda não existe. Não reporte como achado até ter poder estatístico real; se precisar reportar antes disso, rotule explicitamente como piloto/indicativo, nunca como conclusão.

**Caso real (o mais nítido do projeto):** um piloto com 3 instâncias mostrou o método proposto "nunca perdendo" contra um método de comparação (2 vitórias, 0 derrotas, 1 empate em confrontos por instância), com melhoria média de +0,12%. Lido de forma ingênua, é um resultado favorável. Ampliado para as 23 instâncias reais disponíveis na mesma faixa — poder estatístico real, mesmo protocolo — o quadro virou 3 vitórias, 8 derrotas, 12 empates; teste de Wilcoxon não significativo (p=0,42); *effect size* (Cohen's d) de −0,001, praticamente zero; intervalo de confiança de 95% da melhoria cruzando zero. O "nunca perde" de 3 instâncias era regressão à média de uma amostra pequena, não um efeito real — o resultado de poder estatístico real era um nulo genuíno, não uma versão "mais fraca" do resultado favorável.

**Regra derivada:** ao ver uma tendência favorável numa amostra pequena, a pergunta certa não é "isso é bom?" — é "quantas unidades eu preciso para confiar nessa direção?" Trate qualquer alegação baseada em amostra pequena (calibre o limiar para o seu domínio; poucas unidades de dígito único é sempre pouco) como hipótese a testar em escala, nunca como conclusão a citar.

### II.5 — Resultado nulo ou desfavorável é dado, não fracasso

**A regra:** quando a evidência acumulada, com rigor real, aponta para "o mecanismo proposto não ajuda" ou "o método não vence o estado da arte", isso não é motivo para esconder, insistir até mudar, ou abandonar o projeto — é, em si, um resultado científico honesto e citável, especialmente em veículos que valorizam rigor metodológico sobre uma narrativa de vitória.

**Caso real:** a pergunta central de um projeto inteiro de aprendizado de máquina aplicado a otimização — "a orientação por valor de Shapley/ML melhora a busca sobre um prior estrutural simples, dado exatamente o mesmo motor de busca?" — foi respondida, com poder estatístico real cobrindo toda a faixa de tamanho testável, como um nulo, sem diferença detectável em nenhuma faixa. Separadamente, numa bateria completa de 141 instâncias contra 5 métodos de comparação externos, o método proposto nunca venceu o mais forte deles em nenhum dos 141 confrontos diretos, e ficou em 5º lugar de 6 por posto médio (teste de Friedman com diferença real entre métodos, p≈4,5×10⁻¹⁸ — não é ruído). Diante disso, ao ser perguntado diretamente e com franqueza se fazia sentido continuar, quem pesquisava recebeu uma avaliação honesta — não branda, não inflada — e decidiu, não o agente, reformular a contribuição central do artigo: em vez de "propomos um método que vence", "demonstramos rigorosamente quando e se orientação aprendida importa numa learnheurística" — um achado majoritariamente nulo, tratado com o mesmo rigor metodológico que um resultado positivo receberia. Isso não descartou o projeto: o mesmo método batia dois dos cinco métodos de comparação de forma consistente na maior parte da faixa de tamanhos, e tinha uma vantagem real — posteriormente qualificada, não anulada, ver II.12 — numa faixa específica.

**Regra derivada:** a honestidade sobre um nulo não é o fim de uma contribuição — frequentemente é o começo de uma mais defensável. Mas a decisão de reformular a tese central em torno de um nulo é de quem assina o trabalho, nunca uma imposição do agente (Princípio I.6).

### II.6 — Delegação com ceticismo: confiar mas verificar, mesmo sob interrupção

**A regra:** ao delegar uma investigação a um agente independente, ou a vários em paralelo, instrua explicitamente para reconstruir a partir de fonte primária/execução real — não para confirmar o que já está no código ou nos comentários existentes, o que evita viés de confirmação. Ao receber o resultado, reverifique pessoalmente as alegações mais consequentes antes de aceitá-las. Se um agente for interrompido no meio (limite de sessão, erro, timeout), não descarte nem aceite cegamente o trabalho parcial — leia os artefatos que ele de fato produziu e re-derive você mesmo os números centrais antes de decidir o que aproveitar.

**Casos reais:** uma auditoria de fidelidade de métodos de comparação usou 5 agentes independentes, um por método, cada um re-derivando a citação de origem por busca própria — sem confiar no texto já presente no código — e verificando corretude e fidelidade algorítmica de forma separada. Resultado: as 5 citações foram confirmadas de forma independente (uma delas, inclusive, uma citação alternativa já descartada internamente foi ativamente reconferida como de fato errada, em vez de aceita só por já estar arquivada); nenhum bug de corretude grave nos 5; mas 2 lacunas reais de fidelidade ao algoritmo do artigo original, corretamente sinalizadas como decisão editorial a resolver — não como bug a corrigir por conta própria. Separadamente, uma rodada de fundamentação por literatura usou 6 agentes, um por componente do método, com instrução explícita de ler o código-fonte primeiro e pesquisar de forma ampla, não só confirmar as citações já sugeridas — um deles encontrou uma lacuna teórica real: um estimador rotulado "não enviesado" no próprio docstring do código, que na verdade violava a hipótese de uniformidade exigida pela referência citada para justificá-lo. Numa rodada posterior, um dos agentes foi interrompido por limite de sessão no meio de uma investigação empírica; em vez de descartar o trabalho parcial, os arquivos de produção já editados, os scripts de apoio e a evidência bruta que ele havia deixado foram lidos e reconferidos diretamente — incluindo resolver, de forma independente, uma aparente inconsistência nos próprios dados do agente antes de aceitar o achado como válido.

### II.7 — Escopo de refatoração: não empacotar melhoria não pedida junto com o que foi pedido

**A regra:** ao refatorar ou otimizar algo — paralelizar, migrar formato, acelerar — preserve exatamente o comportamento computacional pré-existente, mesmo quando esse comportamento pareça uma decisão questionável, a menos que corrigi-lo tenha sido explicitamente pedido. Misturar "o que foi pedido" com "o que eu também notei que poderia melhorar" torna impossível atribuir efeito a causa depois, e arrisca quebrar algo fora do escopo autorizado.

**Caso real:** ao paralelizar um script antes sequencial, notou-se que uma das variantes testadas dependia do estado residual do gerador de números aleatórios deixado pela variante anterior — nunca fazia seu próprio reseed, um comportamento pré-existente, não necessariamente intencional. A decisão registrada foi preservar exatamente esse comportamento durante a paralelização, documentando-o como comportamento existente, não como algo corrigido — porque corrigi-lo silenciosamente, no meio de uma mudança de infraestrutura não relacionada, teria produzido resultados numéricos diferentes sem que ninguém tivesse pedido ou percebido essa mudança específica.

**Regra derivada:** ao notar algo que merece correção mas está fora do que foi pedido, registre como proposta separada (Parte III) — não a aplique de carona.

### II.8 — Portões de aprovação antes de custo alto ou decisão estratégica irreversível

**A regra:** antes de comprometer horas, ou dias, de computação, ou antes de fechar uma decisão de enquadramento/direção do artigo, apresente a decisão explicitamente — com opções concretas e custo estimado — e obtenha aval explícito. Depois de obtido, não repita o pedido para cada passo subsequente na mesma direção já autorizada; uma leitura razoável de uma autorização ampla cobre passos subsequentes na mesma linha, mas essa leitura deve ficar registrada como interpretação, não apresentada como confirmação literal.

**Casos reais:** antes de uma execução estimada em dezenas de horas, uma rodada de pilotos e auditoria adicional foi feita primeiro, encontrando 2 bugs reais que, sem essa pausa, teriam contaminado a execução inteira. Uma decisão de excluir as instâncias mais caras — responsáveis por 92% do custo total estimado — de uma primeira rodada de bateria completa foi sinalizada como pergunta, não decidida sozinha; a resposta obtida incluiu a escolha exata e uma instrução direta de como comandar a execução. Um escopo de estudo de ablação em escala, estimado em dezenas de horas sequenciais, foi proposto com números concretos e só ativado como comportamento padrão do script depois de aprovação explícita. E, de forma simétrica: diante de um quadro de resultados majoritariamente nulos, 4 opções concretas de caminho a seguir foram apresentadas explicitamente — investigar uma causa-raiz específica, encerrar e reformular, rodar mais um teste específico, ou expandir o escopo já coberto — em vez de presumir qual seria escolhida.

### II.9 — Engenharia de determinismo é obrigação própria em pesquisa computacional/ML

**A regra:** "mesma semente aleatória, mesmo resultado" não é automático — cada fonte de não determinismo (ordem de soma em ponto flutuante sob múltiplas threads, paralelismo de biblioteca numérica, e principalmente critérios de parada baseados em tempo de parede) precisa ser diagnosticada empiricamente, não assumida corrigida por uma única correção. Depois de qualquer correção de determinismo, teste de novo se a variância é de fato zero, não apenas menor.

**Caso real (o mais instrutivo do gênero neste projeto):** uma primeira correção — forçar determinismo interno de uma biblioteca de ML — pareceu resolver completamente o problema, com variância zero em várias instâncias testadas. Testada numa instância adicional, porém, a mesma semente reexecutada de forma isolada produziu valores diferentes em repetições sucessivas — a correção era necessária, mas não suficiente. Restringir threads de álgebra linear (BLAS/OpenMP) a 1 ajudou parcialmente, mas não eliminou toda a variação em todos os casos. A causa-raiz real só apareceu depois, numa investigação dedicada: todo o pipeline usava laços limitados por tempo (`enquanto tempo decorrido < orçamento`), o que faz o número de vezes que o gerador de números aleatórios é consultado depender da velocidade/carga da máquina naquele instante — divergindo a sequência inteira de decisões a partir do primeiro ponto de divergência, mesmo com semente idêntica. Confirmado de forma direta: 5 execuções repetidas do mesmo caso (mesma instância, mesmo código, mesma máquina) produziram 5 valores finais diferentes. A correção certa foi um modo opt-in de contagem fixa de iterações, não de tempo, para replay bit-exato de uma execução já registrada — deliberadamente **não usado no protocolo oficial de comparação entre métodos**, que precisa continuar baseado em tempo por justiça de comparação entre métodos com velocidades de convergência diferentes.

**Regra derivada:** trate reprodutibilidade como uma alegação empírica a testar — reexecute e compare byte a byte — nunca como uma propriedade que uma correção plausível garante por si só. E não deixe um mecanismo de replay/depuração vazar para dentro do protocolo de comparação oficial: são propósitos diferentes, e confundi-los compromete a justiça da comparação.

### II.10 — Higiene de paralelismo e checkpoint/resume

**A regra:** ao paralelizar um processo sequencial de longa duração, espelhe uma arquitetura já validada em produção no mesmo projeto, em vez de inventar um mecanismo novo do zero. Prefira escrita atômica de progresso por tarefa (arquivo temporário seguido de renomeação) a um handle compartilhado mutável entre processos — especialmente sob o modelo de criação de processo "spawn" (Windows), onde o estado do processo pai não é herdado. Limite threads de bibliotecas numéricas por processo antes de qualquer import que as carregue, para evitar sobrecarga quando N processos competem pelos mesmos núcleos — mas meça o efeito antes de assumi-lo, porque pode ser neutro em cargas pequenas e só relevante sob contenção real.

**Casos reais:** medido diretamente, não assumido, restringir threads de álgebra linear a 1 não mudou nada numa execução isolada de carga pequena, mas foi cerca de 27% mais rápido sob concorrência real de múltiplos processos simultâneos (12,63s vs. 17,40s no total) — adotado como comportamento padrão automático justamente por não piorar nenhum caso medido e ajudar no caso que motivava a mudança. Ao paralelizar um script de ablação antes sequencial, a granularidade de tarefa (instância+semente, não instância+semente+variante) foi escolhida deliberadamente mais grossa do que o ideal, para manter a lógica extraída o mais próxima possível do laço sequencial original — aceitando como custo um recomputo mais caro no pior caso, em troca de reduzir o risco de mudar resultados sem querer ao refatorar. Manter o parsing de argumentos de linha de comando isolado do bloco de entrada principal do script evitou um bug de integração real: a mesma função também era chamada programaticamente por outro script do próprio projeto, sem passar por linha de comando — se o parsing estivesse dentro dela, teria tentado interpretar os argumentos de quem a chamou.

### II.11 — Escolha de periódico-alvo é pergunta factual, não de ambição

**A regra:** "deveríamos mirar um periódico melhor?" não se responde por vontade — se responde pesquisando de fato a bibliometria real do periódico atual, o estado real do sistema de classificação que a pergunta pressupõe (que pode estar mudando), e o que periódicos genuinamente superiores tipicamente exigem, contra o que a evidência do próprio projeto de fato sustenta.

**Caso real:** diante da pergunta "dá para mirar mais alto?", a resposta não foi opinativa. Envolveu confirmar que o sistema de classificação por letras que a própria pergunta pressupunha estava sendo descontinuado pelo órgão responsável naquele mesmo ciclo de avaliação — mudando o que a pergunta original sequer significava —, confirmar a bibliometria real do periódico já-alvo (fator de impacto, indicador de ranking de área, posição relativa no campo — já um periódico de elite, perto do teto prático da área) e reconhecer que periódicos genuinamente acima dele tipicamente exigem contribuição teórica mais profunda ou resultado empírico decisivamente positivo, nenhum dos dois descrevendo o projeto depois da reformulação honesta em torno de um resultado majoritariamente nulo (II.5). Mirar mais alto não era sustentado pela evidência disponível — não porque fosse desejável recusar ambição, mas porque a pergunta, investigada de verdade, tinha uma resposta factual diferente da premissa.

### II.12 — Isole a variável que você quer atribuir: confundimento aparece tanto no desenho da comparação quanto ao empacotar duas correções juntas

**A regra:** antes de atribuir um efeito observado a uma causa específica, pergunte se outra variável andou junto com ela sem querer. Isso acontece de duas formas recorrentes: (a) no desenho de uma comparação, quando duas fontes de variação mudam ao mesmo tempo entre os grupos comparados; (b) ao aplicar duas correções/mudanças simultaneamente e depois querer saber qual delas produziu o efeito observado. Nos dois casos, a resposta é a mesma: isole experimentalmente antes de escrever a alegação causal no texto.

**Casos reais (três, em momentos diferentes do mesmo projeto):** um achado inicial favorável a um prior estrutural simples — batendo por quase 1 ponto percentual um componente de aprendizado de máquina mais sofisticado — comparava esse prior rodando num motor de busca mais ajustado contra o componente de ML rodando no motor de busca oficial "cru": a fonte da orientação de busca e o motor de busca em si variavam ao mesmo tempo, confundindo a comparação original. Isolar o motor de busca — usando exatamente o mesmo para as duas variantes — foi o que permitiu reabrir a pergunta de forma controlada (e, quando reaberta dessa forma, o resultado favorável ao prior não se confirmou — ver II.5). Duas correções foram adotadas juntas numa mesma rodada — um mecanismo de diversificação por estagnação e uma correção de determinismo de uma biblioteca de ML — e reduziram a variância entre sementes a zero; só um desenho fatorial dedicado, numa rodada posterior, isolou que o mecanismo de diversificação sozinho já era suficiente, e a correção de determinismo não mostrava efeito mensurável adicional quando o primeiro já estava ativo. E um resultado favorável ao método proposto numa faixa de tamanho grande, já usado como evidência positiva, foi parcialmente reexplicado quando um teste dedicado, com tempo fixo e igual para todos os tamanhos, mostrou que boa parte da vantagem vinha do fato de que, no protocolo oficial, instâncias maiores também recebiam mais tempo absoluto de execução — não de uma propriedade estrutural ligada ao tamanho em si.

**Como detectar:** ao desenhar uma comparação, liste explicitamente tudo que muda entre os grupos comparados — se mais de uma coisa muda, a comparação não isola nada. Ao adotar duas correções na mesma rodada porque as duas pareceram boas ideias ao mesmo tempo, registre explicitamente que o efeito combinado, não o de cada uma, é o que foi medido — e trate a atribuição causal individual como pendência, não como fato, até um teste fatorial dedicado resolvê-la.

---

## Parte III — Arquitetura de documentação de 3 níveis (versão operacional)

Esta arquitetura está detalhada por completo, com templates prontos para copiar, no guia estrutural genérico do projeto de origem (`GUIA_PESQUISA_GERAL.md`, Seção 6 e Anexos A-C). Resumo funcional aqui, para este documento funcionar sozinho se necessário:

- **Diário cronológico** (`diario_pesquisa.md` ou equivalente) — todo experimento, bug, decisão explícita de quem pesquisa, e tentativa rejeitada, na ordem em que aconteceu. Nunca editado retroativamente — uma correção é sempre uma entrada nova que referencia a antiga. Cada entrada numerada e citável.
- **Estado atual consolidado** (`estado_atual.md` ou equivalente) — só os números válidos agora, com a fonte exata de cada um. Contém uma seção permanente **"O que NÃO usar"**: todo número, citação ou abordagem superada migra para lá com motivo de uma linha, em vez de simplesmente desaparecer. Essa seção só cresce, nunca é limpa.
- **Relatório de evolução** (`relatorio_evolucao.md` ou equivalente) — narrativo, para alguém se orientar em minutos sem ler o histórico completo.

**Por que isto compensou, de forma mensurável, no projeto de origem:** a seção "O que NÃO usar" evitou reintroduzir um erro de citação já corrigido. Uma auditoria de prontidão, feita muitas entradas depois do início do projeto, conseguiu reconstruir com precisão total o que já tinha sido verificado e o que ainda faltava, só a partir do estado consolidado e uma busca dirigida no diário — sem depender da memória de ninguém. E o relatório narrativo, quando ficou parado por mais de uma semana enquanto o diário continuava crescendo, foi facilmente identificado como desatualizado (a data no próprio documento denunciou) e reescrito por completo, sem perder nenhum fato, porque o diário — a fonte primária — nunca parou de crescer.

**Regra adicional deste documento, não coberta explicitamente no guia estrutural:** quando um agente de IA está prestes a rodar uma investigação, sua ou delegada, que repete algo já tentado, a seção "O que NÃO usar" e o diário devem ser consultados **antes**, não depois — repetir uma tentativa já rejeitada, sem saber que já foi rejeitada, é o desperdício mais evitável de todos.

---

## Parte IV — Protocolo de herança: assumindo um projeto de pesquisa já em andamento

Esta parte é conteúdo específico deste documento, para o momento em que um agente — você, lendo isto agora — recebe um projeto com histórico já acumulado, iniciado por outra sessão ou outra pessoa.

1. **Leia o estado atual consolidado primeiro, nunca o diário completo a frio.** É a fonte da verdade; o diário completo é para quando você precisar da justificativa completa de um número específico, não como ponto de partida.
2. **Leia a seção "O que NÃO usar" antes de propor qualquer coisa nova.** Ela existe precisamente para evitar que você reproponha, de boa fé, algo já testado e rejeitado.
3. **Identifique o status da decisão de enquadramento atual** — qual é a tese central declarada do trabalho agora, e há quanto tempo isso está fechado — antes de sugerir mudança de direção. Se o projeto já fechou deliberadamente um ciclo de testes comparativos (II.5/II.8) e decidiu pivotar para redação, reabrir esse ciclo por iniciativa própria, sem antes reapresentar a pergunta a quem pesquisa, desrespeita uma decisão de enquadramento que já foi tomada explicitamente.
4. **Não assuma que um número antigo, citado numa conversa anterior ou lembrado do seu próprio conhecimento geral sobre o domínio, ainda vale.** Confirme contra o estado atual consolidado.
5. **Ao encontrar algo que parece um erro óbvio no trabalho já feito, verifique se já não é uma decisão registrada e justificada** — por exemplo, uma escolha de design mantida deliberadamente "como está" por uma razão de custo-benefício já documentada — antes de "corrigi-la". Ver II.7 sobre escopo de refatoração.
6. **Continue o diário; não reinicie a numeração nem crie um documento paralelo.** A entrada seguinte é a próxima em sequência, referenciando as anteriores quando relevante.
7. **Ao final da sua sessão de trabalho, sincronize o estado atual consolidado com a última entrada do diário.** Não deixe a dessincronização para a próxima sessão — é a causa mais comum de retrabalho entre sessões.

---

## Parte V — Checklist mínimo (versão de bolso)

Para o checklist completo de prontidão para submissão — com todos os itens de rigor empírico, redação e mecânica do veículo — use o guia estrutural do seu projeto (Seção 7, se seguir a mesma numeração do projeto de origem), e a instância específica do seu periódico-alvo, se existir. Versão condensada, para checagem rápida no meio de uma sessão de trabalho:

- [ ] Toda citação nova desta sessão foi verificada contra fonte primária — não só aceita de uma busca ou de um agente (II.3)?
- [ ] Toda mudança de código foi validada por execução real, não só leitura — inclusive nos consumidores que podem ter ficado desatualizados (II.1, II.2)?
- [ ] Algum resultado desta sessão é baseado em amostra pequena e está sendo tratado como conclusão em vez de piloto (II.4)?
- [ ] Algum resultado nulo ou desfavorável desta sessão está sendo reportado com o mesmo rigor que um favorável, sem retestar em busca de significância (II.5)?
- [ ] Alguma decisão de escopo, enquadramento ou direção foi tomada sem aval explícito de quem pesquisa, quando deveria ter sido perguntada (II.8)?
- [ ] Algum efeito observado nesta sessão está sendo atribuído a uma causa sem isolar outras variáveis que mudaram junto (II.12)?
- [ ] O estado atual consolidado e o diário estão sincronizados antes de encerrar a sessão?

---

## Parte VI — Cláusula de revisão

Este documento não é definitivo. Ele deveria crescer da mesma forma que o diário de qualquer projeto individual cresce: quando um padrão novo e genuinamente reutilizável — não específico de um único projeto — aparecer numa pesquisa futura, adicione-o à Parte II como um novo item numerado, com o caso real que o motivou. Nunca reescreva um item existente para "generalizar mais" sem preservar o caso concreto por trás dele — um princípio sem um caso real por trás é só uma opinião com formatação de lei.

### Modelo para adicionar um novo item de jurisprudência

```markdown
### II.N — <regra em uma frase, nomeando o comportamento a seguir ou evitar>

**A regra:** <1-2 frases, imperativas, domínio-agnósticas — vale para qualquer área, não só a do projeto que a gerou>

**Caso real:** <o que aconteceu, com números/fatos concretos, e como foi pego>

**Como detectar (se aplicável):** <um sinal prático para reconhecer a situação da próxima vez>
```
