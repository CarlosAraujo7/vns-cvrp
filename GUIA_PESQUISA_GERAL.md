# Guia Geral para Desenvolvimento de Pesquisa e Redação Científica

**Versão 1.0 (generalizada) — 13/08/2026**

Este documento é uma generalização do "Guia Padrão para Desenvolvimento de Artigos Científicos em Otimização — foco EJOR" (mantido, sem alterações, em `GUIA_PESQUISA_EJOR.md`), reescrito para funcionar em qualquer área de pesquisa e qualquer periódico/veículo-alvo, não só otimização computacional/EJOR. Onde o guia original tinha uma regra amarrada a uma editora ou subárea específica, este documento generaliza para o princípio por trás da regra, e indica onde preencher a especificidade do seu próprio projeto. Trate os dois documentos como camadas complementares: este é o núcleo reutilizável; `GUIA_PESQUISA_EJOR.md` é uma instância aplicada (a deste projeto), útil também como exemplo concreto de como preencher cada seção.

**Uso previsto:** referência operacional para agentes de IA e pesquisadores humanos, consultada em três momentos — ao planejar o desenho de um estudo (Seção 3), continuamente durante o desenvolvimento (Seção 6), e antes de declarar qualquer trabalho pronto para submissão (Seção 7).

**Regra de aplicação — leia antes do resto:** as normas deste guia são pontos de referência fortes, não bloqueadores absolutos. Siga-as por padrão; um agente de IA não deve travar todo o progresso de um projeto esperando conformidade de 100% com cada item antes de deixar qualquer outra coisa avançar. Priorização e sequenciamento do trabalho são julgamento do pesquisador, não do guia. Quando um item for deliberadamente adiado ou pulado, registre isso (Seção 6) em vez de ignorá-lo silenciosamente ou tratá-lo como impedimento de tudo o mais.

### Índice

1. O veículo-alvo: entender o público editorial e se adaptar a ele
2. Anatomia completa do trabalho
3. Padrão de rigor empírico
4. Padrão de redação e estilo
5. Fundamentação teórica e revisão de literatura
6. Protocolo de arquivamento e memória do projeto
7. Checklist final de prontidão para submissão
8. Protocolo operacional direto para agentes de IA
9. Como montar sua própria lista de leituras recomendadas
10. Anexos (templates prontos para copiar)

---

## 1. O veículo-alvo: entender o público editorial e se adaptar a ele

Antes de escrever — e periodicamente durante o projeto, porque a resposta pode mudar — responda:

- Qual a seletividade/taxa de aceitação aproximada do veículo-alvo (periódico, conferência, ou outro)? Revisão pesada/múltiplas rodadas costuma ser a norma em veículos fortes, não um sinal de trabalho fraco — mas isso não reduz a barra do que precisa estar correto já na primeira versão.
- O veículo impõe uma estrutura rígida (ex.: IMRaD) ou é flexível quanto à organização? Existe limite de página, formal ou informalmente reportado?
- O que costuma causar rejeição sem avaliação completa ou revisão pesada, especificamente nessa comunidade? Padrões que se repetem entre áreas: contribuição não explicitada (aplicar um método conhecido a um problema novo sem avanço sobre o método em si); problema tratado de forma isolada, sem ancoragem na literatura/famílias de problemas já estabelecidas; elementos obrigatórios de submissão faltando ou incompletos; qualidade de redação/clareza estrutural abaixo do padrão; comparação fraca ou não diferenciada com a literatura mais próxima; rigor metodológico abaixo do que a comunidade espera implicitamente (mesmo quando não é uma regra escrita do veículo).
- Reconfirme as instruções oficiais vigentes diretamente na fonte primária (página do periódico/conferência) na fase de submissão — políticas mudam, e nenhum guia estático substitui essa checagem final.

*Exemplo aplicado a este projeto (EJOR): ver `GUIA_PESQUISA_EJOR.md`, Seção 1.*

---

## 2. Anatomia completa do trabalho

Estrutura macro esperada para um trabalho com estudo empírico/experimental, adaptável a qualquer área. Cada título de seção deve comunicar o conteúdo por si só. Prefira títulos curtos, substantivos, sem verbo.

| # | Seção | O que precisa conter, sem exceção |
|---|---|---|
| — | **Título** | Conciso, informativo, sem abreviação não padronizada, sem termos de marketing — deixe o resultado falar no resumo. Inclua os termos de busca que um leitor da área usaria. |
| — | **Resumo/Abstract** | Autocontido, factual, objetivo + resultado principal + conclusão. Todo número citado aqui precisa ser idêntico ao que aparece no corpo do texto. Nunca promete mais do que a seção de Resultados entrega, nunca omite uma ressalva honesta só para soar mais forte. |
| — | **Destaques/resumo visual (se o veículo pedir)** | Formato "elevator pitch": resultado novo + método novo, sem jargão não explicado. |
| — | **Palavras-chave** | O suficiente para ampliar a superfície de busca sem duplicar literalmente o título; se o veículo tiver uma lista/taxonomia oficial de categorias, o primeiro termo deve vir dela. |
| 1 | **Introdução** | Contextualização + motivação + lacuna identificada na literatura + lista explícita de contribuições + roteiro do restante do trabalho. A lista de contribuições é um dos poucos lugares em que marcadores são bem-vindos no corpo do texto. |
| 2 | **Revisão de literatura** | Ver Seção 5. Pode vir embutida na Introdução em trabalhos curtos; separada é mais seguro em veículos mais extensos/seletivos. |
| 3 | **Definição/formalização do problema** | Premissas explícitas e numeradas, notação/terminologia definida na primeira ocorrência, qualquer posicionamento teórico (ex.: o problema como caso especial ou generalização de algo já estabelecido) com justificativa completa, não apenas enunciada. |
| 4 | **Método/abordagem proposta** | Descrição completa o bastante para reprodução independente. Custo/complexidade de cada componente quando aplicável. Justificativa de design — por que essa abordagem, não só "descrição do que foi feito". |
| 5 | **Configuração experimental/empírica** | Ver Seção 3. Ambiente e instrumentos exatos, dados com proveniência, métodos de referência com protocolo de execução, métricas, protocolo de verificação. |
| 6 | **Resultados e discussão** | Ver Seção 3. Organizado por argumento lógico (não pela ordem cronológica em que os experimentos foram feitos). Toda tabela/figura referenciada e comentada no texto. |
| 7 | **Conclusão** | Síntese interpretativa dos achados (não repetição mecânica dos números), contribuições reaproveitáveis fora do escopo específico, limitações honestas e concretas, trabalhos futuros que decorrem diretamente das limitações identificadas. |
| 8 | **Apêndices** | Resultados completos/desagregados, material que apoia mas não é essencial ao argumento principal do corpo do texto. |
| — | **Elementos obrigatórios de fechamento** | Ver 2.9. Não são opcionais e não podem ficar com placeholder na submissão. |

### 2.9 Elementos obrigatórios de fechamento

Toda editora/veículo tem sua própria lista de elementos obrigatórios de fechamento. Levante essa lista cedo, não na hora da submissão. Itens recorrentes entre editoras/veículos: declaração de contribuição de cada autor; declaração de conflito de interesse; declaração de uso de IA generativa quando aplicável (nome da ferramenta, propósito, extensão da supervisão humana — a IA nunca é listada como autora); declaração de disponibilidade de dados/código com link real e persistente (nunca um placeholder no momento da submissão); financiamento (ou declaração explícita de ausência); agradecimentos. Trate um placeholder esquecido como bloqueador de prontidão (Seção 7), não como detalhe menor.

*Exemplo aplicado a este projeto (checklist específico Elsevier/EJOR): ver `GUIA_PESQUISA_EJOR.md`, Seção 2.9.*

---

## 3. Padrão de rigor empírico

Esta é a seção central do guia — a que mais precisa ser recalibrada para a sua área específica, porque os números de referência abaixo vieram originalmente de otimização computacional. Trate-os como ponto de partida razoável, não como padrão universal fixo.

### 3.1 Métodos de comparação (baselines)

Regra geral: um trabalho empírico raramente se sustenta com menos de 2 pontos de comparação; 3 é mais robusto. Prioridade, do mais ao menos importante: (1) o método mais forte já publicado para o problema mais próximo do seu — nunca comparar só contra métodos desatualizados sem justificar explicitamente por que um método mais forte e aplicável ficou de fora; (2) uma referência de "verdade fundamental" quando existir (valor ótimo/exato, ground truth, limite teórico), mesmo que só para um subconjunto tratável dos dados; (3) um segundo método independente, estruturalmente diferente do primeiro. Orçamento (tempo, dados, tentativas, recursos) igual entre métodos comparados, salvo justificativa explícita. Parâmetros: mesmo protocolo de ajuste para todos os métodos, ou os parâmetros originalmente publicados por cada um — nunca ajustar fino só o método proposto e usar valores padrão para os demais. Método de comparação reimplementado (código original indisponível) exige validação de fidelidade contra números publicados, reportada no trabalho.

### 3.2 Dados/instâncias/amostras de teste

Mínimo de 2 fontes de proveniência distinta; 3 quando o domínio permitir: (i) dados próprios, gerados ou coletados sistematicamente variando os eixos relevantes ao domínio; (ii) um benchmark ou conjunto público já estabelecido na literatura mais próxima; (iii) quando existir, um conjunto de um domínio mais geral/abstrato, para testar generalização. Documente a proveniência de cada fonte e reporte estatísticas descritivas do conjunto. Nunca descarte dados silenciosamente — toda exclusão precisa de uma linha explicando o motivo, com o dado bruto preservado (fora do cálculo agregado, mas disponível). Disponibilize os dados publicamente quando possível, incluindo geradores/scripts de coleta.

### 3.3 Testes estatísticos

Nunca reporte só média/mediana sem teste de significância, nem só p-valor sem *effect size*. "Estatisticamente significativo" e "praticamente relevante" são coisas diferentes; discuta as duas.

Passo 1 — verifique os pressupostos do teste (normalidade, homogeneidade de variância, independência) antes de escolher qual usar. Passo 2 — escolha o teste adequado ao formato dos dados: paramétrico quando os pressupostos se sustentam (ex. teste t pareado), não paramétrico quando não (ex. Wilcoxon signed-rank para dados pareados — frequentemente o caso default em métricas de otimização/simulação). Passo 3 — para 3 ou mais métodos, use um teste global (ex. Friedman) seguido de comparações par a par **com correção obrigatória para múltiplas comparações** (ex. Holm-Bonferroni). Passo 4 — sempre reporte, independentemente do teste: estatística e p-valor, effect size, intervalo de confiança (bootstrap é uma opção robusta quando os pressupostos distribucionais são incertos), contagem de vitórias/empates/derrotas quando aplicável, média **e** mediana.

Regras adicionais: declare a hipótese e o teste antes de rodar a comparação — nunca troque de teste em busca de significância. Um resultado não significativo também é científico e deve ser reportado com a mesma honestidade que um positivo. Ao correlacionar uma métrica com uma propriedade dos dados, verifique confundimento antes de qualquer leitura causal — correlação não é prova de causalidade.

### 3.4 Estudo de componentes (ablação)

Decomponha o método/abordagem proposta em decisões de design identificáveis. Para cada variante testada, reporte o efeito (médio/mediano), o que ela ataca especificamente, e o custo (uma variante pode custar muito mais por um ganho desprezível — isso é informação relevante). **Investigue e explique resultados surpreendentes, não apenas os relate** — às vezes a explicação fortalece a conclusão central do trabalho. "Nenhum componente testado ajudou além do básico" é uma conclusão de ablação legítima e publicável, desde que honestamente reportada e tecnicamente explicada — não é motivo para descartar o experimento ou insistir até "dar certo".

### 3.5 Sensibilidade e robustez

Para qualquer componente estocástico, amostral ou sujeito a variação experimental, um número mínimo de repetições/replicações independentes é necessário para caracterizar a variabilidade real (10 é uma referência razoável em otimização estocástica — recalibre para o que for padrão no seu domínio). Reporte variabilidade por unidade de análise, não só agregada — uma unidade individual de altíssima variância escondida numa média agregada é um erro comum. Diagnostique a causa raiz quando uma unidade específica se comportar de forma qualitativamente diferente do restante, em vez de ignorar. Explore sistematicamente o espaço de parâmetros/hiperparâmetros relevantes, nunca só os valores extremos.

### 3.6 Verificação independente

Trate como etapa metodológica de primeira classe, não como passo opcional de higiene: recompute resultados-chave de forma independente do caminho de código/processo que os gerou; valide contra checagens de sanidade ou ground truth quando existirem; ao encontrar um resultado matematicamente impossível ou contraditório (ex. um resultado supostamente melhor que um limite teoricamente comprovado), trate isso como sinal de um bug no seu próprio processo antes de qualquer outra hipótese — não se resolve por repetição, se resolve por investigação.

### 3.7 Ambiente e protocolo de reprodutibilidade

Documente, sem exceção, tudo que um terceiro precisaria para reproduzir o experimento: ambiente computacional/instrumental usado, versões de ferramentas/software, orçamento (tempo/recursos) declarado e igual entre métodos comparados, e qualquer fator que possa enviesar a comparação (ex. métodos rodando em ambientes ou condições diferentes).

### 3.8 Checklist numérico mínimo (ajustável ao seu domínio)

Valores de referência — ajustáveis livremente para cima; para baixo, só com justificativa explícita no texto: ≥2 métodos de comparação (3 recomendado); ≥2 fontes de dados de proveniência distinta (3 recomendado); ≥10 repetições independentes quando houver componente estocástico; teste estatístico com effect size e intervalo de confiança em toda comparação central da argumentação; resultados completos (não só agregados) disponíveis em anexo; todo número citado no resumo/introdução/conclusão idêntico ao corpo do texto.

*Exemplo aplicado a este projeto (valores calibrados para otimização computacional/EJOR): ver `GUIA_PESQUISA_EJOR.md`, Seção 3.*

---

## 4. Padrão de redação e estilo

### 4.1 Tom e voz

Voz ativa para as ações dos autores ("propomos", "avaliamos"); voz passiva reservada para descrever o objeto de estudo. **Organize por argumento lógico, nunca pela ordem cronológica da pesquisa** — um relatório narra "primeiro tentamos X, depois percebemos Y"; um trabalho científico reorganiza o mesmo conteúdo por resultado e argumento (o lugar certo para a cronologia completa é o diário de bordo da Seção 6, não o texto final). **Calibre toda afirmação à evidência** — prefira "o resultado é consistente com X", "sugere", "indica" a "isso prova X"; reserve termos absolutos para o que foi formalmente demonstrado. Evite hedging redundante (uma ressalva por afirmação é rigor; três formas de dizer a mesma ressalva é excesso). Consistência de tempo verbal: resultados obtidos neste trabalho no passado, fatos/definições gerais no presente. Evite excesso de listas com marcadores no corpo do texto — prefira prosa argumentativa fora de destaques/lista de contribuições/limitações. Nunca encadeie um título de seção direto com um subtítulo sem um parágrafo de transição.

### 4.2 Consistência terminológica

**Um conceito, um termo — do início ao fim do trabalho.** Um erro recorrente (e subestimado, especialmente em texto revisado com apoio de IA) é a variação lexical excessiva: usar sinônimos diferentes para a mesma entidade técnica na tentativa de "variar o vocabulário". Em texto técnico isso obscurece se o autor está falando da mesma coisa, e cria ambiguidade grave quando o sinônimo escolhido já tem outro significado técnico estabelecido no próprio texto. Precisão terminológica pesa mais que variedade estilística. Antes de finalizar qualquer seção, liste os substantivos técnicos centrais e confirme que cada um tem um único referente em todo o texto.

### 4.3 Notação e definições

Defina todo símbolo/termo técnico na primeira ocorrência, mesmo os aparentemente óbvios. Use ambientes formais (definição/proposição/teorema/lema, ou o equivalente da sua área) para qualquer afirmação formal, nunca embutida em prosa corrida. Notação consistente entre seções — confira isso especificamente ao integrar conteúdo escrito em momentos diferentes do projeto.

### 4.4 Tabelas e figuras

Siga o estilo tipográfico padrão do seu veículo-alvo (ex. `booktabs` em Elsevier/OR). Legenda de tabela acima da tabela; legenda de figura abaixo da figura. Toda tabela/figura referenciada e comentada no texto antes ou logo após aparecer — nunca "solta". Tabelas muito largas: resuma no corpo, mande o detalhe completo para o apêndice.

### 4.5 Citações

Estilo consistente (autor-data ou numérico — escolha um e não misture). Evite sequências de mais de 3 citações seguidas sem diferenciar a contribuição de cada uma. Autocitação: enquadre trabalho anterior dos mesmos autores como contexto histórico, nunca inflado para parecer um corpo de literatura externo maior do que é.

---

## 5. Fundamentação teórica e revisão de literatura

"Boa fundamentação atual" significa, de forma concreta e verificável: **recência** — a revisão precisa incluir trabalho publicado nos últimos 2-3 anos na área específica, não só os clássicos fundacionais; **estrutura argumentativa, não lista** — organize por vertentes temáticas, cada uma com uma frase de síntese e uma frase de lacuna, não apenas "Autor (ano) fez X. Autor (ano) fez Y."; **gap explícito** — a lacuna que o trabalho preenche deve ser afirmada em uma frase clara; **diferenciação explícita contra os 2-3 trabalhos mais próximos**, com uma frase dizendo exatamente o que muda, idealmente quantificada; **ancoragem em famílias de problemas/linhas já estabelecidas** — sempre que o trabalho puder ser posicionado como caso especial, generalização ou interseção de algo já consolidado na literatura, isso fortalece substancialmente a leitura editorial, mesmo quando a aplicação em si é nova.

---

## 6. Protocolo de arquivamento e memória do projeto

### 6.1 Por que isso existe

Agentes de IA operam em janelas de contexto finitas e sessões que terminam. Sem um protocolo de arquivamento, três falhas se repetem sistematicamente em projetos de pesquisa longos: (a) um número superado é citado de novo porque a sessão atual não sabia que ele tinha sido corrigido; (b) uma abordagem já testada e rejeitada é retestada do zero; (c) uma decisão que o pesquisador humano tomou explicitamente é revertida sem querer, por não estar registrada em lugar nenhum. Este protocolo elimina essas três falhas — e produz, como subproduto, o material de que a Introdução, a Conclusão e a resposta a revisores vão precisar depois.

**Regra geral:** documentar o processo não é trabalho adicional ao lado da pesquisa — é parte da pesquisa. Um achado (incluindo um bug ou um caminho que não funcionou) só conta como concluído quando está registrado.

### 6.2 Arquitetura em três níveis

Mantenha três documentos distintos, com propósitos que não se sobrepõem.

**Nível 1 — Diário cronológico de pesquisa** (`diario_pesquisa.md` ou equivalente): registro bruto, completo, somente-acréscimo, de tudo que foi tentado, na ordem em que aconteceu. Nunca editar ou apagar uma entrada antiga — uma correção posterior é sempre uma nova entrada, datada, que referencia a entrada que corrige. Cada entrada tem número sequencial estável, citável em qualquer outro documento (inclusive em comentários de código). Registre sem exceção: todo experimento com resultado citável depois; todo bug encontrado, mesmo sem mudar a conclusão principal; toda decisão explícita do pesquisador humano; toda tentativa rejeitada ou superada, com o motivo técnico. Cada entrada, no mínimo: contexto/pergunta, o que foi feito, resultado (com referência ao dado exato que o produziu), decisão (ADOTADO / SUPERADO por [ref.] / REJEITADO porque [razão] / EM ABERTO), impacto no texto final.

**Nível 2 — Estado atual consolidado** (`estado_atual.md` ou equivalente): documento curto e limpo, só os números/afirmações válidos agora, a "fonte da verdade" a consultar antes de escrever ou citar qualquer número — nunca o diário completo. Toda vez que um número muda, a versão antiga não desaparece — move para uma seção permanente "O que NÃO usar", com motivo de uma linha e ponteiro para a entrada do Nível 1 correspondente; essa seção só cresce. Cada número tem fonte explícita (script/arquivo exato). Sincronizado com o Nível 1 dentro de, no máximo, uma sessão de trabalho.

**Nível 3 — Relatório de evolução do projeto** (`relatorio_evolucao.md` ou equivalente): narrativo, para leitura humana direta, permitindo que alguém que não acompanhou o dia a dia se oriente em ~10 minutos. Conteúdo esperado: resumo executivo (idealmente tabela antes/depois se houve reformulação), o que mudou e por quê, estado atual, comparativo antes×depois, contribuições, figuras a partir de dados reais (nunca ilustrações genéricas), status de escrita e pendências, fontes dos números. Atualizado em marcos do projeto, não a cada experimento.

### 6.3 Regras de ouro

Nunca delete arquivos, pastas ou versões de código/trabalho que representem esforço de pesquisa sem antes preservar, em uma nota estruturada, tudo que tiver valor científico ou metodológico — escreva a nota de preservação (data, motivo, conteúdo relevante extraído) antes de qualquer exclusão, nunca depois. Não apague/sobrescreva trabalho superado — comente com nota datada explicando por quê, ou mova para uma pasta de histórico explicitamente fora do caminho de reprodução dos resultados finais, mas preservada para auditoria. Todo comentário que documenta uma decisão inclui data e, quando existir, o número da entrada correspondente no diário. Um achado inesperado (bug, anomalia, contradição) é tratado como resultado a documentar, nunca como algo a corrigir silenciosamente e seguir em frente. Toda decisão que reverte ou supera uma abordagem anterior nomeia explicitamente o que está sendo substituído e por quê.

### 6.4 Quando escrever uma entrada

Sem esperar "ter tempo depois": um experimento termina e produz um número potencialmente citável; um bug é encontrado, mude ou não a conclusão vigente; o pesquisador humano toma uma decisão explícita sobre direção, escopo ou interpretação; uma tentativa é abandonada; um número antes tratado como "final" muda por qualquer razão.

---

## 7. Checklist final de prontidão para submissão

Rode este checklist explicitamente antes de declarar um trabalho "pronto para submissão". Todo item não aplicável precisa de uma justificativa de uma linha, não de uma omissão silenciosa.

**Conteúdo científico**
- [ ] Métodos de comparação suficientes, protocolo de execução justo documentado (Seção 3.1)
- [ ] Fontes de dados de proveniência distinta (Seção 3.2)
- [ ] Testes de significância com effect size e IC em toda comparação central (Seção 3.3)
- [ ] Estudo de componentes/ablação cobrindo as principais decisões de design (Seção 3.4)
- [ ] Repetições/replicações suficientes, variabilidade reportada (Seção 3.5)
- [ ] Protocolo de verificação independente aplicado e reportado (Seção 3.6)
- [ ] Ambiente/instrumentos documentados por completo (Seção 3.7)
- [ ] Resultados completos disponíveis em apêndice, não só agregados
- [ ] Todo número do resumo/introdução/conclusão idêntico ao corpo do texto
- [ ] Resultados neutros/negativos reportados com a mesma honestidade que os positivos
- [ ] Limitações concretas e específicas na Conclusão (não genéricas)

**Fundamentação e posicionamento**
- [ ] Revisão de literatura cobre publicações recentes (últimos 2-3 anos), não só clássicos
- [ ] Lacuna (gap) explicitada em uma frase
- [ ] Diferenciação explícita contra os 2-3 trabalhos mais próximos
- [ ] Problema ancorado em família(s)/linha(s) já estabelecida(s), quando possível

**Redação e estrutura**
- [ ] Estrutura macro completa (Seção 2), títulos de seção autoexplicativos
- [ ] Seção de Resultados organizada por argumento, não por ordem cronológica de descoberta
- [ ] Passagem de consistência terminológica feita (Seção 4.2)
- [ ] Notação/definições consistentes entre todas as seções, inclusive material adicionado depois
- [ ] Toda tabela/figura no estilo do veículo-alvo, comentada no texto
- [ ] Citações com estilo único e consistente

**Mecânica de submissão do veículo-alvo**
- [ ] Elementos obrigatórios de fechamento completos, sem placeholder (Seção 2.9 — lista específica do seu veículo)
- [ ] Disponibilidade de dados/código com link real, não placeholder
- [ ] Guia de autores vigente reconferido diretamente na fonte oficial antes do envio

**Arquivamento (Seção 6)**
- [ ] Nível 2 (estado atual) sincronizado com a última entrada do Nível 1
- [ ] Nenhum número citado no trabalho está na lista "O que NÃO usar" do Nível 2
- [ ] Nível 3 (relatório de evolução) atualizado com o estado final antes da submissão

*Checklist com valores numéricos específicos aplicados a este projeto: ver `GUIA_PESQUISA_EJOR.md`, Seção 7.*

---

## 8. Protocolo operacional direto para agentes de IA

1. Antes de rodar uma bateria de experimentos, confirme que o desenho já inclui métodos de comparação e fontes de dados suficientes (Seção 3.1-3.2) — desenhar incompleto e "completar depois" costuma custar uma re-execução inteira. Isto é uma recomendação forte, não um bloqueio: se o pesquisador preferir rodar algo incompleto para gerar sinal rápido, registre a decisão (item 11) e prossiga.
2. Antes de declarar qualquer número como "final", registre a decisão no diário cronológico (Nível 1) com data, número de entrada, e a fonte de dados exata.
3. Nunca sobrescreva um número antigo silenciosamente — mova-o para "O que NÃO usar" no Nível 2, com o motivo.
4. Nunca delete arquivos ou pastas de esforço de pesquisa sem antes preservar o conteúdo relevante em nota estruturada.
5. Ao encontrar um resultado inesperado (bug, anomalia, contradição, resultado nulo), trate como achado a documentar — nunca como algo a corrigir silenciosamente ou a esconder.
6. Antes de reportar significância estatística, confirme: teste adequado ao formato dos dados e ao número de métodos comparados, correção de comparações múltiplas se houver 3+ métodos, effect size e IC reportados junto ao p-valor.
7. Nunca aceite um resultado empírico sem alguma forma do protocolo de verificação independente da Seção 3.6.
8. Ao escrever ou revisar a seção de Resultados, organize por argumento lógico — se um parágrafo narra "primeiro tentamos X, depois Y", reescreva por resultado antes de considerar o texto pronto.
9. Ao final de qualquer sessão de trabalho substancial, verifique que o Nível 2 está sincronizado com a última entrada do Nível 1 — não deixe a dessincronização para a próxima sessão.
10. Antes de declarar o trabalho "pronto para submissão", rode o checklist completo da Seção 7 item por item — não amostre, não confie na memória do que já foi verificado antes.
11. Quando o pesquisador humano tomar uma decisão explícita (escopo, interpretação, direção, priorização), registre-a no diário cronológico com paráfrase do que foi decidido — não dependa de lembrar isso implicitamente em turnos futuros.
12. Se este guia e uma instrução pontual do pesquisador entrarem em conflito, a instrução pontual vale para aquele caso específico — mas sinalize o conflito explicitamente em vez de silenciosamente desviar do padrão, para que a exceção fique registrada e não vire um desvio permanente não intencional.
13. **Trate as normas deste guia como orientação forte, não como lei rígida a aplicar sem julgamento.** Não converta um checklist ou uma seção numerada em um bloqueador que impede qualquer outro progresso até 100% de conformidade — isso é uma leitura mais rígida do que o próprio guia pretende. Quando em dúvida sobre o quanto insistir num item, pergunte ao pesquisador em vez de presumir a interpretação mais literal.

---

## 9. Como montar sua própria lista de leituras recomendadas

Em vez de uma lista fixa (que fica desatualizada e é específica de área), procure por, no início do projeto e sempre que retomar depois de uma pausa longa: o guia de autores oficial do seu veículo-alvo, reconferido na fonte primária; um tutorial de referência sobre desenho e relato de experimentos na sua subárea específica; uma referência metodológica para o tipo de teste estatístico que os seus dados exigem; uma revisão recente (idealmente dos últimos 1-2 anos) do seu subcampo, para calibrar o que "recente" significa nele especificamente; e, se aplicável, a política vigente do seu veículo-alvo sobre uso de IA generativa na redação.

*Lista concreta de leituras usada neste projeto (otimização computacional/EJOR): ver `GUIA_PESQUISA_EJOR.md`, Seção 9.*

---

## 10. Anexos (templates prontos para copiar)

### Anexo A — Template de entrada do diário cronológico (Nível 1)

```markdown
## [N]. <Título curto do que foi tentado/decidido> (DD/MM/AAAA)

**Contexto/pergunta:** por que isso foi feito agora, o que motivou.

**O que foi tentado:** descrição técnica objetiva.

**Resultado:** números/achado, com referência ao arquivo de dados exato (ex.: `resultado_XYZ.csv`, gerado por `script_XYZ.py`).

**Decisão:** ADOTADO / SUPERADO por [Seção N] / REJEITADO porque [razão] / EM ABERTO.

**Impacto no trabalho final:** o que muda (ou explicitamente "nada muda") no texto atual.
```

### Anexo B — Esqueleto do estado atual consolidado (Nível 2)

```markdown
# Estado atual consolidado — [nome do projeto]

Última atualização: DD/MM/AAAA. Fonte da verdade para citar números — nunca cite diretamente do diário cronológico completo.

## Resumo executivo
[2-3 frases com os números centrais válidos agora]

## Resultado central
[tabela com os números atuais + fonte de cada um]

## O que NÃO usar (números descartados/superados)
- ~~[número antigo]~~ → corrigido para [número atual] (motivo: [1 linha];
  ver Seção [N] do diário cronológico).
- [continua crescendo, nunca é limpo]

## Pendências
[o que ainda falta decidir ou escrever — não o que falta pesquisar, se já fechado]
```

### Anexo C — Esqueleto do relatório de evolução (Nível 3)

```markdown
# Relatório de Evolução do Projeto — [nome]

**Data:** DD/MM/AAAA · **Escopo:** [o que este relatório cobre]

## 1. Resumo executivo
[tabela antes/depois, se houver reformulação]

## 2. Estado anterior (se aplicável)
## 3. Motivo da mudança (se aplicável)
## 4. Estado atual
[problema, método, dados, comparações, resultados]

## 5. Comparativo antes×depois
## 6. Contribuições
## 7. Gráficos
[gerados a partir de dados reais do projeto, com script/processo reprodutível referenciado]

## 8. Status de escrita e pendências
## 9. Fontes dos números
[aponta para os arquivos/scripts exatos por trás de cada resultado citado]
```

### Anexo D — Checklist de uma página (versão condensada da Seção 7)

Métodos de comparação suficientes · Fontes de dados de proveniência distinta · Testes estatísticos com effect size+IC · Estudo de componentes/ablação · Repetições suficientes · Verificação independente · Ambiente documentado · Apêndice com dados completos · Resumo/destaques batendo com o corpo · Literatura recente + gap explícito + diferenciação · Estrutura completa (Seção 2) · Consistência terminológica · Elementos obrigatórios de fechamento sem placeholder · Nível 2 sincronizado com Nível 1 · Nenhum número citado está em "O que NÃO usar".
