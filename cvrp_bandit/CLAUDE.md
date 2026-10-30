# CLAUDE.md — Projeto HG-UCB Bandit para CVRP

> Arquivo de onboarding para sessões do Claude Code neste projeto. Criado em 04/10/2026, a pedido do pesquisador, a partir da leitura completa do diário de pesquisa (Entradas 1-59, achados reais #1-#39) e dos dois documentos gerais de disciplina do workspace. Substitui o workflow anterior (Claude via Cowork, sandbox de nuvem sem acesso direto ao disco) — a partir de agora a sessão roda com acesso de shell/filesystem direto na máquina real do pesquisador.
>
> **Nota de transparência:** este arquivo é uma versão condensada-com-referências do diário, não uma duplicata literal das 344KB. Cada achado abaixo tem 2-4 linhas; números e racional completos continuam exclusivamente no diário. Esta condensação foi uma decisão de baixo risco tomada autonomamente (puramente de apresentação, sem afetar interpretação de resultado) — registrada aqui por exigência da própria disciplina do projeto de documentar esse tipo de escolha. Os números abaixo foram extraídos do diário por um agente auxiliar nesta sessão de criação deste arquivo; **para qualquer número que vá parar no paper final, confirme contra a entrada original do diário antes de usar.**

## Índice

0. [Leia isto primeiro — estado atual](#0-leia-isto-primeiro--estado-atual)
1. [Identidade do projeto](#1-identidade-do-projeto)
2. [Limites de escopo CRÍTICOS](#2-limites-de-escopo-críticos)
3. [Mapa de diretórios](#3-mapa-de-diretórios)
4. [Disciplina operacional para sessões Claude Code](#4-disciplina-operacional-para-sessões-claude-code)
5. [Referência técnica — hiperparâmetros](#5-referência-técnica--hiperparâmetros)
6. [Referência técnica — CVRPLIB e peculiaridades dos solvers](#6-referência-técnica--cvrplib-e-peculiaridades-dos-solvers)
7. [Log de pesquisa — achados reais #1 a #39](#7-log-de-pesquisa--achados-reais-1-a-39)
8. [Log de bugs encontrados e corrigidos](#8-log-de-bugs-encontrados-e-corrigidos)
9. [Estado atual — onde paramos (LEIA ISTO)](#9-estado-atual--onde-paramos-leia-isto)
10. [Apêndice A — log completo de decisões (AskUserQuestion)](#apêndice-a--log-completo-de-decisões-askuserquestion)
11. [Apêndice B — gotchas de ambiente/engenharia](#apêndice-b--gotchas-de-ambienteengenharia)

---

## REGRA MASTER (pesquisador, 08/10/2026) — vale para toda sessão

**Ciclo obrigatório: (1) plano de melhorias → (2) execução → (3) auditoria como REVISOR (EJOR/CEC), procurando as fragilidades que seriam apontadas → (4) novo plano com os achados → repetir até o trabalho estar sólido.** Nenhuma entrega é declarada pronta sem a auditoria do ciclo; registrar cada ciclo no diário (Entrada 88 em diante) com o plano, os achados da auditoria e as correções. Alvo: EJOR primeiro, CEC em segundo. Autoria/declarações: texto genérico (o pesquisador edita).

## 0. Leia isto primeiro — estado atual

- **Última entrada do diário:** Entrada 59, 16/09/2026 (achado real #39). Confirmado nesta sessão (04/10/2026) que **nada foi executado desde então** — o pesquisador ainda não rodou o teste de `--lr_model 1e-5` proposto como próximo passo.
- **ATUALIZAÇÃO (Entrada 61, 03/10/2026):** o diagnóstico do #39 foi rodado — com `--lr_model 1e-5` (graph_size=100) o custo de validação termina em 19,22 em vez de degradar (33,45 com 1e-4); achado #40. **CORRIGIDO na Entrada 63:** o pré-treinado vale 19,27 no mesmo conjunto de validação, então lr 1e-5 *evita degradação*, não melhora o modelo (diferença -0,25%, dentro do ruído), e a degradação do lr 1e-4 começa já na época 0 em todos os tamanhos. Só 1 corrida/1 seed/n=100. Kool zero-shot medido: 9,43% de gap, 58/74 viáveis no critério estrito (`results/kool_zeroshot_74.csv`). Também foi medido o **HGS-CVRP** nas 74 instâncias (`results/hgs_74_{matched,n01}.csv`): gap médio 0,035%/0,010% vs. 3,41% do HG-UCB, HG-UCB melhor em 0/71 instâncias (achado #41). A vitória sobre LKH3/OR-Tools vem de orçamentos de 0,2-10 s, não de qualidade de método. **O reescopo da Direção 1 está pendente de decisão do pesquisador (Eixo B)** — leia a Entrada 61 do diário antes de qualquer coisa.
- **ESTUDO AOS (Entradas 62-72, 04/10/2026) — Direção A.** Decisões do pesquisador: Direção A "em etapas"; Kool = contexto zero-shot; B=20.000 completo CANCELADO (custo ~34 h; família confirmatória = 12 testes); Etapa 2 = "decompor + AOS no VND". **Etapa 1 CONCLUÍDA** (B=800 e B=4.000, 74 inst. x 10 seeds x 7 políticas, pré-registro Entrada 64, resultados nas Entradas 66/68; subconjunto exploratório B=20.000 na Entrada 72): HG-UCB-norm ≈ seleção uniforme e ≈ 3 AOS clássicos (n.s. após Holm); HG-UCB literal (sem normalizar a recompensa) é ~1-1,3pp pior em 74/74; GVNS clássico (VND de ordem fixa) é ~0,5pp melhor que qualquer política dentro do loop do Algoritmo 3 (Entrada 70: ~87% das 40 aplicações/passo ocorrem depois da última melhora). **Etapa 2 CONCLUÍDA (Entrada 74, confirmatório, 21/28 testes significativos):** o término do laço interno (parar no ótimo local em vez de 40 aplicações fixas) é o fator dominante (−0,5 a −0,6pp); a ordem ALEATÓRIA de operadores bate a fixa por 0,3-0,5pp (hipótese H5 refutada); dentro do VND nenhuma política adaptativa (UCB1, ε-greedy, prob. matching, HG-UCB normalizado) supera a ordem aleatória — ε-greedy e HG-UCB são significativamente PIORES. Melhor braço = VND de ordem aleatória/aceitação best/término LO (gap 1,94% em B=4.000), 0,8pp melhor que o loop do Algoritmo 3, sem aprendizado. Pré-registro: Entrada 71 (NÃO editar). Código: `src/aos_vnd.py`, `estudo_aos/run_aos_etapa2.py`, `estudo_aos/analisar_etapa2.py`; dados `results/aos_study/etapa2_full.csv`. **Decisão do pesquisador (05/10/2026, Entrada 75): "Etapa 3 + redigir em paralelo".** **Etapa 3 CONCLUÍDA (Entrada 81, confirmatório, 13/34 testes significativos):** o conjunto estendido (10 operadores, com SWAP*) melhora o gap em ~0,5pp sob o mesmo esforço de avaliações de distância; nenhuma política adaptativa (UCB1, prob. matching, HG-UCB, com recompensa sensível a custo, término LO ou R5) supera a escolha uniforme — HG-UCB é pior em E2 (+0,11/+0,20pp); a ordem por custo é PIOR que a aleatória; hipótese positiva H10 refutada. Pré-registro: Entrada 77 (NÃO editar). Código: `src/neighborhoods_ext.py` (1.600 testes contra força bruta), `src/instancia_contadora.py`, `estudo_aos/run_aos_etapa3.py`, `estudo_aos/analisar_etapa3.py`; dados `results/aos_study/etapa3_full.csv` (19.240 execuções; houve uma interrupção/retomada — Entrada 80). **Flanco aberto: as AOS clássicas não foram ajustadas** (decisão do pesquisador pendente sobre uma Etapa 4 de robustez com ajuste em instâncias geradas, disjuntas das 74). **Redação em paralelo:** `paper/main.tex` (IEEEtran, rascunho NÃO compilado — sem LaTeX na máquina), `paper/refs.bib` (só referências VERIFICADAS por Crossref/arXiv), `paper/RASTREIO_NUMEROS.md` (cada número → fonte). Estudo 3 está marcado PENDENTE no rascunho; itens `% TODO` bloqueiam a submissão (autores, links de código/dados, declaração de IA, verificação de 4 referências). Código/dados da Etapa 1: `src/aos_policies.py`, `estudo_aos/run_aos_study.py`, `estudo_aos/analisar_aos.py`, `results/aos_study/full.csv`. Para retomar um runner interrompido: `python3 -u run_aos_etapa2.py --full --workers 11` no WSL (venv de `direcao1_local`; pula chaves já gravadas). O "ETA" do log é enganoso. **Cuidado ao lançar jobs no WSL:** use `nohup setsid ... < /dev/null &` seguido de `sleep` dentro do MESMO `bash -c`; o WSL ocioso sobe o pool devagar (I/O em /mnt/c) — confira com `ps`, não conclua que "não iniciou" nos primeiros 2 min, e NUNCA rode um segundo runner no mesmo CSV.
- **PONTO DE RETOMADA ATUAL (Entrada 122, 30/10/2026) — LEIA ISTO PRIMEIRO.** **Estudo 16 (ALNS no CVRPTW, regime estocástico) CONCLUÍDO, integrado e auditado** (pré-registro Entrada 121, commit `390a2de`; dados `6d960ac`; resultados e auditoria Entrada 122): 6 de 7 hipóteses confirmatórias (esperado 4,6); o padrão central replica (pesos adaptativos vencem U por 0,17pp, HG-UCB por 0,43; a mistura estática ajustada faz melhor, −0,52pp; **nenhuma política supera a mistura, 0 de 7**), com **uma exceção documentada** (H83 refutada: HG-UCB − reparo fixo −0,39pp, p aj. 0,019; o reparo fixo, o barato em ordem aleatória, não é melhor que U). O ALNS-U vence as referências VND por 4,5-5,3pp. `checar_afirmacoes.py` U8 agora lista a exceção e falha se o conjunto mudar. Código: `src/tw_alns.py` (validado em `tests/validate_tw_alns.py`: 9.818 verificações, 9 mutações detectadas), `estudo_aos/run_aos_etapa16.py`, `analisar_etapa16.py`, `fila_16.sh`. Manuscrito: **texto principal 34 páginas com cerca de UMA linha de folga na última (recompile e confira o número de páginas depois de QUALQUER edição)**; a tabela-resumo dos estudos saiu do texto principal (fica no suplemento, Tabela S27) e a lista de commits da limitação de pré-especificação é uma referência ao suplemento; suplemento 46 páginas. Verificação independente OK (738 contrastes). Nota estimada como árbitro: ~81-82. Continuam do pesquisador: edição humana da prosa, código/resposta dos autores originais, repositório anonimizado com DOI, licença, autoria/CRediT/financiamento, venue, remoto git.
- **(Anterior) PONTO DE RETOMADA (Entrada 120, 30/10/2026).** Ciclo de consolidação sem novos dados (pedido: "consolidar ainda melhor esse trabalho"): conclusões reescritas para os 15 estudos, referências e descrição dos Estudos 13-15 em 4.2 e trabalhos relacionados, limitação do Estudo 15, contextuais movidas para o suplemento, cortes de redundância. **Texto principal com 34 páginas** (o limite do EJOR é 35; suplemento 42; 0 "??"). Causa do estouro: a figura de síntese `[!htbp]` formava uma página própria de flutuantes (34% em branco); agora `[!t]`. Verificações OK (685 contrastes, macros, U1-U7). Continuam do pesquisador: edição humana da prosa, código/resposta dos autores originais, repositório anonimizado com DOI, licença, autoria/CRediT/financiamento, venue, remoto git. Conferir o número de páginas depois de QUALQUER edição do corpo.
- **(Anterior) PONTO DE RETOMADA (Entrada 119, 29/10/2026).** **Pedido do pesquisador: após uma revisão cega, nota >= 85-90/100.** Feito neste ciclo: (i) manuscrito REESTRUTURADO de 60 para 34 páginas (limite do EJOR: 35): texto por estudo movido para as Seções 8-9 do suplemento (`sec_design_detail.tex`, `sec_results_detail.tex`; referências cruzadas por `suptab.tex`: `\stab`, `\mref`, `\ssec`), cinco RQs, `fig_sintese.pdf`, tabela de orientação prática; (ii) **Estudo 15** (teste prospectivo da regra do peso morto do pool, pré-registro Entrada 118, commit `7a6c7f6`): 2 de 5 hipóteses; a regra de limiar acertou 5 de 8 células (p 0,36), mas **nenhuma política venceu a descida de ordem fixa do mesmo pool em 48 contrastes** (U7); a regularidade do pool ficou como tendência; (iii) prontidão para revisão cega (sem identificação; README com caminho rápido, `requirements.txt`, `CITATION.cff`). Nota estimada por mim como árbitro: ~81 (rigor 92; clareza 80; originalidade 72). Falta para 85-90, no que é do pesquisador: edição humana da prosa, código/resposta dos autores originais, repositório anonimizado com DOI, licença, autoria; e, se houver tempo, ALNS sob janelas de tempo. Comandos: `python estudo_aos/run_aos_etapa15.py --tuning-pools|--predizer|--eval-pools` (WSL), `analisar_etapa15.py`; verificação independente: 685 contrastes OK. Incidente: o Gemini devolveu um parecer genérico/elogioso (sem notas) e foi descartado.
- **(Anterior) PONTO DE RETOMADA (Entrada 117, 29/10/2026).** **Estudo 14 (CVRPTW, regime VND) CONCLUÍDO, integrado e auditado** (pré-registro Entrada 116, commit `d66a191`; resultados e auditoria Entrada 117): 5 de 8 hipóteses confirmadas (esperado 4,95). Cenário mais favorável à adaptação: 7 de 14 contrastes política − U melhores (alocação APRENDIDA: teste pós-hoc com pool permutado), mas nenhuma política é significativamente melhor que a melhor alternativa não adaptativa (ordem fixa de custo crescente, 0/6; pool menor P5 também melhor que P9). O que sobrevive nos três problemas: "nenhuma política supera a melhor alternativa não adaptativa" (U5 TSP, U6 CVRPTW em `checar_afirmacoes.py`). Código: `src/tw_inst.py`, `tw_neighborhoods.py` (numba; instalado no venv do WSL), `solver_pyvrp_tw.py`; runner `estudo_aos/run_aos_etapa14.py`, análise `analisar_etapa14.py`, fila `fila_14.sh`; instâncias `instances/SOLOMON` (56 + BKS, baixadas da CVRPLIB com autorização) e `SYN_TUNE_W`. Manuscrito: 60 páginas (suplemento 19); `verificacao_independente.py` OK (589 contrastes). **Pendências do pesquisador:** corte de páginas; ALNS sob janelas de tempo ou outro problema (opcional); demais decisões abaixo. Escopo do Estudo 14 (só VND) foi decisão de projeto declarada como limitação.
- **(Anterior) PONTO DE RETOMADA (Entrada 115, 28/10/2026).** **Estudo 13 (generalidade: TSP) CONCLUÍDO, integrado e auditado** (pré-registro Entrada 113, commit `27792ca`; resultados Entrada 114; auditoria Entrada 115). 4 de 12 hipóteses confirmadas (esperado 7,75). O padrão central replica (nenhuma política supera uma alternativa não adaptativa igualmente simples: pool menor P5, VND de ordem fixa crescente, mistura estática, reparo único), mas a afirmação universal "nenhuma política supera a uniforme" NÃO: no VND do TSP (pool P9, razão de custo ~60, E1) 6 de 21 contrastes são melhores (eficiência-UCB, UCB1-c, HG-UCB-c), todos dominados pelo VND de ordem fixa (pós-hoc). Teste causal pós-hoc (pool permutado): a eficiência-UCB aprende de verdade; o ganho de UCB1-c/HG-UCB-c é em parte ordem de inicialização. Com UMA rota o regret-2 coincide com o guloso (`tests/test_tsp_regret_equiv.py`). Manuscrito: 54 páginas (suplemento 18); `verificacao_independente.py` OK (506 contrastes), `checar_afirmacoes.py` OK (U1/U2 agora valem só para o CVRP, U1t lista as exceções do TSP, U5 confirma a dominância da ordem fixa), `checar_macros.py` OK (resumo 248 palavras). Código TSP: `src/tsp_inst.py`, `tsp_neighborhoods.py`, `tsp_alns.py`, `solver_lkh_tsp.py`; runner `estudo_aos/run_aos_etapa13.py` (modos incl. `--explor-ordem`, `--explor-perm`), análise `analisar_etapa13.py`, fila `fila_13.sh`. **Pendências do pesquisador (novas):** decidir se faz o Estudo 14 (CVRPLIB com janelas de tempo, o teste forte de generalidade; NÃO iniciado) e o corte de páginas; as demais decisões abaixo continuam abertas. Lançar jobs no WSL a partir do Bash tool (Git Bash) converte `/mnt/c` em caminho do Windows: use PowerShell `wsl -e bash -lc "cd /mnt/c/...; (nohup setsid ... &)"`. Edições em Python via heredoc do Bash tool perdem barras invertidas (`\n`, `\\num`): use um arquivo .py escrito com a ferramenta Write.
- **(Anterior) PONTO DE RETOMADA (Entrada 112, 28/10/2026).** Em 11/10/2026 o pesquisador pivotou: "conseguir soluções melhores a qualquer custo; resolver as fraquezas; sem restringir páginas (ele corta para ~30 depois)"; alvo EJOR se os resultados melhorarem, senão outro periódico/evento de nível. **Estudos 8-12 CONCLUÍDOS e integrados** (pré-registros: Entradas 101, 103-104, 106, 109; resultados: 102, 105, 108/110, 111, 112): controle positivo (Estudo 8: o protocolo detecta perdas plantadas), ALNS com operadores estocásticos (Estudo 9: pesos adaptativos RP −0,36pp contra U, mas Estudo 12: mistura estática e reparo fixo fazem igual ou melhor, i.e. "a adaptação substitui o ajuste, não o supera"), instâncias grandes (Estudo 10: efeito pós-hoc de n não se replica) e SWAP* eficiente (Estudo 11: as conclusões centrais sobrevivem; efeito do conjunto −1,03/−0,93pp). Manuscrito: `paper/ejor/` com 46 páginas (suplemento 16), `verificacao_independente.py` OK (387 contrastes), `checar_afirmacoes.py` OK (94 contrastes política − baseline: 0 melhores, 23 piores). Nota dada ao pesquisador em 27/10: 74/100. **Falta:** ciclo final de auditoria de revisor sobre os doze estudos (regra master) e as decisões do pesquisador (remoto git para o push — NÃO existe remote e `gh` não está instalado; escrever aos autores originais; depósito Zenodo/OSF; autoria/CRediT/financiamento/DOI/licença; corte de páginas; edição humana do texto; estratégia de venue). Opcional sugerido: segundo problema/solver para a generalidade. **Ordem de regeneração:** `estudo_aos/analisar_etapa{...}.py` → `analise_robustez.py` → `paper/ejor/gerar_tabelas.py` → `verificacao_independente.py` → `gerar_tabelas.py` → `gerar_figuras.py` → `gerar_suplemento.py` → `checar_macros.py` → `checar_afirmacoes.py` → `bash compilar.sh` (no WSL). **Fila de execuções:** `fila_12_11.sh` (resumível; `nohup setsid`; confira com `ps` e MATE POR PID — `pkill -f` dentro de `wsl bash -lc` NÃO mata os python; nunca dois runners no mesmo CSV; o WSL reinicia quando a máquina dorme). Armadilhas de edição no Windows: ver a memória `feedback-bash-tool-backslashes.md` (o Bash tool reduz barras invertidas pela metade; `read_text/write_text` do Python troca LF/CRLF — o diário é LF, a maioria dos .tex/.py de `paper/ejor/` é CRLF). Nunca tocar `../_arquivo_edp_2026-08-29/`. Commits com o trailer `Co-Authored-By: Claude Sonnet 5.5`.
- **PONTO DE RETOMADA (Entrada 99, 10/10/2026, anterior; mantido por histórico): o alvo é o EJOR (CEC em segundo); autoria/declarações com texto genérico (`\authorpend{}`, o pesquisador edita).** Estudos 1-4 concluídos e re-analisados após a correção F1 (Entradas 90, 92); manuscrito em `paper/ejor/` (elsarticle, 11 pt, espaçamento 1,5, ≤ 35 páginas por exigência do EJOR) com todos os números gerados por script (`gerar_tabelas.py` → macros `\num{}`; `checar_macros.py`; `verificacao_independente.py` v3) e suplemento online (`supplement.tex`). **Estudos 5 e 6 CONCLUÍDOS e analisados** (Entradas 94-95; Estudo 5 refutou H16-H18; Estudo 6: nenhuma política supera a uniforme, a de eficiência é a pior); diagnóstico de complementaridade e auditoria do ciclo 3 na Entrada 96; **Estudo 7 (movimento adaptativo dentro do HGS-CVRP, pré-registro Entrada 97, commit `8a06b17`) CONCLUÍDO: 0/20 significativos, efeito ≤ 0,05pp (Entrada 98).** **Auditoria de revisor do ciclo 4 (Entrada 99): 12 defeitos corrigidos (RQ6 duplicada; "todas as execuções viáveis" falso para o Estudo 7; citação não resolvida no suplemento; margem delta=0,10 "menor que todo efeito" falsa; afirmação universal agora com o Estudo 7: 66 contrastes), Estudo 7 declarado como teste fraco.** O manuscrito está completo (34 páginas, mas o total é sensível a poucas linhas: uma linha a mais no texto ou na bibliografia já levou a 35 (o limite do EJOR) e cortar texto do corpo nem sempre encolhe a última página — recompile e confira o número de páginas depois de QUALQUER edição, e corte também a bibliografia se preciso; suplemento S1-S17, 11 páginas, 0 `\pend{}`); falta só o que é do pesquisador: autoria/CRediT/financiamento/DOI/licença (`\authorpend{}`) e a decisão EJOR/CEC. Para regenerar tudo: ver a ordem em `paper/RASTREIO_NUMEROS.md`; compilar com `bash paper/ejor/compilar.sh` no WSL. Requisitos oficiais do EJOR e restrições de formato: Entrada 92. **Leia as Entradas 88-93 antes de qualquer coisa.**
- **Pendência anterior (agora parcialmente resolvida, ver acima):** rodar o diagnóstico do achado #39 (fine-tuning do Kool et al. parece degradar o modelo, pior conforme `graph_size` cresce). Comando e critério de interpretação completos na Seção 9.
- **Direção 1** (comparação justa com OR-Tools/LKH3/Kool et al.) está parcialmente executada: OR-Tools e LKH3 já rodaram nos 74 instâncias (achados #36/#37); fine-tuning do Kool está feito para os 4 tamanhos mas com resultado suspeito (achado #39, não resolvido); avaliação final (`avaliar_kool_finetunado.py`) ainda não deve rodar até essa pendência ser resolvida.
- **Direção 2** (ablação dos 3 mecanismos de aleatoriedade) está **concluída**: nenhum dos 3 mecanismos é individualmente significativo na config de produção (α=0.5) em n=74; só o drift-reset em α=5.0 (não-produção) é robusto (p<0.0001). Ver achados #17-#25.
- **Tarefa #178** (calibração de alpha/max_inner_iter) está **concluída**: default de produção mantido (α=0.5, max_inner_iter=40), só um bug de consistência de código corrigido. Ver Seção 5 e achados #15/#16.
- Este é o primeiro arquivo CLAUDE.md do projeto — não existe um anterior para comparar ou migrar.

---

## 1. Identidade do projeto

**Objetivo:** reproduzir fielmente o algoritmo HG-UCB (bandit multi-armado — UCB1 + gradiente de preferência softmax híbrido, com cold-start, ε-greedy e reset por detecção de drift via ADWIN) guiando uma Variable Neighborhood Search (VNS) para o CVRP (Capacitated Vehicle Routing Problem), validar contra o paper original, e construir uma contribuição própria em cima disso para submissão ao **IEEE CEC 2027** (Edimburgo, Escócia, 25-28/jul/2027). Prazos: paper completo **28/fev/2027**, paper curto **15/mar/2027**, notificação **07/abr/2027**.

**Paper de referência:** Kalatzantonakis, Sifaleras & Samaras (2023), "A reinforcement learning-Variable Neighborhood Search method for the capacitated vehicle routing problem", *Expert Systems with Applications*, DOI `10.1016/j.eswa.2022.118812`. PDF em `../literatura/` (obtido legitimamente via autoarquivamento institucional do autor correspondente — green open access; preprint de 09/2022 confirmado tecnicamente idêntico à versão publicada). Equações citadas neste arquivo (Eq. 5, 10-14) referem-se a esse paper.

**As 3 direções de contribuição** (decisão: todas as três, não uma só — ver Apêndice A, decisão #2):
1. **Reprodução fiel** do HG-UCB + validação contra as Tabelas 7-13 do paper nas 74 instâncias do CVRPLIB.
2. **Direção 1 — comparação justa**: HG-UCB vs. OR-Tools (CP-SAT/Guided Local Search), LKH3, e Kool et al. (2019) fine-tunado, todos com orçamento de tempo casado ao tempo de parede do próprio HG-UCB por instância.
3. **Direção 2 — ablação**: isolar a contribuição individual de cold-start, ε-greedy e drift-reset.

**Benchmark:** 74 instâncias CVRPLIB — conjuntos A (26, Augerat et al.), B (23, Augerat et al.), X (21, Uchoa et al. 2017), CMT (4, Christofides-Mingozzi-Toth). Baixadas de `https://galgos.inf.puc-rio.br/cvrplib/` (domínio novo; `vrp.atd-lab.inf.puc-rio.br` redireciona para lá).

**Baseline próprio validado:** GVNS clássico (sem bandit) bate 3.35% de gap vs. ótimo conhecido nas instâncias A/B/X, contra 3.39% do paper — fundação considerada sólida (Entrada 1-3).

---

## 2. Limites de escopo CRÍTICOS

- **NUNCA toque em `../_arquivo_edp_2026-08-29/`** — projeto anterior (EDP), arquivado, completamente não relacionado. Não ler, não editar, não listar além do necessário para confirmar que é outra coisa.
- **Todo trabalho deste projeto fica confinado a `cvrp_bandit/`** (este diretório) **mais um pequeno conjunto de arquivos COMPARTILHADOS que vivem FORA dele**, na raiz `C:\Pesquisas\learnheuristica\` (ou seja, em `../` a partir daqui):
  - `../diario_pesquisa.md` — o diário de pesquisa cronológico (ver Seção 4.1).
  - `../GUIA_PESQUISA_GERAL.md` e `../LEI_GERAL_PESQUISA.md` — disciplina de pesquisa geral, válida para todos os projetos do workspace, não específica ao CVRP (ver Seção 4.2).
  - `../literatura/` — PDFs de fonte primária (`Kalatzantonakis_Sifaleras_Samaras_2023_HGUCB.pdf`).
  - `../Estilo do CEC/` — 17 papers de exemplo do IEEE CEC, usados como referência de estilo de escrita.
  - **Isto é importante e fácil de errar:** o diário e os dois guias gerais NÃO estão dentro de `cvrp_bandit/`. Um erro já cometido numa sessão anterior (e autocorrigido antes de vazar para qualquer documento) foi assumir que estavam. Sempre confirme o caminho relativo (`../diario_pesquisa.md`, não `./diario_pesquisa.md`).
- **Nunca sobrescreva um resultado já reportado silenciosamente.** Resultados substituídos vão para `results/superseded/` com sufixo descritivo, ou recebem backup com timestamp antes de serem sobrescritos (ex.: `lkh3_74_pre_retry_*.csv`).
- **Dependências de terceiros não são sincronizadas como artefato do projeto**: LKH3 (fonte + binário compilado), o repositório do Kool et al., os scripts oficiais do gerador Uchoa, o dataset XML100. Ficam documentados por URL em `direcao1_local/README.md`/`setup.sh` e são reprodutíveis a partir daí — não precisam (e não devem) ir para controle de versão ou backup deste projeto.
- CSVs de resultado gerados na máquina do pesquisador **não são automaticamente puxados para análise** — ele decide quando compartilhar ou quando pedir a análise final.

---

## 3. Mapa de diretórios

```
C:\Pesquisas\learnheuristica\                    (raiz compartilhada — múltiplos projetos)
├── cvrp_bandit\                                  ← ESTE projeto (working directory do Claude Code)
│   ├── src\                  implementação principal:
│   │                           bandit.py          — HGUCBBandit (UCB1 + softmax de preferência + ADWIN)
│   │                           construction.py    — heurística construtiva inicial (+ repair_route_count)
│   │                           instance.py        — parsing .vrp/.sol, convenção K-exato/K-mínimo
│   │                           neighborhoods.py   — operadores de vizinhança do VNS
│   │                           solver_lkh3.py     — wrapper do LKH3
│   │                           solver_ortools.py  — wrapper do OR-Tools (CP-SAT/GLS)
│   │                           vns.py             — bandit_vns(), loop principal, todos os _fn/_mode hooks
│   ├── tests\                 34 arquivos: diag1-18_*.py (diagnósticos ad-hoc da investigação),
│   │                           validate_*.py ×5, compare_*.py ×2, ablation_*.py ×2 (Direção 2),
│   │                           run_*.py ×2, analise_*.py ×2 (estatística), bruteforce_crosscheck.py,
│   │                           smoke_test_bandit_vns.py
│   ├── direcao1_local\        execução pesada da Direção 1 — RODA NA MÁQUINA REAL, não em sandbox
│   │                           (pivô decidido 07/09/2026 — ver Seção 4.4). Contém: kool_finetune_runner.py,
│   │                           kool_uchoa_monkeypatch.py, avaliar_kool_finetunado.py, medir_tempo_hgucb.py,
│   │                           rodar_ortools_74.py, rodar_lkh3_74.py, retry_lkh3_piso_maior.py,
│   │                           rodar_finetuning_overnight.sh, sondar_custo_kool.py, uchoa_gerador_core.py,
│   │                           validar_uchoa_gerador.py, setup.sh, README.md, 5× log_finetune_*.txt, venv\
│   ├── instances\             A\, B\, CMT\, X\ — 74 pares .vrp+.sol
│   ├── external\              dependências de terceiros, NÃO sincronizadas (ver Seção 2):
│   │                           attention-learn-to-route\ (Kool et al., MIT), LKH-3.0.7\, uchoa_generator\
│   └── results\               CSVs/logs de todas as validações + kool_finetuned\{cvrp_10,20,50,100}\
│                               (cada um com 1 pasta uchoa_ft_*_20260915T..., 15 checkpoints) + superseded\
├── literatura\                 Kalatzantonakis_Sifaleras_Samaras_2023_HGUCB.pdf
├── Estilo do CEC\               17 papers de exemplo IEEE CEC
├── diario_pesquisa.md           ← O DIÁRIO (fora de cvrp_bandit/!) — continue daqui, Entrada 60+
├── GUIA_PESQUISA_GERAL.md       ← disciplina geral (fora de cvrp_bandit/)
├── LEI_GERAL_PESQUISA.md        ← disciplina geral (fora de cvrp_bandit/)
├── resumo_estado_pesquisa_tarefa178.md   snapshot consolidado da Tarefa #178 (fora de cvrp_bandit/)
├── _arquivo_edp_2026-08-29\     ARQUIVADO, projeto não relacionado — NUNCA tocar
└── _sync_tmp\, _to_delete\      scratch de sincronização (resquício do workflow via Cowork — ver 4.3)
```

---

## 4. Disciplina operacional para sessões Claude Code

### 4.1 O diário de pesquisa

`../diario_pesquisa.md` é o registro cronológico central do projeto — 59 entradas até agora, template fixo:

```
## N. Título (DD/MM/AAAA)
[Achado real #M — quando aplicável, nem toda entrada tem um]
[corpo: o que foi feito/encontrado]
**Decisão de processo:** [parágrafo]
**Arquivos desta entrada:** [lista]
**Impacto:** [parágrafo de fechamento]
```

**Toda sessão de Claude Code deve continuar esse diário, não começar um novo.** Próxima entrada = **60**; próximo número de achado real, se houver um novo = **#40**. Documente cronologicamente: toda execução real (não leitura de código), toda decisão de processo, toda mudança de arquivo, todo bug encontrado/corrigido — exatamente como já vinha sendo feito.

### 4.2 Os dois guias gerais

`../GUIA_PESQUISA_GERAL.md` (estrutura: anatomia de paper, padrões de rigor empírico, padrões de escrita, protocolo de arquivamento em 3 níveis, checklist de submissão, protocolo operacional de 13 pontos para agentes de IA) e `../LEI_GERAL_PESQUISA.md` (7 princípios não-negociáveis + 12 estudos de caso numerados — config drift, validar por execução não leitura, citações não verificadas, amostras pequenas mentem, resultados nulos são dados, delegação com ceticismo, disciplina de escopo de refatoração, portões de aprovação, engenharia de determinismo, higiene de paralelismo, venue-alvo é factual, isolar a variável atribuída) **se aplicam a este projeto sem modificação** — são documentos de disciplina cross-projeto, não específicos ao CVRP, e continuam valendo desde o projeto EDP arquivado. Releia-os diretamente (`Read ../GUIA_PESQUISA_GERAL.md`, `Read ../LEI_GERAL_PESQUISA.md`) sempre que uma decisão parecer cair numa zona cinzenta — este CLAUDE.md não os substitui nem os resume por completo.

### 4.3 O workflow de sincronização está OBSOLETO

Até 04/10/2026, este projeto era operado via Claude (Cowork) rodando num sandbox de nuvem **sem acesso direto ao disco do pesquisador**. Todo arquivo criado ou lido exigia uma dança: `SendUserFile` → `file_uuid` → `device_commit_files` → verificação por tamanho/timestamp via `device_list_dir` (verificação por SHA-256 estava quebrada desde ~08/09/2026, atribuída a uma atualização do Windows que bloqueou o ponto de montagem `~/mnt/...`).

**Isso não existe mais a partir de uma sessão Claude Code rodando na própria máquina do pesquisador.** Use `Read`/`Write`/`Edit`/`Bash` diretamente nos caminhos reais (`C:\Pesquisas\learnheuristica\cvrp_bandit\...` ou equivalente WSL). Não recrie a dança de sincronização por hábito — ela era uma solução de contorno para uma limitação de capacidade que não existe mais aqui. As pastas `_sync_tmp\` e `_to_delete\` são resquício desse workflow antigo; podem ser ignoradas ou limpas quando convier.

### 4.4 Autonomia de execução vs. gate de metodologia — dois eixos distintos

Em 07/09/2026 (Entrada 41) foi decidido que toda execução pesada da Direção 1 (fine-tuning do Kool, OR-Tools/LKH3 nas 74 instâncias, medição de tempo do HG-UCB) rodaria na máquina real do pesquisador, não no sandbox de nuvem — nessa época por limitação de capacidade (sem GPU/WSL no sandbox). Ao migrar para Claude Code rodando direto na máquina real, essa limitação desaparece, o que levanta uma pergunta nova: quem decide quando/como rodar os comandos?

Perguntado diretamente ao pesquisador (04/10/2026), a resposta escolhida foi **"Autonomia total, reporta depois"** — explicitamente NÃO a opção recomendada ("Executa com aviso prévio"). Isso precisa ser aplicado com cuidado, porque na verdade existem **dois eixos independentes**, e a resposta do pesquisador só resolve o primeiro:

**EIXO A — "quem aperta o play" (execução/orquestração).** ✅ Autonomia total concedida. Claude Code pode, sem pedir permissão prévia: rodar scripts e testes, instalar dependências, iniciar treinos/validações, rodar comandos de diagnóstico, mover/organizar arquivos dentro do escopo da Seção 2, startar processos longos (overnight, etc.). Reporte o que foi feito depois — no diário (sempre) e numa mensagem de resumo ao pesquisador (quando ele estiver presente).

**EIXO B — "o que conta como uma decisão metodológica válida" (interpretação/escopo do resultado). ⚠️ Continua exigindo apresentar a decisão ao pesquisador ANTES de se comprometer, e esperar a resposta dele** — exatamente a mesma disciplina que antes passava pelo `AskUserQuestion` no Cowork (o Claude Code pode não ter uma ferramenta com esse nome exato; o que importa é o *comportamento*: perguntar e esperar, não assumir). Isso inclui, por exemplo:
- mudar qualquer hiperparâmetro de produção (alpha, max_inner_iter, epsilon, C, parâmetros do ADWIN, etc.) para um valor diferente do já decidido na Seção 5;
- mudar o escopo do benchmark — quais instâncias, quais solvers, quantos seeds contam para uma comparação que vai para o paper;
- decidir se um achado "fecha" uma linha de investigação ou exige mais uma rodada (padrão recorrente neste projeto — ver Seção 4.6);
- decidir como interpretar ou reportar um resultado nulo/negativo;
- qualquer mudança que afete como um resultado será lido no paper final.

Se a sessão estiver rodando sem o pesquisador presente (ex.: ele disparou algo e saiu) e uma decisão do Eixo B surgir, não adivinhe: documente a decisão pendente claramente (diário + resumo) e pare **esse ramo específico** — mas pode continuar em paralelo qualquer trabalho que seja puramente Eixo A.

### 4.5 Uso do Gemini como segunda opinião

O projeto usa o Gemini (`gemini-2.5-flash`, via MCP) como segunda opinião independente em leituras ambíguas do pseudocódigo do paper. Padrão sempre seguido: apresentar os fatos ao Gemini **de forma neutra**, sem revelar a conclusão prévia do Claude, para evitar viés de confirmação; toda sugestão do Gemini é testada contra a fonte primária (o PDF) ou empiricamente — nunca aceita por autoridade. Já aconteceu de uma hipótese do Gemini ser refutada ao reconferir o PDF (Entrada 18) e de outra ser confirmada (Entrada 20). Verifique no início da sessão quais ferramentas MCP de Gemini estão disponíveis (os nomes podem diferir dos usados no Cowork).

### 4.6 Regras de rigor estatístico

- Unidade de pareamento: **instância** (média de 5 seeds por instância) — prática padrão para comparação de metaheurísticas.
- **Sempre dois testes juntos**: Wilcoxon signed-rank (primário, não-paramétrico) + teste t pareado (cruzamento). Quando divergem, **Wilcoxon é tratado como autoridade** (foi o que decidiu a Entrada 26: manter o default apesar do teste t "quase" significativo).
- Limiar de significância convencional: α=0,05.
- **Nenhum teste formal de significância foi aplicado nesta linha de pesquisa até a Entrada 25** — lacuna que o Gemini apontou duas vezes antes de ser corrigida; registrado como lição explícita sobre confiar demais em resultados "limpos" de amostra pequena.
- Aplicar um teste a dados **já coletados** (sem nova execução) não exige um novo gate de decisão — é tratado como de alto valor/baixo custo e feito diretamente.

**Padrão recorrente — "quase significativo" não é "provavelmente real":** pelo menos 3 ocorrências explícitas no diário de um sinal que parecia limpo numa amostra pequena e evaporou ao aumentar a amostra:
1. Achado #13 (Entrada 21): vantagem de α=5.0/mii=160 em 8 instâncias se **inverte** em 16, por causa de 1 instância nova (CMT1).
2. Achado #23 (Entrada 31): sinais de ε-greedy e do braço "remover os 3 mecanismos" encolhem a quase zero ao ir de n=16 para n=74.
3. Achado #29 (Entrada 38): o efeito do cold-start (o único que tinha sobrevivido n=16→n=74) encolhe de +0,107pp para +0,007pp (quase empate exato) ao triplicar seeds de 5 para 15.

Lição registrada literalmente na Entrada 38: *"'quase significativo' não deveria ser tratado como 'provavelmente real'."* Trate qualquer novo resultado "quase significativo" com a mesma suspeita — aumente a amostra antes de reportar.

### 4.7 Licenciamento e proveniência de dados

- **Kool et al.** (`wouterkool/attention-learn-to-route`): MIT. Checkpoints pré-treinados (10/20/50/100) publicados pelos próprios autores.
- **Nazari et al.** (`OptMLGroup/VRP-RL`): sem arquivo LICENSE, treino do zero proibitivo sem GPU/tempo — **excluído** da comparação (Entradas 33/34).
- **LKH3**: `http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.7.tgz`, licença "uso em pesquisa, autor reserva todos os direitos" — compatível com reprodução acadêmica. Binário/fonte deliberadamente não sincronizados (reprodutível a partir da URL).
- **Gerador X do Uchoa**: script oficial hospedado no próprio CVRPLIB — mesma fonte dos instâncias X reais; usado em vez de uma reconstrução a partir da prosa do paper (que diverge do código real na distribuição de nº de clusters — código vence).
- **Gurobi**: não instalado, sem licença disponível — fora de escopo a menos que o pesquisador diga o contrário.
- Nunca cite a Tabela 13 do paper original (tempos em hardware/linguagem diferentes, i9-7940X/28 threads/Windows 10/R 4.2.1, ela mesma emprestada de Ardon 2022 arXiv:2201.05393) como base de comparação de superioridade — é contexto, não baseline.

---

## 5. Referência técnica — hiperparâmetros

| Parâmetro | Valor do paper | Default atual no código | Local | Histórico resumido |
|---|---|---|---|---|
| **alpha** (Eq. 11, taxa de aprendizado do gradiente-bandit) | 5,0 (Tabela 7/8, "melhor") | **0,5** | `HGUCBBandit.__init__()`, `src/bandit.py` | Literal 5,0 causa colapso do softmax(H) (achado #4). Sweep mostra α pequeno evita colapso (achado #5). Normalização da recompensa (achado #6) resolve o colapso mantendo α literal possível. Fatorial completo em n=74 (achado #15) + teste formal (achado #16, p<0,004) confirmam α=0,5 robustamente melhor. Fixado como default do construtor só na Entrada 27 (antes disso o construtor dizia 5,0 por inconsistência — bug corrigido, ver Seção 8). |
| **max_inner_iter** (cap de segurança do loop interno — NÃO é parâmetro do paper) | — (condição de parada do paper é inatingível em k=5) | **40** | `bandit_vns()`, `src/vns.py` (`8 * len(neighborhoods)`) | Introduzido na Entrada 5. Loop sempre atinge o cap (achados #11/#12). Subir para 160 ajuda um pouco em ambos os α na base completa (achado #15), mas efeito estatisticamente frágil (Wilcoxon p=0,188 — achado #16) contra 4x de custo. Mantido em 40 (Entrada 26). |
| **b_t** (limiar de filtro do pool de operadores não-bloqueados) | 0,2 | **0,2 — nunca mudou** | — | Provado matematicamente inatingível como condição de *parada* para k=5 (softmax de 5 valores nunca <0,2); sweep 0,05-0,30 é achatado, descartando "b_t é erro de digitação". Continua funcionando como filtro de pool. |
| **epsilon** (ε-greedy, Eq. 14) | 0,1 | **0,1 — nunca mudou** | — | Ablação (Direção 2): removê-lo *melhora* o gap em n=16 (achado #17), mas encolhe a quase zero em n=74 (achado #23) — ruído estatístico, não efeito real. |
| **C** (fator de escala do UCB1, Eq. 5) | 2,0 | **2,0 — nunca mudou/questionado** | — | — |
| **ADWIN** (W=5, delta=0,02, via `river`) | — | valores do paper, nunca variados diretamente | `src/bandit.py` (~linha 208) | Dois toggles próprios adicionados e testados (ver abaixo). |
| **adwin_feed_mode** | "q_value" (leitura literal do §5.4) | **"reward"** | hook em `src/bandit.py` | Achado #8 (Entrada 14): código alimentava ADWIN com a recompensa bruta, não com softmax(H) como o paper e a documentação própria (errada) afirmavam. Testado o modo "q_value" fiel ao paper (Entrada 15): reduz taxa de drift como esperado, mas **piora** o gap em ambos os α — default ficou em "reward" (menos fiel ao paper, empiricamente melhor). |
| **drift_reset_enabled** | True (implícito) | **True** | hook em `src/bandit.py`/`vns.py` | Ablação: negligível em α=0,5 (produção; ADWIN raramente dispara); grande e robusto em α=5,0 (achados #20/#25, p<0,0001, não-produção). Transição entre os dois é não-monotônica, com um "vale" onde chega a prejudicar em α≈3-4 (achado #31) — não investigado além disso. |
| **reward_positive** (Eq. 12) | `rs=\|sol_g−sol_c\|`, absoluto, sem normalização documentada (confirmado 3x contra o PDF) | **`reward_positive_normalized`** (divide por custo da solução atual) | hook `reward_positive_fn` em `src/vns.py`, adicionado na Entrada 12 | Adição própria do projeto (achado #6) — resolveu o colapso do softmax mantendo α alto literal viável. Criou uma assimetria nova com a recompensa negativa (achado #9); tentativa de reescalar para corrigir foi testada e **rejeitada** (achado #10 — reabre o colapso). |
| **reward_negative** (Eq. 13) | dimensional conforme publicado | **nunca modificada** | — | Confirmado 3x contra o PDF que não tem termo de custo; hipótese do Gemini de que precisaria do mesmo tratamento foi checada e refutada (Entrada 18). |
| **p_hat_mode** (ambiguidade do termo de exploitation do UCB1, Eq. 5) | duas leituras possíveis | **"q_values"** (=softmax(H)) | — | "sliding_window" (janela W=5 de recompensas brutas) testado cabeça-a-cabeça: empate exato em 2 decimais (Entrada 6). Mantido "q_values" por simplicidade; a outra opção fica implementada mas não usada. |
| **progressive_x_linha** (x' avança ou fica fixo durante o loop interno — escolha de interpretação, não hiperparâmetro nomeado no paper) | ambíguo no pseudocódigo | **True** (x' avança a cada melhoria aceita) | — | Leitura fixa (`False`) causou a maior regressão do projeto (achado #2, gap de 5,76% vs 1,61% do paper). Trocado para progressivo (achado #3 em diante). Modo fixo mantido no código só para comparação. |

**Trajetória do gap vs. GVNS clássico (nosso próprio baseline, não o ótimo):** x' fixo +1,89pp → x' progressivo +0,93pp (ainda invertendo o sinal esperado do paper) → α=5,0 normalizado +0,16pp → α=0,5 normalizado +0,06pp → **em n=74 completo, o HG-UCB final (α=0,5/mii=40) na verdade fica -0,03pp, ou seja, ligeiramente MELHOR que o GVNS clássico** (achado #22) — pequeno e frágil, mas na direção certa. Melhor config isolada encontrada: α=0,5/mii=160, 3,315% de gap vs. ótimo (não adotada como default por causa do custo 4x e significância frágil do efeito do mii).

---

## 6. Referência técnica — CVRPLIB e peculiaridades dos solvers

1. **Convenção K-exato vs. K-mínimo**: conjuntos A/B (e E/M) exigem **exatamente** K veículos (igualdade); X (Uchoa et al. 2017) e CMT exigem **no mínimo** K. CMT já foi erroneamente assumido como K-exato numa sessão anterior — Uchoa et al. 2017 é a fonte primária definitiva (conferida via WebFetch) de que CMT, ao ser republicado, abandonou tanto o arredondamento quanto a fixação de K. Auto-detectado por nome de instância (`_vehicles_exact_default`) — nunca hardcode isso de novo por script.
2. **Convenção de arredondamento de distância**: A/B/X usam `nint()`; CMT usa float sem arredondamento — auto-detectado comparando custo recalculado vs. declarado.
3. **Offset de id CVRPLIB**: `.sol` numera clientes excluindo o depósito; `.vrp` numera todos os nós incluindo o depósito — qualquer leitor novo precisa reaplicar o offset +1 pós-depósito (`_sol_routes_to_vrp_ids`), nunca copiar ids diretamente.
4. **OR-Tools não aplica K-exato por padrão** — precisa de restrições explícitas `Start≠End` por veículo (K-exato) ou um teto de slack (+5) com restrição de soma (K-mínimo).
5. **Gotchas da API do OR-Tools**: `IsEqualVar` compara var-com-var, não var-com-constante (usar `IsEqualCstVar`); o resultado 0/1 dele não tem `.Not()` — negue via uma restrição de soma equivalente. Internamente escala distâncias ×1000 para custos inteiros — nunca reporte esse número, sempre recalcule via `inst.solution_cost(routes)`.
6. **LKH3 não tem modo "K-mínimo" nativo** — `VEHICLES`+`MTSP_MIN_SIZE=1` funciona como K-exato; conjuntos K-mínimo precisam de um sweep de K a K+5, orçamento de tempo dividido entre candidatos.
7. **LKH3 arredonda distância internamente mesmo quando a convenção real é float** (CMT) — nunca confie na linha "Cost=" impressa por ele; sempre recalcule a partir das rotas retornadas. Isso quase gerou uma falsa alegação de "melhor que o ótimo provado" para CMT1 (521 vs. 524,611 real).
8. **A restrição de nº de veículos do LKH3 (MTSP) é um termo suave/penalizado internamente, não rígido** — sob o orçamento de tempo casado ao HG-UCB, instâncias X maiores/mais difíceis podem convergir para algo que parece completo mas está penalizado/inviável. Sempre cheque `is_feasible()` explicitamente — nunca assuma que "rodou sem erro" = "viável". Causou 23,5% de taxa de "sem solução" (vs. 4,1% do OR-Tools), 7 das 8 perdas concentradas em membros X de K grande.
9. **Mais tempo não resolve uniformemente o problema de penalidade do LKH3** — e previsões feitas em sandbox sobre quais instâncias são "irrecuperáveis" não são confiáveis na borda (X-n172-k51 foi resolvida na máquina real depois de não resolver no sandbox). Sempre reverifique casos de borda na máquina alvo.
10. **O modelo do Kool et al. não consegue forçar exatamente K rotas** — incompatibilidade estrutural com a comparação estrita em A/B (não em X/CMT). Sempre reporte um critério de viabilidade estrito E um sem essa restrição; nunca descarte silenciosamente os ~49/74 instâncias A+B.
11. **Checkpoints pré-treinados do Kool usam coordenadas uniformes [0,1]², demanda discreta {1..9}, ≤100 clientes** — aplicar fora dessa distribuição ao CVRPLIB (clusterizado/excêntrico, X até ~1000 clientes, CMT até 199) é risco conhecido de degradação; mitigado via fine-tuning com o gerador Uchoa oficial (não uma aproximação por prosa).
12. **A prosa do paper do Kool e o código real do gerador divergem** na distribuição de nº de clusters (texto: Uniforme[3,8], peso exp(-d/40); código: Uniforme[2,6], peso 2^(-d/40)) — código vence, porque o objetivo é casar com o conjunto X real.
13. **O gerador Uchoa é estruturalmente inviável em ~7% dos casos especificamente em graph_size=10** (o esquema meio-clusterizado pode precisar de até 6 semente-de-cluster contra só ~5 clientes reservados) — quase certo de travar uma época de treino completa. Precisa de rejection sampling, não de um cap de retry maior; tamanhos 20/50/100 não têm esse problema (0 falhas em 20.000+ tentativas cada).
14. **`train.py` do Kool tem `num_workers=1` hardcoded**, sem flag de CLI para mudar — trava em pelo menos uma configuração real (WSL2+Python 3.14) com "too many fds". Corrigido via patch direto no arquivo vendorizado (`sed` para 0), aplicado de forma idempotente em `setup.sh` — a única vez que este projeto editou um arquivo de terceiro diretamente. Não revertam sem testar de novo numa máquina real.
15. **O baseline "rollout" do Kool (usado nos checkpoints públicos) não é leve** — faz um rollout guloso extra do tamanho de uma época inteira, mais uma reavaliação por callback de época. Uma sonda de custo por passo que não invoca isso vai subestimar o custo real se o rollout for usado depois.
16. **Checkpoints publicados pelo Kool contêm só os pesos do modelo** (sem estado de otimizador/baseline/RNG) — seguro carregar sob uma escolha de baseline diferente; o código remove defensivamente uma chave `'baseline'` obsoleta ao retomar (`--resume`).
17. **Fine-tuning com lr/clip de gradiente não ajustados do Kool parece degradar o modelo, pior conforme graph_size cresce** (achado #39, aberto) — nunca trate um checkpoint fine-tunado como melhoria sem checar o custo de validação contra a época 0 primeiro.
18. **O problema de penalidade suave do LKH3 (item 8) é a peculiaridade de comparabilidade mais consequente encontrada na Direção 1** — terceira manifestação da mesma tensão "comparadores precisam de mais tempo do que o orçamento casado dá para alcançar viabilidade sob restrição de tamanho de frota", já vista também com Kool (item 10) e OR-Tools (achado #36).

---

## 7. Log de pesquisa — achados reais #1 a #39

*(condensado; data e nº de entrada apontam para o diário completo — ver nota de transparência no topo deste arquivo sobre a origem destes números)*

- **#1** [inferido/sem tag explícita] (Entrada 5, 01/09/2026) — b_t=0,2 provado matematicamente inatingível como condição de parada para k=5; bug separado de deadlock cold-start/filtro-b_t achado por teste. Decisão: manter b_t=0,2 literal, adicionar `max_inner_iter` como mecanismo real de parada.
- **#2** (Entrada 7, 01/09/2026) — "x' fixo" tem desempenho péssimo: gap de 5,76% vs. 1,61% do paper no conjunto A (+4,15pp). Fixar x' bloqueia composição de melhorias entre operadores dentro de um passo.
- **#3** (Entrada 8, 02/09/2026) — mesmo com x' progressivo, nosso bandit fica 0,93pp **pior** que nosso próprio GVNS clássico (paper diz que deveria ser 0,29pp melhor). O achado mais consequente da investigação.
- **#4** (Entrada 9, 02/09/2026) — mecanismo de colapso do softmax(H) encontrado: com α=5,0 e recompensa de custo bruto, softmax(H) satura quase one-hot em 99,6% das iterações, neutralizando o filtro b_t e o ε-greedy. UCB1 isolado (sem softmax) já bate o GVNS clássico.
- **#5** (Entrada 10, 02/09/2026) — sweep de alpha confirma monotonicamente: colapso% e gap% caem juntos; α=0,001 bate o GVNS clássico na amostra de 5 instâncias.
- **#6** (Entrada 12, 02/09/2026) — **normalização da recompensa é a correção mais importante**: dividir a Eq. 12 pelo custo da solução atual permite que o α=5,0 *literal* evite o colapso (99,6%→12,5%). Gera os dois candidatos padrão: α=5,0 normalizado e α=0,5 normalizado.
- **#7** (Entrada 13, 02/09/2026) — ambos candidatos validados nas 74 instâncias: gap vs. GVNS clássico cai de +1,89pp (fixo) → +0,93pp (progressivo) → +0,16pp (α=5,0 norm.) → +0,06pp (α=0,5 norm.), redução >90%. Residual honesto: nenhum candidato ainda bate o GVNS clássico ou a linha do próprio VNS do paper.
- **#8** (Entrada 14, 02/09/2026) — ADWIN era alimentado com recompensa bruta, não Q-values, contradizendo o paper §5.4 e a própria documentação do projeto (que afirmava, errado, que isso já estava correto).
- *(Entrada 15, sem tag)* — testar `adwin_feed_mode="q_value"` reduz taxa de drift como previsto mas **piora** o gap em ambos os α; default "reward" mantido.
- **#9** (Entrada 16, 02/09/2026) — a normalização do #6 criou uma assimetria nova: recompensa negativa (Eq. 13, intocada) ficou ~115x maior em magnitude típica que a positiva agora normalizada.
- **#10** (Entrada 17, 02/09/2026) — reescalar a recompensa positiva para corrigir o #9 reabre o #4 (resultado negativo): piora o gap% monoticamente em todas as 8 configurações testadas. #6 e #9 estão em tensão direta; não existe correção multiplicativa.
- *(Entrada 18, sem tag)* — última hipótese do Gemini (Eq. 13 também precisaria de normalização por custo) checada contra o PDF pela 3ª vez e **refutada** — Eq. 13 nunca teve termo de custo.
- **#11** (Entrada 19, 02/09/2026) — calibração de max_inner_iter, resultado misto: loop sempre atinge o cap configurado (confirmado por contagem direta). Sem ganho para α=0,5; para α=5,0, subir o cap para 160 dá o melhor gap visto até então (3,25%, amostra pequena) a 3,6x de custo.
- **#12** (Entrada 20, 03/09/2026) — colapso% cai genuinamente com cap maior: 21,4%→1,5% (mii 20→160), monotônico nas 8 instâncias sem exceção — explica mecanisticamente o #11.
- **#13** (Entrada 21, 04/09/2026) — a vantagem do #11 se **inverte** em n=16: expandir a amostra de 8→16 inverte o resultado α=5,0/mii=160 vs. α=0,5/mii=160; dirigido quase todo por uma instância CMT1 nova.
- **#14** (Entrada 22, 04/09/2026) — resolve o #13: testar as demais CMT2/3/11 mostra α=0,5/mii=160 batendo α=5,0/mii=160 nas 4 instâncias CMT (3 vitórias, 1 empate exato) — efeito real, concentrado em CMT, não um acaso do CMT1.
- **#15** (Entrada 24, 05/09/2026) — fatorial 2×2 completo nas 74 instâncias fecha a questão: α=0,5 bate α=5,0 em ambos os níveis de mii (~-0,10pp cada); mii=160 bate mii=40 em ambos os α (~-0,11pp cada), efeitos quase independentes. Melhor das 4: α=0,5/mii=160 (3,315%). CMT isolado (5,4% do benchmark) explica mais de 100% da vantagem agregada do mii.
- **#16** (Entrada 25, 05/09/2026) — o ganho do mii é estatisticamente frágil, o do alpha é robusto: Wilcoxon p=0,188 (não significativo), teste t p=0,049 (apenas) para mii=160 vs. 40; alpha=0,5 vs. 5,0 é robusto nos dois testes (p<0,004, n=148).
- **#17-#20** (Entrada 29, 06/09/2026) — Fase 1 da ablação da Direção 2 (n=16), quatro achados numa entrada: **#17** os 3 mecanismos de aleatoriedade não são igualmente importantes; remover ε-greedy *melhora* o resultado (oposto do esperado). **#18** remover cold-start não deixa nenhum operador sem testar (o piso do termo de exploração do UCB1 já garante cobertura). **#19** remover os 3 mecanismos juntos é o pior braço, pior que sem bandit algum. **#20** em α=5,0 (onde ADWIN dispara ~16x mais), remover drift-reset custa +0,68pp — importância condicional à taxa de disparo do ADWIN, não fixa.
- **#21** (Entrada 30, 06/09/2026) — das 5 comparações de ablação, só a de drift-reset em α=5,0 sobrevive a teste formal (p=0,0007/0,0073); as 4 comparações em config de produção (α=0,5) não (n=16 pequeno demais).
- **#22-#25** (Entrada 31, 06/09/2026) — Fase 2 da ablação em n=74 completo, quatro achados: **#22** (contexto) hgucb_completo vs. gvns_classico inverte de +0,13pp (n=16) para -0,03pp (n=74) — pequeno e frágil, consistente com o #7. **#23** sinais de ε-greedy e do braço de remoção combinada encolhem a quase zero em n=74 (ruído confirmado). **#24** cold-start é o único mecanismo cujo efeito **não** encolhe (p=0,071, próximo mas não passa). **#25** drift-reset em α=5,0 fica ainda mais nítido (p<0,0001, 64 vitórias vs. 1) — o único achado robusto da Direção 2, mas em regime não-produção.
- **#26** (Entrada 35, 07/09/2026) — OR-Tools precisa de modelagem explícita de K (ver Seção 6, item 4); 2 bugs de API do OR-Tools corrigidos no processo.
- **#27** (Entrada 36, 07/09/2026) — dois fatos do LKH3 numa validação: (i) não tem modo "K-mínimo" nativo (ver Seção 6, item 6); (ii) arredonda distância internamente mesmo para CMT float, produzindo um falso "melhor que o ótimo" para CMT1 se confiado (ver Seção 6, item 7).
- **#28** (Entrada 37, 07/09/2026) — gerador oficial X do Uchoa localizado no próprio site do CVRPLIB — elimina risco de fidelidade vs. reconstruir da prosa do paper. Também achada discrepância prosa-vs-código na distribuição de nº de clusters (código vence).
- **#29** (Entrada 38, 07/09/2026) — o efeito quase-significativo do cold-start evapora em 15 seeds: triplicar seeds (5→15) encolhe a diferença de +0,107pp para +0,007pp e o placar de vitórias para um quase-empate exato (36/37) — mesma assinatura "encolhe com a amostra" do #13/#23.
- **#30** (Entrada 40, 07/09/2026) — sonda de custo do fine-tuning do Kool: ~13,5s/passo de treino em CPU, sem GPU; formato exato do tensor de treino confirmado lendo `problem_vrp.py`; não existe flag `--train_dataset` — treino sempre gera dados on-the-fly.
- **#31** (Entrada 41, 07/09/2026) — a transição do efeito do drift-reset entre os α não é monotônica: mapear α 0,5→5,0 mostra um "vale" onde drift-reset *prejudica* (α≈3-4, -0,066/-0,351pp) antes de ficar fortemente benéfico em α=5,0 (+0,431pp).
- **#32-#34** (Entrada 42, 07/09/2026) — pacote local da Direção 1, três achados: **#32** primeira versão do gerador Uchoa tinha bug real de ordem de amostragem, corrigido e validado byte-a-byte (10/10) contra o script oficial. **#33** baseline "rollout" do Kool (usado nos checkpoints públicos) tem custo extra não-trivial não capturado pela sonda do #30; baseline "exponential" usado em vez disso para o fine-tuning. **#34** modelo do Kool não tem mecanismo para forçar exatamente K rotas — lacuna estrutural de justiça só para A/B.
- **#35** (Entrada 44, 08/09/2026) — a máquina real do pesquisador tem GPU NVIDIA (RTX 2050) — desconhecido durante toda a fase de design no sandbox (que assumiu pior caso CPU-only); melhora expectativas de tempo de fine-tuning, mas nenhuma decisão de design anterior precisou ser revertida.
- **#36** (Entrada 47, 09/09/2026) — "sem solução" do OR-Tools se concentra em exatamente 3 instâncias: sob o protocolo de tempo casado, 15/370 execuções (4,1%) falham, 100% concentradas em A-n45-k6, A-n61-k9, B-n57-k7 (todos os 5 seeds cada) — zero falhas em qualquer outra instância, zero em X/CMT.
- **#37** (Entrada 49, 09/09/2026) — taxa de falha do LKH3 é muito peor (23,5% vs. 4,1%): 8 instâncias totalmente perdidas (1 B + 7 X). Causa raiz: LKH3 trata a restrição de nº de veículos como termo suave/penalizado internamente (ver Seção 6, item 8).
- **#38** (Entrada 53, 15/09/2026) — gerador Uchoa é ~7% estruturalmente inviável em graph_size=10 (ver Seção 6, item 13). Corrigido via rejection sampling (opção escolhida pelo pesquisador em vez de modificar o gerador já validado).
- **#39** (Entrada 59, 16/09/2026) — **[ABERTO — ver Seção 9]** fine-tuning parece degradar o modelo do Kool, pior conforme o tamanho cresce: custo de validação piora +1,1%/+16,7%/+20,9%/+67,7% (época 0→14) para graph_size 10/20/50/100; norma do gradiente pré-clip explode com o tamanho (56,8→2477). Hipótese não testada: a vantagem de custo bruto do REINFORCE escala com graph_size enquanto lr/clip não se adaptam.

---

## 8. Log de bugs encontrados e corrigidos

**Núcleo bandit/VNS**
1. (Entrada 3, `instance.py`) off-by-one entre ids `.sol` (sem depósito) e `.vrp` (com depósito) — corrigido via `_sol_routes_to_vrp_ids()`.
2. (Entrada 3, `neighborhoods.py`) Relocate(2,0) `removal_gain` faltando o termo de arco interno `dist(c1,c2)` — corrigido, confirmado por verificador de força bruta independente.
3. (Entrada 3, `construction.py`) sem restrição de nº de veículos — baseline produzia custo abaixo do ótimo provado — corrigido via `repair_route_count()` (fallback estilo Fisher-Jaikumar).
4. (Entrada 3, `instance.py`) CMT erroneamente assumido K-exato — corrigido com `vehicles_exact` auto-detectado por nome.
5. (Entrada 5, `bandit.py`) contador de timestep incrementado dentro de `update()` em vez do topo de `select()` — termo log(s) do UCB1 atrasado uma iteração — corrigido.
6. (Entrada 5, `bandit.py`) deadlock cold-start/filtro-b_t com α=5,0 — corrigido (cold-start ignora o filtro b_t).
7. (Entrada 14, `bandit.py`) ADWIN alimentado com recompensa bruta, contradizendo paper e documentação própria — alternativa testada e descartada por ser pior (Entrada 15); documentação corrigida.
8. (Entradas 26/27, `bandit.py`) default do construtor `HGUCBBandit.__init__()` ainda dizia α=5,0 mesmo com 0,5 sendo o valor empiricamente escolhido desde as Entradas 6/12 — 5 scripts dependiam disso implicitamente. Corrigido: default do construtor → 0,5; os 5 scripts passados a especificar `alpha=5.0` explicitamente para preservar seus resultados históricos exatos.
9. (Entrada 27) a própria afirmação da Entrada 26 ("nenhum script dependia do fallback") estava errada — corrigida por grep exaustivo em vez de memória.

**Baseline OR-Tools**
10. (Entrada 35, achado #26) modelo padrão permite veículos vazios/não usados, quebrando a exigência de K-exato de A/B — corrigido com restrições `Start≠End` por veículo (K-exato) / teto de slack + restrição de soma (K-mínimo).
11. (Entrada 35) `IsEqualVar` (var-vs-var) usado erroneamente para comparar var-vs-constante (deveria ser `IsEqualCstVar`); resultado 0/1 sem `.Not()` — negação trocada por restrição de soma equivalente.
12. (Entrada 46, `rodar_ortools_74.py`) `RuntimeError` não capturado travava a execução das 74 instâncias no primeiro caso "sem solução" — corrigido espelhando o try/except do script irmão do LKH3.

**Baseline LKH3**
13. (Entrada 36, achado #27a) sem modo "ao menos K" nativo — corrigido com sweep de `VEHICLES` de K a K+5 para conjuntos K-mínimo.
14. (Entrada 36, achado #27b) LKH3 arredonda distância internamente mesmo para CMT float — corrigido: wrapper nunca confia no custo impresso pelo LKH3, sempre recalcula via `inst.solution_cost(routes)`.
15. (Entrada 48, `rodar_lkh3_74.py`) travava com `RuntimeError` quando nenhum candidato de K achava solução viável — a Entrada 46 tinha **erroneamente** afirmado que isso já estava tratado (checagem `if routes is None:` existente era código morto). Corrigido espelhando a correção do OR-Tools.
16. (Entrada 49, achado #37) 23,5% de taxa "sem solução" — não é bug de código; `is_feasible()` rejeita corretamente o resultado penalizado.
17. (Entrada 50) mais tempo resolve *algumas* instâncias perdidas (ex. X-n157-k13 em 60s) mas não outras (X-n172-k51, travada entre 15-60s e múltiplos seeds, no sandbox).
18. (Entrada 52) na máquina real, o retry recuperou 31/40 seeds perdidos (77,5%), mas X-n172-k51 — prevista como irrecuperável na Entrada 50 — na verdade teve 4/5 seeds recuperados; discrepância sandbox-vs-máquina-real de build/determinismo do LKH3 sinalizada como limitação genuína de previsão de casos de borda só em sandbox.

**Kool et al. (attention-learn-to-route)**
19. (Entrada 42, achado #32) primeira versão de `uchoa_gerador_core.py` amostrava `n_seeds` condicionalmente e calculava tamanho de rota após posicionar clientes, divergindo da ordem do script oficial — corrigido, validado byte-a-byte (10/10).
20. (Entrada 42, achado #33) baseline "rollout" do Kool tem custo extra não capturado pela sonda #30 — `kool_finetune_runner.py` usa "exponential" por padrão; remove defensivamente qualquer chave `'baseline'` obsoleta de checkpoints retomados.
21. (Entrada 42, achado #34) modelo do Kool não força exatamente K rotas — estrutural, não bug; `avaliar_kool_finetunado.py` reporta `viavel_estrito` e `viavel_sem_restricao_k` separadamente.
22. (Entrada 53, achado #38) gerador Uchoa ~7% inviável em graph_size=10 — corrigido via wrapper de rejection sampling em `kool_uchoa_monkeypatch.py`.
23. (Entrada 54) `train.py` do Kool hardcoda `num_workers=1`, sem override — travava com "too many fds" no WSL2/Python 3.14 do pesquisador — corrigido via patch (`sed` para 0) no arquivo vendorizado, aplicado idempotentemente em `setup.sh`.
24. (Entrada 55) após essa correção, travava em `loss.backward()` — kernel JIT Triton precisava de `Python.h` (faltava `python3.14-dev`) — corrigido instalando o pacote de sistema.
25. (Entrada 59, achado #39) fine-tuning parece degradar o modelo, pior com `graph_size` — **não corrigido, aberto no fim do diário.**

**Scripts de análise estatística**
Nenhum bug encontrado ou corrigido no código de teste de significância — todos validados contra CSVs sintéticos/valores conhecidos antes do primeiro uso real, nenhuma correção precisou ser feita depois.

---

## 9. Estado atual — onde paramos (LEIA ISTO)

**Última entrada registrada:** 59, 16/09/2026, achado real #39. Confirmado nesta sessão (04/10/2026) que nada novo foi executado desde então.

**Pendência aberta — achado #39 (degradação do fine-tuning do Kool):** o custo de validação piora de época 0 para 14 em todos os 4 tamanhos (pior quanto maior o `graph_size`), e a norma do gradiente pré-clip explode. Hipótese não testada: a recompensa bruta (não-normalizada) do baseline `ExponentialBaseline` do REINFORCE (`reinforce_baselines.py`, EMA com beta=0,8) escala com `graph_size`, mas `lr_model`/`max_grad_norm` ficam fixos — potencialmente sobre-corrigindo em tamanhos maiores.

**Próximo passo concreto, pronto para rodar:**
```bash
python3 kool_finetune_runner.py --graph_size 100 --lr_model 1e-5 --run_name teste_lr_baixo 2>&1 | tee log_teste_lr_baixo.txt
```
(rodar em `direcao1_local/`, com o venv ativado). Contrato confirmado do `kool_finetune_runner.py`: usa `argparse.parse_known_args()`, só `--graph_size` é obrigatório; monta os defaults de treino (`n_epochs=15, epoch_size=3200, batch_size=32, val_size=200, eval_batch_size=200, checkpoint_epochs=1, baseline=exponential, seed=2026`), carrega o checkpoint pré-treinado via `--load_path` (a menos que `--resume` esteja presente), e **os argumentos extra do usuário são sempre anexados por último** — ou seja, `--lr_model 1e-5` e `--run_name teste_lr_baixo` vencem os defaults pela semântica de "última ocorrência vence" do argparse.

**Como interpretar o resultado:** comparar custo de validação e norma do gradiente na época 0 vs. 14 deste novo run contra os logs já existentes (`log_finetune_100.txt`). Se o lr menor eliminar ou reduzir drasticamente a degradação, isso aponta para a hipótese do lr/clip-norm fixo como causa; se a degradação persistir igual, a causa é outra (ex.: algo na normalização Uchoa, ou no próprio baseline exponential sob fine-tuning). **Esta decisão — se/como re-rodar o fine-tuning para os tamanhos 20/50/100 a partir do resultado — é Eixo B (Seção 4.4): apresente o resultado ao pesquisador antes de se comprometer a uma re-execução completa.**

**Depois de resolvido:** só então rodar `avaliar_kool_finetunado.py` (Passo 4 da Direção 1) e fechar a comparação de 3 vias (OR-Tools + LKH3 + Kool fine-tunado) vs. HG-UCB.

---

## Apêndice A — log completo de decisões (AskUserQuestion)

*(cronológico; "Entrada X→Y" indica a decisão tomada ao fim da entrada X, aplicada a partir da Y)*

1. (Entrada 3) Como proceder com CVRPLIB+GVNS baseline? → "Seguir como planejado" — implementação nova, não reaproveitar código do EDP arquivado. *(antes da prática formal de AskUserQuestion, introduzida na Entrada 5)*
2. (Entrada 4) Qual(is) direção(ões) de contribuição? → **As três escolhidas**, não uma.
3. (Entrada 5) O que fazer com b_t=0,2 morto para k=5? → Manter b_t=0,2 literal + adicionar cap `max_inner_iter`.
4. (Entrada 5→6) Implementar e comparar as duas leituras de p̂ antes de escolher? → Sim, ambas (resultado: empate).
5. (Entrada 6→7) Prosseguir para validação completa da Tarefa #178 com p̂="q_values"? → Sim.
6. (Entrada 7→8) Como proceder com "x' fixo" tendo desempenho ruim? → Deixar "fixo" terminar, depois rodar "progressivo" completo.
7. (Entrada 8→9) Como proceder com a inversão de sinal (achado #3)? → Rodar os 3 diagnósticos do Gemini (opção recomendada).
8. (Entrada 9→10) Como proceder após achar o mecanismo de colapso (#4)? → Testar um sweep de alpha (recomendada).
9. (Entrada 10→11) Comprometer com validação completa, ou refinar mais? → Sweep mais fino + amostra expandida primeiro.
10. (Entrada 11→12) Dado que a faixa segura de alpha depende do conjunto, como proceder? → Investigar normalização de recompensa primeiro.
11. (Entrada 12→13) Validar só α=5,0 / só α=0,5 / ambos nas 74? → **Ambos** (mais completo).
12. (Entrada 13→14) Como perseguir o gap residual (3 sugestões do Gemini)? → Auditoria manual de bandit.py vs. paper.
13. (Entrada 14→15) Como testar a correção do ADWIN (#8)? → Testar na amostra barata de 8 instâncias primeiro.
14. (Entrada 15→16) Próximo passo após o resultado nulo do ADWIN? → Rastreamento detalhado iteração-a-iteração.
15. (Entrada 16→17) Como endereçar a assimetria de recompensa (#9)? → Testar reescalar a recompensa positiva.
16. (Entrada 17→18) Fechar a investigação (Gemini recomenda) ou checar mais uma hipótese? → Checar mesmo assim, por rigor (hipótese depois refutada).
17. (Entrada 18→19) Qual nova dimensão investigar? → `max_inner_iter` (recomendada).
18. (Entrada 19→20) O que fazer com o resultado misto do mii (#11)? → Medir %colapso em α=5,0/mii=160 (recomendada).
19. (Entrada 20→21) Ir direto para validação completa nas 74, ou mais uma rodada barata? → Mais uma rodada, expandida para 16 instâncias (**não** a opção recomendada).
20. (Entrada 21→22) Como resolver a inversão causada pelo CMT1 (#13)? → Testar as restantes CMT2/3/11 (recomendada; coincidiu com o Gemini independentemente).
21. (Entrada 22→23) Próximo passo após o efeito CMT unânime (#14)? → Validar ambos os candidatos nas 74 completas (opção cara, postergada duas vezes antes).
22. (Entrada 25→26) Decisão final de alpha/mii: manter default / mudar para mii=160 / mais seeds primeiro? → **Manter o default atual** (α=0,5/mii=40) — coincidiu com o Gemini.
23. (Entrada 26→27) Alinhar o default literal do construtor para α=0,5, ou manter α=5,0 como "default de fábrica"? → Alinhar para 0,5.
24. (Entrada 27→28) Confirmar que ambas direções ainda são necessárias; qual primeiro? → Ambas necessárias; começar pela Direção 2.
25. (Entrada 28a) Aprovar o design de ablação de 5 braços em 16 instâncias? → Aprovado como proposto.
26. (Entrada 28b) Adicionar um 6º braço "remover os 3" agora? → Sim, agora.
27. (Entrada 28c) Rodar o teste de estresse de drift-reset em α=5,0 agora, ou guardar para depois? → Rodar agora, em paralelo.
28. (Entrada 30→31) Depois da ablação Fase 1, expandir só os braços promissores / parar aqui / outro? → Expandir **todas as 5** comparações para n=74.
29. (Entrada 31→32) Depois da ablação n=74, mais seeds para cold-start / fechar Direção 2 / expandir em outro lugar? → Mais seeds, só cold-start.
30. (Entrada 32→33) Escopo da Direção 1 — todas as 74 ou só CMT4? Quais solvers? Qual protocolo de tempo? → Todas as 74; incluir Kool/Nazari também; casar orçamento de tempo com o do HG-UCB (**nenhuma** das opções conservadoras).
31. (Entrada 33→34) Lista final de solvers e estratégia de dados de fine-tuning, dados os achados de GPU/licença? → OR-Tools+LKH3+Kool(fine-tunado); **Nazari excluído**; reaproveitar o próprio gerador do Kool adaptado à clusterização estilo Uchoa.
32. (Entrada 38→39) Fechar Direção 2 / investigar interações entre mecanismos / mapear a transição de alpha? → Mapear a transição α 0,5→5,0.
33. (Entrada 41) Escopo de "mover execução pesada para a máquina local"? Incluir script de setup? → "Tudo que é pesado"; sim, incluir script de setup (recomendada).
34. (Entrada 47→48) O que fazer com as 3 instâncias "sem solução" do OR-Tools (#36)? → Aceitar como está (exclusão documentada).
35. (Entrada 49→50) O que fazer com as 8 instâncias "sem solução" do LKH3 (#37)? → Investigar mais primeiro (mais tempo ajuda em algo?).
36. (Entrada 50→51) Dado resultado não-uniforme, tentar de novo as 8 / aceitar como está (recomendada) / mapear as recuperáveis primeiro? → Tentar de novo mesmo assim, **aceitando um resultado parcial**.
37. **(04/10/2026, esta sessão)** Modelo de autonomia de execução para sessões Claude Code? → **"Autonomia total, reporta depois"** — ver Seção 4.4 para a reconciliação com o gate de metodologia, que continua valendo.
38. **[PENDENTE — ver Seção 9]** O que fazer com a degradação do fine-tuning do Kool (achado #39)? → Ainda não respondida.

---

## Apêndice B — gotchas de ambiente/engenharia

- **GPU presente:** a máquina real do pesquisador tem uma NVIDIA RTX 2050 (achado #35) — detectada automaticamente pelo `kool_finetune_runner.py` via `torch.cuda.is_available()`; use `--no_cuda` para forçar CPU deliberadamente.
- **protobuf vs. `tensorboard_logger`** (Entrada 61): o venv de `direcao1_local` falha ao importar `tensorboard_logger` (protobuf novo). Contorno: `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` antes de lançar `kool_finetune_runner.py`. Um job Python em background no WSL fica com o log vazio por buffering de stdout até terminar — o progresso real aparece nos checkpoints `epoch-N.pt` ou use `python3 -u`.
- **HGS-CVRP** (Entrada 61): código em `external/HGS-CVRP/` (MIT, terceiros, não sincronizado), compilado com `g++ -std=c++17 -O3 -DNDEBUG Program/*.cpp -o hgs` (WSL sem cmake). `-veh K` é teto de frota, não igualdade.
- **Risco de versão de ambiente:** pipeline roda em versões incomumente recentes e pouco testadas (Python 3.14, build muito novo de PyTorch), implicadas em 2-3 falhas consecutivas puramente de ambiente (Entradas 54/55). PyTorch **não está pinado** em `setup.sh` (Entrada 55) — considerar pinar se mais fricção aparecer.
- **Riscos de execução overnight em WSL2** (Entrada 58): suspender/hibernar pausa o WSL; bateria não sustenta ~10h de execução em GPU; incerto se fechar todas as janelas de terminal mata a distro mesmo com um processo em background rodando — mitigar desabilitando hibernação, mantendo conectado à energia, minimizando (não fechando) terminais.
- **Peculiaridade de captura de PID em background** (Entrada 41): este tipo de ambiente envolve cada chamada Bash numa camada extra de `bash -c 'eval ...'`, então `$!` capturado dentro de uma chamada Bash multi-linha pode apontar para o PID errado — sempre confirme via `ps aux`, nunca confie em `$!` às cegas depois de um lançamento em background multi-linha.
- **Um job em background pode morrer silenciosamente** (sem traceback, sem evidência de OOM) por problemas transitórios de conectividade do container/MCP (Entradas 21, 41) — sempre verifique conclusão/estado de PID real, não assuma que um job longo ainda está rodando só porque nada deu erro.
- **Limite de tokens do Read tool** (descoberto nesta sessão, 04/10/2026): leituras de arquivo muito grandes falham acima de ~25000 tokens por chamada. Para o diário (conteúdo denso em prosa+tabela, ~89 tokens/linha), isso significa um máximo de ~200-220 linhas por chamada de `Read` com `offset`/`limit`. Relevante para qualquer leitura futura de arquivo grande neste projeto.
- **Verificação por SHA-256 estava quebrada** (desde ~08/09/2026 até pelo menos esta sessão) no workflow antigo via Cowork, por um problema de montagem atribuído a uma atualização do Windows — mas isso só importava para o workflow de sincronização obsoleto (Seção 4.3); rodando Claude Code direto na máquina, não há sincronização para verificar.
- **Processo anterior (EDP, arquivado) deixou uma lição estrutural** que já moldou este projeto desde a Entrada 1: "complexidade acumulada sem um resultado convincente" levou à disciplina de "um mecanismo de aprendizado só, não uma pilha de flags de ablação independentes" — mantenha essa disciplina ao adicionar qualquer novo hook/flag.
