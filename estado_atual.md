# Estado atual consolidado — Direção A (AOS no VNS para CVRP), alvo EJOR

*Nível 2 do arquivamento (GUIA_PESQUISA_GERAL.md, Seção 6). Sincronizado com a Entrada 112 do diário (28/10/2026). O diário (`diario_pesquisa.md`) é o Nível 1; este arquivo diz o que é verdade AGORA.*

## Resumo executivo
Reproduzimos o HG-UCB (Kalatzantonakis et al., 2023, ESWA) acoplado a um VNS para CVRP e o auditamos em **doze estudos controlados e pré-especificados** (74 instâncias CVRPLIB + 86 sintéticas de até 500 clientes, 10 ou 5 seeds, orçamentos iguais, testes pareados com Holm, bootstrap, equivalência, controle positivo). O ganho publicado (−0,29pp sobre o VNS) **não é reproduzido**; a recompensa em unidades de custo satura o softmax (normalizar resolve); com a recompensa corrigida e operadores determinísticos, **nenhuma política adaptativa testada supera a seleção uniforme** (Estudos 1-4, 6, 7, 10 e 11; 94 contrastes, 0 melhores, 23 piores), nem com o SWAP* eficiente (Estudo 11); o controle positivo (Estudo 8) mostra que o protocolo detecta efeitos (Thompson e eficiência-UCB recuperam 66-84% da perda plantada). No ALNS com operadores estocásticos (Estudo 9) os pesos adaptativos de Ropke-Pisinger vencem a uniforme por 0,36pp, **mas uma mistura estática ajustada offline e um reparo fixo (regret-2) fazem igual ou melhor (Estudo 12): a adaptação SUBSTITUI o ajuste, não o supera**. O que melhora o resultado é a estrutura (término no ótimo local, ordem aleatória, conjunto de operadores e, sobretudo, destruir-e-reconstruir: o ALNS bate o melhor VNS por 0,72-1,68pp no mesmo orçamento).

## Resultado central (números finais, com fonte)
Todos os números do texto de Resultados são gerados por script (`paper/ejor/gerar_tabelas.py` → `numeros.json`); proveniência em `cvrp_bandit/paper/RASTREIO_NUMEROS.md`. Verificação independente: `cvrp_bandit/estudo_aos/verificacao_independente.py` v3 (102 médias + 94 contrastes OK em 09/10/2026, antes do Estudo 6). Afirmações universais conferidas por `paper/ejor/checar_afirmacoes.py`.
* Estudo 1 (loop original): HG-UCB ≈ uniforme (+0,06/+0,10pp, n.s.); literal +0,96/+1,32pp pior que normalizado em 74/74; GVNS clássico ~0,5pp melhor que HG-UCB em aplicações iguais.
* Estudo 2: término LO −0,53 a −0,66pp; ordem aleatória melhor que fixa por 0,30-0,48pp; dentro do VND ε-greedy e HG-UCB piores que ordem aleatória, UCB1 marginalmente pior em B=4.000 (p aj. 0,043); 23/28 testes significativos.
* Estudo 3 (10 operadores): conjunto estendido −0,50/−0,54pp; ordem por custo pior que aleatória; nenhuma AOS supera U; HG-UCB pior em E2; 13/34 significativos.
* Estudo 4 (ajuste, mistura estática, validação sintética): nenhuma AOS ajustada supera U; HG-UCB ajustado pior em E2 (+0,14pp, p aj. 0,004); mistura estática −0,13pp (E1, n.s. após Holm)/0,00 (E2); o efeito do conjunto replica no SYN_VAL (−0,79pp); 2/14 e 1/6 significativos.
* Estudo 5: ver acima; 18/20 testes significativos; efeitos de A1-A5 em `analise_etapa5_confirmatoria.csv` (Entrada 94).
* Estudo 6 (8 braços, Holm m=16; Entrada 95): 0/16 melhoras, 9/16 piores; a política de eficiência (ganho por avaliação) é a PIOR (+0,87pp em E2) porque tira avaliações dos operadores caros; EXP3 e SW-UCB empatam com a uniforme; contextuais pioram. Entre os 50 contrastes política − baseline dos Estudos 1-4 e 6: 0 melhoras significativas, 17 piores.
* **Estudo 7 (Entrada 98): dentro do HGS-CVRP (busca local de estado da arte), a ordem adaptativa dos movimentos de cada par de nós não tem efeito: 0 de 20 testes significativos, maior |dif| 0,05pp (≤ 0,03pp entre as adaptativas); a ordem original do HGS e a aleatória não diferem; as adaptativas são 1,8-10x mais lentas por geração que a ordem original.**
* **Estudo 8 (Entradas 101-102, controle positivo):** com operadores inúteis plantados, o aparato detecta a perda; Thompson sampling e eficiência-UCB recuperam 66% e 84% (pool D4x8: quatro operadores inúteis de 8n²); ambos perdem no pool limpo.
* **Estudo 9 (Entradas 103, 107, 108, 110, ALNS com contabilidade corrigida):** pesos adaptativos RP −0,36pp contra U (p<1e-4); 5 de 12 secundários significativos; ALNS-U bate o melhor VNS por 0,72-1,68pp.
* **Estudo 10 (Entrada 105):** a hipótese pós-hoc (a mistura estática ajudaria em n grande) NÃO se replica em 20 instâncias de 200-500 clientes; Thompson e eficiência pioram.
* **Estudo 12 (Entrada 111):** mistura estática −0,37pp e reparo fixo regret-2 −0,47pp contra U; RP − estática +0,01pp (n.s.); RP − reg2 +0,11pp (p aj. 0,023).
* **Estudo 11 (Entrada 112, SWAP* eficiente):** H45 confirmada (0 de 18 políticas melhores que U; eficiência-UCB +0,46 a +0,96pp e HG-UCB-c +0,18pp em E2 piores); H46 não confirmada (mistura estática −0,07/+0,03/−0,32pp, p aj. 1); H47 confirmada (efeito do conjunto −1,03/−0,93pp, maior que −0,73/−0,75 com o SWAP* ingênuo); SWAP* rápido absorve 17% das avaliações; o braço P5 reproduz o Estudo 5 em 1.480/1.480 execuções.
* Diagnóstico de complementaridade (Entrada 96, exploratório): em ótimos locais dos 5 operadores originais, SWAP* acha melhora em 63,8% e CROSS em 32,3%; em 49,8% só os caros acham.
* Robustez: equivalência a ±0,10pp em 18/58 contrastes (Estudos 1-4 e 6); ganhos de AOS > ~0,15pp excluídos (limite inferior do IC90% mais favorável entre as AOS); MDE mediano 0,12pp.
* Teto HGS: 0,035% (orçamento do HG-UCB) / 0,010% (0,1 s/cliente). Kool et al. zero-shot: 9,43%.
* Diagnóstico de estrutura (`diag_estrutura.py`, 74 inst. x 10 seeds): 82% das aplicações do laço original vêm depois da última melhora do passo; 54% depois que os 5 operadores já falharam.

## O que NÃO usar (números descartados ou superados)
| Item | Motivo |
|---|---|
| Qualquer número das Entradas 66, 68, 73, 74, 79, 81, 84, 85, 86, 87 (computados com a solução inicial não fixada) | Superados pela correção F1 (Entradas 90, 92); usar os valores gerados dos CSVs corrigidos. |
| "O efeito do conjunto estendido vem do SWAP*" (rascunho de 05/10) e "vem dos operadores baratos" (Entrada 88, F2) | Refutado pela Etapa 5: os dois grupos importam; o par caro vale mais que o trio barato (Entrada 94). |
| "HG-UCB (3,41%) vence OR-Tools (5,22%) e LKH-3 (5,02%), p≈0,001" (Entradas 47/49/61) | Vale só sob orçamentos de 0,2-10 s que inviabilizam LKH-3 (23,5% sem solução); NÃO é afirmação de qualidade de método. |
| "Vantagem de α=5,0/mii=160 sobre α=0,5" (Entradas 19-20) | Inverteu-se ao ampliar a amostra (Entradas 21-25). |
| "AOS adaptativa melhora sobre a ordem fixa" (Etapa 2, D5-D8; Etapa 3 vs. FOC) | Reflete o efeito da ordem aleatória / a fraqueza da ordem por custo, não a adaptação. |
| "p ajustado ≥ 0,15" para HG-UCB vs UCB1/ε-greedy/PM (Etapa 1) | Corrigido para ≥ 0,097 após a correção F1. |
| Resultados do fine-tuning do Kool (#39/#40) | Zero-shot (9,43%) é o único número de contexto. |
| "Tempo e custo estimados" das Entradas 64-65 | Erradas por ~5x (Entrada 67). |
| `paper/main.tex` (rascunho IEEE antigo) e `paper/cec_rascunho_antigo.tex` | Versão antiga; o manuscrito vigente é `paper/ejor/main.tex`. |

## Pendências (ver Entrada 92-94 e o PONTO DE RETOMADA do `cvrp_bandit/CLAUDE.md`)
1. Doze estudos concluídos e integrados (Entrada 112); o manuscrito compila com 46 páginas (o pesquisador vai cortar para ~30-35) e o suplemento com 16; verificação independente OK (387 contrastes). Falta o ciclo final de auditoria de revisor sobre o conjunto dos doze estudos (regra master) e as decisões do pesquisador abaixo. Opcional: segundo problema/solver para ampliar a generalidade.
2. Manuscrito EJOR em `cvrp_bandit/paper/ejor/` compila no WSL (`ferramentas/tectonic`); limite de 35 páginas (11 pt, 1,5) — verificar a contagem final.
3. Autoria, CRediT, declaração de IA, financiamento, conflitos, DOI do repositório: preenchimento genérico (`\authorpend{}`) a ser editado pelo pesquisador.
4. Licença do repositório (sugestão: MIT para o código, CC BY 4.0 para dados) — decisão do pesquisador.
5. Implicação do guia do EJOR ("extended versions of papers published in copyrighted conference proceedings are not considered"): submeter o EJOR primeiro; qualquer versão CEC deve ser claramente distinta — decisão do pesquisador.
