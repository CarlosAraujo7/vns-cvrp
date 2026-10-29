# Rastreio dos números do manuscrito EJOR (regenerado em 28/10/2026; Estudos 1-12)

Regra (LEI_GERAL_PESQUISA.md, Parte I.2): nenhum número "final" sem rastro até o script/arquivo exato. **Todos os números do texto de Resultados, as tabelas e as figuras são GERADOS por script a partir dos CSVs brutos** — nada é digitado à mão nessas partes. Esta versão substitui a tabela manual anterior (que listava valores pré-correção da solução inicial, Entrada 90).

## Pipeline

```
results/aos_study/*.csv (execuções brutas, uma linha por braço x nível x instância x seed)
   |  estudo_aos/analisar_*.py          testes confirmatórios (Holm, bootstrap) -> analise_*_confirmatoria.csv
   |  estudo_aos/analise_robustez.py    equivalência, IC em dois níveis, MDE      -> analise_robustez.csv
   v
paper/ejor/gerar_tabelas.py             -> tab_*.tex, numeros.tex (macros \num{chave}), numeros.json, suptab.tex
paper/ejor/gerar_figuras.py             -> fig_*.pdf
paper/ejor/gerar_suplemento.py          -> suplemento/*.csv, tab_s2set.tex, tab_s3set.tex
   v
estudo_aos/verificacao_independente.py  recalcula, só com a biblioteca padrão, médias e contrastes (diferença, vitórias, derrotas)
paper/ejor/checar_macros.py             chaves \num{} usadas existem; sem caracteres de controle; highlights, palavras-chave, resumo
```

Ordem para regenerar tudo: `analisar_aos.py`, `analisar_etapa{2,3,4,5,6,7,8,9,10,11,12}.py`, `analise_robustez.py`, `gerar_tabelas.py`, `verificacao_independente.py`, `gerar_tabelas.py` (de novo, para ler o resultado da verificação), `gerar_figuras.py`, `gerar_suplemento.py`, `checar_macros.py`, compilar `main.tex` e `supplement.tex`.

## Famílias de macros (`numeros.json`; prefixo → origem)

| Prefixo | Significado | Origem |
|---|---|---|
| `s1_gap_*`, `s1{d,a,ci,p,w,l}_*` | Estudo 1: gap médio por braço; diferença (com sinal, módulo), IC95%, p ajustado (Holm m=12), vitórias, derrotas | `full.csv`, `analise_confirmatoria.csv` |
| `s2a_gap_*`, `s2{d,a,ci,p,w,l}_C*/D*` | Estudo 2: gaps por estrutura; contrastes C1-C6 (estrutura) e D1-D8 (AOS no VND), Holm m=28 | `etapa2_full.csv`, `analise_etapa2_confirmatoria.csv` |
| `s3_gap_*`, `s3{d,a,ci,p,w,l}_*` | Estudo 3: gaps; 17 contrastes por nível, Holm m=34 | `etapa3_full.csv`, `analise_etapa3_confirmatoria.csv` |
| `s4_*`, `s4{d,a,ci,p,w,l}_V*/W*`, `s4v_gap_*` | Estudo 4: ajuste, mistura estática, contrastes V1-V7 (CVRPLIB, Holm m=14) e W1-W6 (SYN_VAL, Holm m=6) | `etapa4_*.csv`, `etapa4_tuning_escolhidas.json`, `analise_etapa4_confirmatoria.csv` |
| `s5_*` | Estudo 5: atribuição do efeito do conjunto de operadores (Holm m=20) | `etapa5_full.csv`, `analise_etapa5_confirmatoria.csv` |
| `s6_*`, `s6x_*` | Estudo 6: famílias adicionais de AOS (Holm m=12); `s6x` = exploratório contra a mistura estática | `etapa6_full.csv`, `analise_etapa6_confirmatoria.csv` |
| `eq_*`, `mde_*`, `rb2_*`, `rb90_*`, `rb1_*`, `rbd_*`, `rb_wider_*`, `var_*`, `seed_*` | Robustez: equivalência, IC em dois níveis, MDE, componentes de variância | `analise_robustez.csv`, `analise_robustez.py` |
| `s7_*`, `s7{d,a,ci,p,w,l}_*`, `s7eq_*` | Estudo 7 (HGS-CVRP): gaps por braço/população/ponto de controle, contrastes (Holm m=20), CPU por geração, taxas de sucesso por movimento | `etapa7_full.csv`, `analise_etapa7_confirmatoria.csv`, `analise_etapa7_sensibilidade.csv` |
| `s7pr_*`, `s7_mde_med` | Estudo 7: p bruto (Wilcoxon, antes do Holm) por contraste e MDE mediano (80% de poder) dos 20 testes confirmatórios (adicionados na auditoria do ciclo 4, Entrada 99) | `analise_etapa7_confirmatoria.csv` (colunas `p_bruto`, `mde_pp`) |
| `s8_*`, `s8{d,a,ci,p,w,l}_*`, `s8rec_*` | Estudo 8 (controle positivo, operadores inúteis plantados): gap por braço e pool, contrastes (Holm m=12), recuperação da perda plantada | `etapa8_full.csv`, `analise_etapa8_*.csv` |
| `s9_*`, `s9{d,a,ci,p,w,l}_*`, `s9prim_*`, `s9ta_*` | Estudo 9 (ALNS com operadores estocásticos): gaps, teste primário (RP − U, α=0,05), secundários (Holm m=12), terciários (referências VNS, Holm m=4) | `etapa9_eval.csv`, `etapa9_vns.csv`, `etapa9_tuning_escolhidas.json`, `analise_etapa9_confirmatoria.csv` |
| `s10_*`, `s10{d,a,ci,p,w,l}_*`, `s10pr_*`, `s10mde_*`, `s10rho_*`, `s10dn*_*`, `s10ph_*` | Estudo 10 (instâncias grandes): gaps, contrastes (Holm m=7), tendência com n (exploratório), participação do SWAP* e CROSS, hipótese pós-hoc dos Estudos 3-4 | `etapa10_eval.csv`, `etapa10_tuning_escolhida.json`, `analise_etapa10_*.csv` |
| `s11_*`, `s11{d,a,ci,p,w,l}_*`, `s11pr_*`, `s11mde_*`, `s11vs3_*`, `s11rho_*` | Estudo 11 (SWAP* eficiente): gaps por braço e cenário (E1, E2, L), Família A (Holm m=21) e B (Holm m=2), participação do SWAP* e CROSS, custo/eficiência do SWAP* rápido, comparação descritiva com o SWAP* ingênuo (Estudos 3 e 5), controle de regressão exata do braço P5 (1.480/1.480) | `etapa11_eval74.csv`, `etapa11_evalL.csv`, `etapa11_tuning_escolhida.json`, `analise_etapa11_*.csv` |
| `s14_*`, `s14{d,a,ci,p,w,l}_*` (A, C), `s14e3*`, `s14x*`, `s14q*`, `s14pc*`, `s14rec*`, `s14pr_*`, `s14mde_*` | Estudo 14 (CVRPTW, VND em Solomon): gaps por braço e cenário (E1, E2, E3), Família A (Holm m=14), B (m=2), C (m=6), exploratórias E3 (m=11) e X (m=18), análise PÓS-HOC do pool permutado (m=20), controle positivo (m=12) e recuperação, uso de operadores, ajuste, reprodução do BKS (56/56) | `etapa14_eval_{v,pc}.csv`, `etapa14_explor_perm.csv`, `etapa14_tuning_*`, `etapa14_solomon_bks_check.csv`, `analise_etapa14_*.csv` |
| `s13_*`, `s13{d,a,ci,p,w,l}_*` (A, B), `s13e3*`, `s13dd_*`/`s13da_*`..., `s13p5*`, `s13o*`, `s13q*`, `s13pc*`, `s13rec*`, `s13a_*`, `s13a{d,a,ci,p,w,l}_*`, `s13ah*`, `s13r_*`, `s13apr_*` | Estudo 13 (TSP): gaps por braço e cenário (T1 = TSPLIB E1, T2 = E2, S1 = sintéticas E1, T3 = E3), Família A (Holm m=21), B (m=3), exploratórias C/D, controle positivo (m=12) e recuperação, ALNS (primário, secundários m=12, terciários m=4, estrutura m=6), análises PÓS-HOC (P5 m=9, ordem fixa m=24, pool permutado m=12, ALNS m=2), uso de operadores, ajuste, equivalência regret-2/guloso (`etapa13_regret_equiv.txt`) | `etapa13_eval_{v,pc,a,ref}.csv`, `etapa13_explor_{ordem,perm}.csv`, `etapa13_tuning_*`, `analise_etapa13_*.csv` |
| `s12_*`, `s12{d,a,ci,p,w,l}_*`, `s12pr_*`, `s12cap_*`, `s12capci_*` | Estudo 12 (controle estático no ALNS): gaps, contrastes (Holm m=5), fração do ganho do RP capturada, pesos da mistura ajustada | `etapa12_eval.csv`, `etapa12_tuning_escolhida.json`, `analise_etapa12_*.csv` |
| `g_ord_r1`, `g_pool_r1`, `g_gvns_r1`, `g_struct_*` | Resumo e Discussão: faixas com 1 casa decimal (ordem aleatória, conjunto, GVNS−HG-UCB), menor/maior efeito estrutural (término, ordem, conjunto, incl. conjunto em SYN_VAL) e razão em relação a 0,14% (Turkeš et al.) | `gerar_tabelas.py::agrupamentos` sobre `s2a_*`, `s3a_T1_*`, `s4a_W1`, `s1a_*` |
| `compl_*` | Diagnóstico de complementaridade (exploratório): % de ótimos locais dos 5 operadores originais em que cada operador adicionado acha melhora | `diag_complementaridade.csv`, `estudo_aos/diag_complementaridade.py` |
| `u_*` | Contagens das afirmações universais (contrastes política − baseline: significativamente melhores/piores; 94 contrastes dos Estudos 1-4, 6, 7, 10 e 11 desde a Entrada 112; 66 sem os Estudos 10 e 11 (Entrada 99); antes 50 sem o Estudo 7; HG-UCB: 15 contrastes incl. o HG-UCB-lite do Estudo 7) | `analise_etapa*_confirmatoria.csv` via `paper/ejor/checar_afirmacoes.py` |
| `eff_*`, `s2_*`, `s3_swap*`, `diag_*`, `sh_*` | Instrumentação: eficiência por operador, % das avaliações por operador, diagnóstico de estrutura, nº de shakes | colunas `calls_por_operador`, `suc_por_operador`, `ganho_por_operador`, `shakes` dos CSVs; `diag_estrutura.csv` |
| `rep_*` | Fidelidade da reprodução (tabela 1) | `results/gvns_baseline_*.csv`, `results/bandit_vns_validation_*.csv` (A e B reexecutados em 09/10/2026) |
| `hgs_*`, `kool_*`, `vns_gap_*` | Tetos externos e faixa de gap das variantes VNS | `results/hgs_74_*.csv`, `results/kool_zeroshot_74.csv` |
| `sub_*` | Subconjunto exploratório B=20.000 (22 instâncias) | `exploratorio_b20000.csv` |
| `overshoot_*`, `ver_*` | Estouro médio do orçamento; contagem da verificação independente | CSVs das Etapas 3-6; `verificacao_independente.json` |

## Valores que NÃO são gerados por script (fonte primária)

| Afirmação no texto | Valor | Origem |
|---|---|---|
| Valores publicados do VNS e do HG-UCB (tabela 1) | A 2,040/1,610; B 1,832/1,655; X 6,291/6,023; média 3,388/3,096 | Kalatzantonakis, Sifaleras e Samaras (2023), Tabelas 10-12; PDF em `literatura/`; Diário, Entrada 84 (conferido contra o PDF) |
| Colapso do softmax (α=5, recompensa bruta): 99,6% das iterações com max q>0,99 (8 instâncias); 12,5% com a recompensa normalizada | — | Diário, Entradas 9 e 12; `tests/diag6_reward_normalization.py` |
| Condição de parada max(q)<b_t inatingível para k=5, b_t=0,2 | prova | Diário, Entrada 5 |
| ADWIN alimentado com Q-values piora o gap | — | Diário, Entrada 15 |
| Ajuste do artigo original em 3 instâncias, 2 delas avaliadas depois (B-n78-k10, X-n120-k6) | — | Diário, Entrada 84 (releitura do PDF, Tabela 3) |
| Custo por avaliação de distância 0,35-0,57 µs (n de 53 a 195) | — | Diário, Entrada 76 (perfil) |
| Operadores novos validados por força bruta | 1.600 testes; teste de mutação | `tests/validate_neighborhoods_ext.py`; Diário, Entrada 76 |
| Solução inicial não determinística: 26 de 740 execuções divergentes; 11 instâncias com fallback CP-SAT | — | Diário, Entradas 88 e 90; `instances/_init/_relatorio.csv` |
| HGS-CVRP: commit `1a92795`, g++ 15.2.0 -O3 | — | `cvrp_bandit/README.md`; Diário, Entrada 61 |
| Meta-análise do ALNS: 25 implementações; a camada adaptativa melhora a função objetivo em 0,14% em média (IC95% 0,06-0,21%) | — | Turkeš, Sörensen e Hvattum (2021), EJOR 292, 423-442; resumo lido na página oficial (ScienceDirect, `S037722172030936X`) em 09/10/2026 |
| Requisitos do EJOR (35 páginas, 11 pt, 1,5, 250 palavras, 1-5 palavras-chave, highlights 85 car.) | — | Guia de autores do EJOR lido em 09/10/2026 (sciencedirect.com/journal/european-journal-of-operational-research/publish/guide-for-authors); Diário, Entrada 92 |

## Pendente / não verificado
* Licença do repositório, DOI do depósito (Zenodo ou equivalente), autoria, CRediT, financiamento, declaração de IA: preenchimento genérico no manuscrito (`\authorpend{}`), a editar pelo pesquisador.
