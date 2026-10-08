# Rastreio dos números do rascunho (paper/main.tex)

Regra (LEI_GERAL_PESQUISA.md, Parte I.2): nenhum número "final" sem rastro até o script/arquivo exato. Esta tabela liga cada número do texto à sua origem. `Diário` = `../../diario_pesquisa.md`. Estado: rascunho de 05/10/2026; Estudo 3 pendente.

| Afirmação no texto | Valor | Origem |
|---|---|---|
| GVNS clássico reproduzido vs. artigo (A/B/X) | 3,35% vs. 3,39% | Diário, Entrada 13 (achado #7); `results/gvns_baseline_*.csv` |
| Colapso do softmax (α=5,0, recompensa bruta) | 99,6% das iterações (8 instâncias x 5 seeds) | Diário, Entrada 12 (tabela do diag6); `tests/diag6_reward_normalization.py` |
| Colapso com recompensa normalizada, mesmo α | 12,5% | idem |
| Condição de parada max(q)<b_t inatingível (k=5, b_t=0,2) | prova + 200 mil amostras | Diário, Entrada 5 (achado #1) |
| ADWIN alimentado com Q-values piora o gap | — | Diário, Entrada 15 (diag7) |
| Tabela 1 (gaps, 74 instâncias, B=800/4.000) | 3,01/2,30; 3,42/2,74; 3,42/2,75; 3,47/2,79; 3,47/2,84; 3,50/2,87; 4,43/4,15 | Diário, Entradas 66 e 68; `results/aos_study/full.csv`; `estudo_aos/analisar_aos.py` |
| HG-UCB vs. uniforme (B=800) | +0,056pp, p ajust. 1,0 | Entrada 66/68; família de 12 testes (Entrada 69) |
| HG-UCB vs. uniforme (B=4.000) | +0,100pp [0,013; 0,188], p ajust. 0,107 (m=12) | Entrada 68 (tabela m=12 vs. m=18) |
| HG-UCB vs. UCB1/ε-greedy/PM (p ajust. mínimo) | 0,158 (UCB1, B=4.000) | Entrada 68, coluna p_holm_m12 |
| Literal vs. normalizado | +0,960pp (B=800), +1,310pp (B=4.000), 0/74 vitórias | Entradas 66 e 68; `results/aos_study/analise_confirmatoria.csv` |
| GVNS vs. HG-UCB | 0,457 / 0,540pp; 63/74 | idem |
| Subconjunto B=20.000 (22 inst., 5 seeds) | −0,060, −0,110, +0,212pp | Entrada 72; `results/aos_study/exploratorio_b20000.csv` |
| Diagnóstico de estrutura (85-90% após a última melhora; 60/165/138 shakes) | 3 instâncias, 1 seed | Entrada 70 (bancada, não versionada) |
| Tabela 2A (6 estruturas) | 2,69/1,94; 2,90/2,11; 3,01/2,30; 3,21/2,60; 3,33/2,54; 3,42/2,74 | Entradas 73 e 74; `results/aos_study/etapa2_full.csv` + `full.csv`; `estudo_aos/analisar_etapa2.py` |
| Término LO vs. F40 | −0,637/−0,518 (B=800); −0,597/−0,623 (B=4.000); 66-70 de 74 | Entrada 74; `results/aos_study/analise_etapa2_confirmatoria.csv` (C1, C2) |
| Aceitação best vs. shaken | −0,090 a −0,210; C3 B=4.000 p ajust. 0,054 | idem (C3, C4) |
| Ordem aleatória vs. fixa | +0,326/+0,358 (best), +0,308/+0,488 (shaken) | idem (C5, C6) |
| 2-Opt absorve 39% das aplicações na ordem fixa | 39,4-39,7% | Entrada 74, tabela de uso de operadores |
| Políticas adaptativas vs. ordem aleatória (2B) | UCB1 +0,019/+0,063; PM +0,091/+0,053; ε-greedy +0,226/+0,186; HG-UCB +0,160/+0,138 | idem (D1-D4) |
| Adaptativas vs. ordem fixa | −0,100 a −0,308pp | idem (D5-D8) |
| 21 de 28 testes significativos (Holm m=28) | 21/28 | Entrada 74 |
| Do loop original à estrutura aleatória/best/LO | 2,74 → 1,94 (B=4.000), −0,80pp | Entrada 74 |
| HGS-CVRP | 0,035% (orçamento do HG-UCB, 0,2-10 s); 0,010% (0,1 s/cliente); 370/370 viáveis | Entrada 61; `results/hgs_74_matched.csv`, `results/hgs_74_n01.csv` |
| Kool et al. zero-shot | 9,43% (58/74 viáveis no critério estrito) | Entrada 63; `results/kool_zeroshot_74.csv` |
| Tempo por aplicação ~8-12 ms | — | Entrada 74 (exploratório); `analisar_etapa2.py` |
| Estudo 3: tabela de gaps (13 braços, E1/E2) | ver Tabela 3 do texto | Entrada 81; `results/aos_study/etapa3_full.csv`; `estudo_aos/analisar_etapa3.py` |
| Conjunto estendido (T1) | −0,500 / −0,532pp; 59/58 de 74 | Entrada 81; `analise_etapa3_confirmatoria.csv` (T1) |
| Ordem fixa vs. aleatória em P5, orçamento em distâncias (T3) | +0,258 / +0,342pp | idem (T3) |
| Ordem por custo pior que aleatória | LO +0,133/+0,190; R5 E2 +0,278 | idem (LO_FOCvsU, R5_FOCvsU) |
| HG-UCB pior que uniforme em E2 | LO +0,113; R5 +0,199 | idem |
| UCB1/PM vs. FOC em E2 | −0,176 a −0,312pp | idem |
| Recompensa sensível a custo vs. cega | −0,002 (p=0,71); −0,042 (p=0,22) | Entrada 81 (exploratório) |
| Fração das avaliações em SWAP* | 47-61% (U e AOS); 24-45% (FOC) | Entrada 81; coluna `calls_por_operador` |
| 13/34 testes significativos (3 em E1, 10 em E2) | 13/34 | Entrada 81 |
| Calibração E2 vs. B=4.000 (conjunto original, LO, aleatória) | 1,98% vs. 1,94% | Entrada 81 (P5_LO_U E2) e Entrada 74 (U/best/LO B=4.000) |
| Custo por aplicação, n=100 (2-Opt 633 ... SWAP* 34.817 dist.) | 0,6k-35k | Entrada 76 (perfil, 3 instâncias) |
| Validação dos operadores novos | 1.600 testes, mutação detectada 29/40 | Entrada 76; `tests/validate_neighborhoods_ext.py` |

## Pendente / não verificado
- Referências pendentes de verificação: Augerat et al. (1995), Christofides-Mingozzi-Toth (1979), Bifet & Gavaldà (2007), Sutton & Barto (2018). O texto NÃO as cita ainda.
- Número de páginas e formato exato do CEC 2027 [VERIFICAR no chamado oficial].
- Declaração de uso de IA generativa, disponibilidade de dados/código (link persistente), conflito de interesse, financiamento.
- A pré-especificação está num diário datado sem carimbo de tempo externo (o projeto não é repositório git); considerar `git init` + depósito em OSF/Zenodo.
