# Direção 1 — execução na sua máquina

Este diretório reúne TUDO que é "pesado" da Direção 1 (comparação do
nosso HG-UCB contra OR-Tools, LKH3 e Kool et al. fine-tunado, nas 74
instâncias completas A/B/X/CMT) para rodar na sua própria máquina, em vez
do sandbox de pesquisa em nuvem — decisão sua, 07/09/2026 ("Não execute
em background, monte os scripts e eu executo na minha máquina"), depois
que uma execução em background no sandbox (Diagnóstico 18) morreu
misteriosamente no meio, reforçando que jobs longos/pesados são mais
confiáveis rodando localmente, sob seu controle direto.

Tudo aqui foi desenhado e VALIDADO em pequena escala no sandbox antes de
ser entregue (mesma disciplina do resto do projeto: nunca confiar em
código não testado) — mas a escala completa (74 instâncias × 5 seeds,
fine-tuning de 4 tamanhos) nunca rodou de ponta a ponta em lugar nenhum,
porque é exatamente o que deveria rodar pela primeira vez na sua máquina.

**Se sua máquina principal é Windows:** rode tudo isto dentro do WSL
(Windows Subsystem for Linux) — não existe uma variante deste setup para
PowerShell/cmd nativo, porque a compilação do LKH3 pressupõe um ambiente
tipo Unix (gcc/make). Se ainda não tem WSL instalado: `wsl --install` num
PowerShell como administrador, reiniciar, e depois abrir o terminal
"Ubuntu" que aparece no menu Iniciar — a partir daí é um Linux normal.

## 1. Setup (uma vez só)

```bash
bash setup.sh
```

Isso clona o repositório de Kool et al. (com os checkpoints
pré-treinados), baixa o gerador oficial de instâncias Uchoa (só para
validação, não é usado em produção), baixa e compila o LKH3, e cria um
venv Python (`venv/`) com todas as dependências. Leia a saída do script —
ele avisa se faltar `gcc`/`make` (necessários para o LKH3) e detecta se
sua máquina tem GPU NVIDIA (`nvidia-smi`) para instalar a versão certa do
PyTorch.

Depois do setup, **sempre ative o venv antes de rodar qualquer script
deste diretório**:

```bash
source venv/bin/activate
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # ver nota abaixo
```

(A variável de ambiente evita um conflito conhecido entre a versão do
protobuf instalada e o pacote `tensorboard_logger`, que o repositório do
Kool importa incondicionalmente mesmo com `--no_tensorboard` — não é bug
nosso, é um workaround documentado desse pacote. Só é necessária para os
scripts que tocam no repositório do Kool: `sondar_custo_kool.py`,
`kool_finetune_runner.py`, `avaliar_kool_finetunado.py`.)

## 2. Ordem de execução

**Passo 0 (opcional, recomendado, ~1 min): validar o gerador Uchoa.**

```bash
python3 validar_uchoa_gerador.py
```

Compara `uchoa_gerador_core.py` byte-a-byte contra o gerador oficial (10
casos cobrindo todas as combinações de posicionamento/demanda) + checagens
estruturais em 40 combinações aleatórias. Deve terminar com "TODOS OS
TESTES PASSARAM" — se não terminar, PARE e me avise antes de continuar
(algo mudou no ambiente que invalida a premissa de reprodutibilidade do
gerador).

**Passo 1 (obrigatório, rodar PRIMEIRO — tempo depende do seu hardware,
no sandbox de pesquisa levou ~3-4h para as 8 configurações do
Diagnóstico 16 nas mesmas 74 instâncias, isto aqui é só 1 configuração
então bem mais rápido):**

```bash
python3 medir_tempo_hgucb.py
```

Mede o tempo de parede do nosso próprio HG-UCB (configuração de
produção) nas 74 instâncias, 5 seeds cada. Gera
`../results/tempos_hgucb_74.csv`. **Este passo define o orçamento de
tempo que OR-Tools e LKH3 vão receber por instância** (protocolo de
tempo igualado, diário Entrada 34) — rode-o sem outros programas pesados
competindo por CPU na mesma máquina, senão o orçamento fica inflado.

**Passo 2 (depende do passo 1):**

```bash
python3 rodar_ortools_74.py
python3 rodar_lkh3_74.py
```

Cada um roda nas 74 instâncias, 5 seeds/restarts por instância, com o
orçamento de tempo lido do CSV do passo 1. Geram `../results/ortools_74.csv`
e `../results/lkh3_74.csv`. Ambos salvam incrementalmente (uma instância
por vez, com flush) — se o processo for interrompido, o que já rodou fica
salvo. Quando OR-Tools/LKH3 não acham NENHUMA solução dentro do orçamento
de tempo para uma instância/seed, isso é registrado no CSV (`viavel=False`,
com o `motivo`) em vez de travar a execução — ver diário, Entradas 46/48.

**Passo 2b (opcional, só se `rodar_lkh3_74.py` deixou alguma instância com
ZERO seeds viáveis — confira com o CSV; diário, Entradas 49/50/51):**

```bash
python3 retry_lkh3_piso_maior.py --somente NOME_DA_INSTANCIA --piso_segundos 60   # teste barato, 1 instancia
python3 retry_lkh3_piso_maior.py --piso_segundos 60                              # lote completo
```

Dá ao LKH3 um piso de tempo bem maior (60s por candidato de K, bem acima
do orçamento igualado ao HG-UCB) só para as instâncias que ficaram
totalmente sem solução — investigação no sandbox mostrou que isso resgata
ALGUMAS instâncias mas não outras (o problema não é uniforme: algumas só
precisavam de mais tempo, outras estagnam num mínimo local da formulação
do LKH3 independente do tempo dado). Leia a estimativa de custo impressa
no início (pode passar de 1-3h no lote completo) antes de confirmar, e
use `--somente` para testar 1-2 instâncias baratas primeiro. Faz backup
automático do CSV original (`../results/lkh3_74_pre_retry_AAAAMMDD_HHMMSS.csv`)
antes de alterar qualquer linha, e marca com um `motivo` diferente
("retry com piso de Ns/candidato") as linhas que deixaram de seguir o
protocolo de tempo igualado — para a análise final poder tratar isso de
forma transparente.

**Passo 3 (independente dos passos 1-2, mas o mais demorado e incerto —
faça o probe ANTES de comprometer o treino completo):**

```bash
# Para CADA tamanho (10, 20, 50, 100) -- ver aprovação do pesquisador,
# Entrada 34: "cobertura total", não só 50/100.
python3 sondar_custo_kool.py --graph_size 50
# ... olhe a estimativa impressa, decida se topa o tempo, então:
python3 kool_finetune_runner.py --graph_size 50
```

Repita para 10, 20 e 100. Cada `kool_finetune_runner.py --graph_size N`
salva checkpoints incrementais (a cada época, por padrão) em
`../results/kool_finetuned/cvrp_N/<run_name>/epoch-K.pt` — se for
interrompido, retome com:

```bash
python3 kool_finetune_runner.py --graph_size 50 --resume ../results/kool_finetuned/cvrp_50/<run_name>/epoch-K.pt
```

(troque `<run_name>` e `K` pelo diretório/época reais do último
checkpoint salvo, que aparecem no terminal quando o treino roda).

GPU: `kool_finetune_runner.py` e `sondar_custo_kool.py` detectam e usam
GPU automaticamente, sem precisar de nenhuma flag (o próprio `torch`
verifica `cuda.is_available()`) — passe `--no_cuda` só se quiser forçar
CPU mesmo tendo GPU disponível.

⚠️ **Sobre o tempo estimado:** no sandbox de pesquisa (2 CPUs, sem GPU),
1 passo de treino levou ~13,5s, o que daria ~22,5h para os 4 tamanhos ×
~1500 passos — mas essa medida quase certamente NÃO vale para a sua
máquina (núcleos, GPU, etc. são desconhecidos daqui; se sua máquina tem
GPU, como o `setup.sh` detectou na sua, espere ser bem mais rápido), por
isso o `sondar_custo_kool.py` existe: ele mede o custo real, na SUA
máquina, com as mesmas flags que o treino de verdade vai usar, em poucos
minutos, ANTES de você comprometer horas de treino baseado numa estimativa
que pode
estar bem errada. **Rode-o primeiro para cada tamanho.**

**Passo 4 (depende do passo 3 ter produzido pelo menos 1 checkpoint
fine-tunado, mas roda mesmo que só alguns tamanhos tenham sido
fine-tunados — usa o pré-treinado original como fallback para os que
faltarem, deixando isso claro no CSV):**

```bash
python3 avaliar_kool_finetunado.py
```

Roda o(s) modelo(s) fine-tunado(s) nas 74 instâncias reais (decodificação
gulosa), escolhendo para cada instância o checkpoint de tamanho mais
próximo do seu número de clientes. Gera `../results/kool_avaliacao_74.csv`.

**Leia as duas ressalvas metodológicas no topo de `avaliar_kool_finetunado.py`
antes de interpretar os números** — em especial: o modelo de Kool não tem
nenhum mecanismo para impor exatamente K rotas, então instâncias A/B
(que exigem K exato) quase certamente aparecem como "inviável" no
critério estrito mesmo quando a rota em si é boa — por isso o CSV traz
DOIS critérios de viabilidade lado a lado (`viavel_estrito` e
`viavel_sem_restricao_k`). Isso não é um bug — é uma limitação estrutural
do método que vale a pena discutir explicitamente na escrita do artigo.

## 3. O que fazer com os CSVs resultantes

Ao final, `../results/` terá:

- `tempos_hgucb_74.csv` — nosso próprio tempo/gap por instância/seed.
- `ortools_74.csv`, `lkh3_74.csv` — gap/viabilidade/tempo gasto por
  instância/seed para os dois solvers exatos/metaheurísticos.
- `kool_avaliacao_74.csv` — gap/viabilidade (dois critérios) por
  instância para o Kool fine-tunado.

Há um script de agregação em `../tests/analise_direcao1_comparativa.py`
(fora deste diretório, junto dos outros scripts de análise do projeto) —
consolida os 4 CSVs acima numa tabela comparativa + Wilcoxon/t-test
pareado contra o HG-UCB (geral e por conjunto A/B/X/CMT), tratando a
questão do K do Kool com os dois critérios (`viavel_sem_restricao_k` /
`viavel_estrito`) e excluindo (com contagem reportada, não escondida)
execuções de OR-Tools/LKH3 sem solução. Ele roda com o que já existir —
não precisa esperar os 4 CSVs completos:

```bash
python3 ../tests/analise_direcao1_comparativa.py
```

Você pode rodar isso você mesmo a qualquer momento (só precisa do
`scipy`, que já está no venv do `setup.sh`), ou me mandar os CSVs de
volta que eu rodo por aqui — qualquer um dos dois funciona.

## 4. Manifesto de arquivos

| Arquivo | O que faz | Depende de |
|---|---|---|
| `setup.sh` | Clona/baixa/compila tudo, cria o venv | — |
| `uchoa_gerador_core.py` | Gerador Uchoa reimplementado (fiel, validado byte-a-byte) | — |
| `validar_uchoa_gerador.py` | Valida o gerador contra a fonte oficial | `external/uchoa_generator/generator.py` |
| `medir_tempo_hgucb.py` | Mede tempo do HG-UCB nas 74 instâncias | `../src/*` |
| `rodar_ortools_74.py` | Roda OR-Tools nas 74 instâncias | `tempos_hgucb_74.csv` |
| `rodar_lkh3_74.py` | Roda LKH3 nas 74 instâncias | `tempos_hgucb_74.csv`, LKH3 compilado |
| `retry_lkh3_piso_maior.py` | Re-tenta com mais tempo as instâncias que o LKH3 não resolveu de jeito nenhum | `lkh3_74.csv` |
| `kool_uchoa_monkeypatch.py` | Substitui a geração de dados do Kool por Uchoa-style | `external/attention-learn-to-route` |
| `sondar_custo_kool.py` | Mede custo real de fine-tuning na sua máquina | idem |
| `kool_finetune_runner.py` | Roda o fine-tuning de verdade | idem |
| `avaliar_kool_finetunado.py` | Avalia o(s) checkpoint(s) fine-tunado(s) nas 74 instâncias | checkpoints do passo 3 |

Qualquer dúvida ou resultado inesperado, me chama antes de tentar
consertar por conta própria — em especial se `validar_uchoa_gerador.py`
falhar (passo 0), porque isso quebraria a premissa de todo o resto.
