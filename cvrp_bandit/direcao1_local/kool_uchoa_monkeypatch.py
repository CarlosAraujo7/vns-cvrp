#!/usr/bin/env python3
"""
kool_uchoa_monkeypatch.py
============================
Aplica dois monkeypatches ao repositório clonado de Kool et al. (2019)
"Attention, Learn to Solve Routing Problems!" (wouterkool/attention-learn-to-route,
MIT license), necessários para o fine-tuning nos dados estilo Uchoa (achado
real #28/#29, diario Entradas 33/37):

1) `problems.vrp.problem_vrp.VRPDataset.__init__` -- no código original
   (lido linha a linha em `external/attention-learn-to-route/problems/vrp/
   problem_vrp.py`), quando `filename is None`, os dados de treino/validação
   são gerados INLINE, uniformemente em [0,1]² com demanda uniforme 1-9 e
   capacidade FIXA por tamanho (`CAPACITIES = {10:20,20:30,50:40,100:50}`).
   O parâmetro `distribution` existe na assinatura mas NÃO é usado em
   lugar nenhum do corpo -- ou seja, `--data_distribution` do `options.py`
   não tem nenhum efeito real sobre o VRP (só serve de gancho para outros
   problemas do mesmo repositório). Não há, portanto, nenhum jeito de
   injetar uma distribuição diferente sem substituir este método
   diretamente -- confirmado lendo `run.py`/`train.py` por completo: o
   dataset de treino é recriado do zero A CADA ÉPOCA (`train.py` linha 75,
   comentário "Generate new training data for each epoch"), então o
   patch abaixo já entrega "sempre-novo, em processo" (decisão do
   pesquisador, Entrada 33) automaticamente, sem precisar de nenhuma
   lógica extra de cache/refresh.

   O patch gera cada instância com `gerar_instancia_uchoa` (fiel ao
   gerador oficial do conjunto X, validado byte-a-byte em
   `validar_uchoa_gerador.py`), sorteando a combinação de atributos
   (posição do depósito/clientes, tipo de demanda, tamanho de rota) via
   `amostrar_combinacao` -- as mesmas 378 combinações do espaço XML100
   (Queiroga et al. 2022). Diferença importante em relação ao Kool
   original: aqui a CAPACIDADE é a do próprio Uchoa (varia por instância,
   dependendo de route_size/demandas), não uma constante fixa por
   tamanho -- decisão deliberada (ver docstring de `aplicar_monkeypatches`
   abaixo) porque as 74 instâncias reais de avaliação (A/B/X/CMT) também
   têm capacidades bem variadas e não relacionadas ao dicionário
   `CAPACITIES` do Kool original.

   ATENÇÃO (achado real #38, diário Entrada 53): para `graph_size=10`
   especificamente, ~7% das combinações sorteadas são INFACTÍVEIS para
   este tamanho (via `cust_pos==3`, o nº de seeds sorteado por
   `gerar_instancia_uchoa`, 2-6, pode exceder o nº de clientes reservados
   para clusterização, que é só 5 quando n=10) -- `gerar_instancia_uchoa`
   rejeita corretamente essas combinações com `ValueError` (comportamento
   fiel ao gerador oficial, que nunca precisou lidar com n tão pequeno).
   `_gera_um_exemplo_uchoa` abaixo trata isso como rejection sampling
   padrão: sorteia outra combinação em vez de propagar o erro. Medido:
   0% de impacto em graph_size 20/50/100 (nunca acontece nesses tamanhos).

2) `utils.functions.torch_load_cpu`, apenas como é referenciado dentro do
   módulo `run` (`run.py` faz `from utils import torch_load_cpu`, então o
   nome já fica vinculado no namespace de `run` no momento do import --
   por isso o patch abaixo substitui `run.torch_load_cpu`, não
   `utils.functions.torch_load_cpu`, que não teria efeito nenhum sobre
   uma referência já importada). Esse segundo patch remove a chave
   'baseline' do checkpoint pré-treinado ao carregar via `--load_path`,
   SE E SOMENTE SE o `--baseline` escolhido para o fine-tuning for
   diferente de 'rollout' -- necessário porque o checkpoint pré-treinado
   do Kool foi originalmente treinado com baseline='rollout' (confirmado
   em `pretrained/cvrp_50/args.json`), e `RolloutBaseline.load_state_dict`
   espera um dicionário com chaves `{'model','dataset','epoch'}`
   completamente diferente do que `ExponentialBaseline.load_state_dict`
   espera (`{'v'}`) -- sem este patch, rodar com `--baseline exponential`
   (o padrão recomendado por `kool_finetune_runner.py`, ver lá o porquê)
   e `--load_path` ao mesmo tempo QUEBRA com KeyError. Pesos do modelo e
   estado do otimizador continuam sendo carregados normalmente -- só o
   estado do baseline antigo (que não faria sentido reaproveitar com um
   mecanismo de baseline diferente) é descartado.

Uso: chamar `aplicar_monkeypatches(seed=...)` UMA VEZ, logo após importar
o repositório do Kool (mas antes de chamar `run.run(...)`), a partir de
`kool_finetune_runner.py` ou `avaliar_kool_finetunado.py`.
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uchoa_gerador_core import gerar_instancia_uchoa, amostrar_combinacao  # noqa: E402


_MAX_TENTATIVAS_COMBINACAO = 20


def _gera_um_exemplo_uchoa(size, rng, max_tentativas=_MAX_TENTATIVAS_COMBINACAO):
    """Gera 1 instância Uchoa-style e converte para o formato de tensores
    exigido por `problems/vrp/problem_vrp.py` (loc/demand/depot, coords em
    [0,1], demanda como fração da capacidade DA PRÓPRIA instância).

    Algumas combinações sorteadas por `amostrar_combinacao` são infactíveis
    para tamanhos pequenos -- na prática só `graph_size=10` via `cust_pos==3`
    (achado real #38, diário Entrada 53; validado: 0% de ocorrência em
    20/50/100). Em vez de propagar o `ValueError` de `gerar_instancia_uchoa`,
    tratamos como rejection sampling padrão: sorteamos outra combinação.
    `max_tentativas` é só uma rede de segurança (taxa de falha medida ~7%
    por tentativa em n=10 -- 20 tentativas consecutivas falhando seria
    ~10^-24, praticamente impossível; se acontecer, é sinal de que algo
    mudou de verdade em `uchoa_gerador_core.py` e merece investigação, não
    um retry maior)."""
    for _ in range(max_tentativas):
        dp, cp, dt, rs = amostrar_combinacao(rng)
        try:
            depot_xy, clientes_xy, demandas, capacidade = gerar_instancia_uchoa(
                size, dp, cp, dt, rs, rng=rng)
        except ValueError as e:
            if "clusters" not in str(e):
                raise
            continue
        return {
            'loc': torch.tensor(clientes_xy, dtype=torch.float) / 1000.0,
            'demand': torch.tensor(demandas, dtype=torch.float) / float(capacidade),
            'depot': torch.tensor(depot_xy, dtype=torch.float) / 1000.0,
        }
    raise RuntimeError(
        f"_gera_um_exemplo_uchoa: {max_tentativas} tentativas consecutivas infactiveis "
        f"para size={size} -- taxa de falha muito acima do esperado (~7% em n=10, 0% em "
        f"20/50/100), verifique se uchoa_gerador_core.py mudou.")


def _vrpdataset_init_uchoa(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
    """Substitui `VRPDataset.__init__` original -- mesma assinatura exata
    (inclusive o `distribution` não utilizado, mantido só para não quebrar
    as chamadas existentes em run.py/train.py/reinforce_baselines.py que
    sempre passam esse kwarg)."""
    import torch.utils.data as _tud
    _tud.Dataset.__init__(self)

    if filename is not None:
        # Mesmo comportamento original: carregar de um .pkl pré-gerado
        # (usado, por ex., se o pesquisador quiser fixar um --val_dataset
        # específico em vez de gerar on-the-fly). Reimportado aqui para
        # não duplicar lógica.
        import os
        import pickle
        from problems.vrp.problem_vrp import make_instance
        assert os.path.splitext(filename)[1] == '.pkl'
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.data = [make_instance(args) for args in data[offset:offset + num_samples]]
    else:
        self.data = [_gera_um_exemplo_uchoa(size, random) for _ in range(num_samples)]

    self.size = len(self.data)


def aplicar_monkeypatches(seed=None, manter_baseline_rollout=False):
    """Aplica os dois patches descritos na docstring do módulo. Deve ser
    chamada DEPOIS que `sys.path` já inclui a raiz do repositório do Kool
    (attention-learn-to-route/) e ANTES de chamar `run.run(opts)`.

    seed: se dado, semeia o módulo global `random` (usado internamente
    pelo gerador Uchoa) -- reprodutibilidade do EXPERIMENTO COMO UM TODO
    (mesma sequência de instâncias vistas ao longo do treino), mas como o
    estado do `random` global segue avançando a cada chamada (nunca é
    re-semeado por época), cada época ainda recebe dados novos --
    exatamente o comportamento "sempre-novo, em processo" aprovado.

    manter_baseline_rollout: passar True SE `--baseline rollout` for
    usado (o padrão do checkpoint pré-treinado) -- nesse caso o patch (2)
    é pulado, porque o estado salvo É compatível com RolloutBaseline.
    Deixar False (padrão) quando `--baseline` for 'exponential', 'critic'
    ou None.
    """
    if seed is not None:
        random.seed(seed)

    from problems.vrp import problem_vrp
    problem_vrp.VRPDataset.__init__ = _vrpdataset_init_uchoa

    if not manter_baseline_rollout:
        import run as run_module
        from utils.functions import torch_load_cpu as _torch_load_cpu_original

        def _torch_load_cpu_sem_baseline(load_path):
            dados = _torch_load_cpu_original(load_path)
            if isinstance(dados, dict) and "baseline" in dados:
                dados = dict(dados)
                del dados["baseline"]
                print(f"  [aplicar_monkeypatches] descartando estado de baseline antigo "
                      f"(incompatível com o baseline escolhido para o fine-tuning) de {load_path}")
            return dados

        run_module.torch_load_cpu = _torch_load_cpu_sem_baseline

    print(f"  [aplicar_monkeypatches] VRPDataset.__init__ substituído por geração Uchoa-style "
          f"(seed={seed}, baseline_rollout_mantido={manter_baseline_rollout}).")
