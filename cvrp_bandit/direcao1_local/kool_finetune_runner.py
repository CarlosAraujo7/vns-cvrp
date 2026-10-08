#!/usr/bin/env python3
"""
kool_finetune_runner.py
==========================
Wrapper de linha de comando para fine-tunar um checkpoint pré-treinado de
Kool et al. (2019) nos dados sintéticos estilo Uchoa (Direção 1, diário
Entradas 33/34/37 -- decisões do pesquisador via `AskUserQuestion`:
fine-tuning nos dados sintéticos, gerador estilo Uchoa gerado sempre-novo
em processo, tamanhos 10/20/50/100 com cobertura total, ~1500 passos por
tamanho).

RODAR ISTO NA SUA MÁQUINA, NÃO no sandbox de pesquisa -- ver README.md
deste diretório para o porquê (pivô decidido em 07/09/2026: toda execução
pesada da Direção 1 passa a rodar aqui, não mais no ambiente de nuvem).

------------------------------------------------------------------------
ANTES DE RODAR O TREINO COMPLETO: rode `sondar_custo_kool.py` primeiro,
na SUA máquina, para medir o custo real por passo/por época -- a
estimativa feita no sandbox de pesquisa (~13,5s/passo, 2 CPUs, sem GPU)
NÃO necessariamente vale para o seu hardware, e o probe honesto
(`sondar_custo_kool.py`) é rápido (poucos minutos) comparado ao custo de
descobrir a estimativa errada só depois de já ter comprometido horas de
treino.
------------------------------------------------------------------------

Por que `--baseline exponential` por padrão (diferente do checkpoint
original, que foi treinado com `--baseline rollout`, ver
`pretrained/cvrp_{N}/args.json`): medido empiricamente no sandbox de
pesquisa (achado real #31 -- ver diário) que o baseline "rollout" faz,
adicionalmente aos passos de treino em si, um rollout guloso completo do
tamanho de `--epoch_size` A CADA ÉPOCA (dentro de `train_epoch`, para dar
baseline a cada instância do batch), MAIS uma reavaliação candidata a
cada `epoch_callback` -- um multiplicador de custo não-trivial e difícil
de estimar sem medir no seu hardware. Para FINE-TUNING (adaptar um modelo
já convergido a uma nova distribução, não treinar do zero), um baseline
mais barato (exponential moving average, sem nenhum rollout extra) é uma
escolha padrão razoável na literatura e maximiza quantos passos de
gradiente de fato cabem no orçamento de tempo aprovado (~1500
passos/tamanho). Quem tiver GPU ou tempo de sobra pode passar
`--baseline rollout` explicitamente (o script cuida de manter/descartar
o estado salvo do baseline antigo corretamente em qualquer um dos casos
-- ver `kool_uchoa_monkeypatch.aplicar_monkeypatches`).

GPU: detectada automaticamente (via `torch.cuda.is_available()`, igual ao
próprio `options.py` do Kool et al.) -- não precisa passar nenhuma flag
para usá-la se existir. Para forçar CPU mesmo com GPU disponível (ex.:
para manter a GPU livre para outra coisa, ou para uma comparação
controlada), passe `--no_cuda` (repassado direto ao Kool et al.).

Exemplos de uso:
    python kool_finetune_runner.py --graph_size 50
    python kool_finetune_runner.py --graph_size 100 --n_epochs 20 --epoch_size 6400
    python kool_finetune_runner.py --graph_size 20 --no_cuda   # forçar CPU mesmo tendo GPU
    # Retomar um treino interrompido (em vez de --load_path, usar --resume
    # apontando para o último checkpoint salvo por ESTE script):
    python kool_finetune_runner.py --graph_size 50 --resume outputs_uchoa/cvrp_50/uchoa_ft_50_.../epoch-6.pt

Uso: python kool_finetune_runner.py --graph_size {10,20,50,100} [demais
flags do Kool et al. passados adiante -- ver external/attention-learn-to-route/options.py]
"""
import argparse
import sys
import time
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
KOOL_DIR = ROOT / "external" / "attention-learn-to-route"

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(KOOL_DIR))

TAMANHOS_VALIDOS = (10, 20, 50, 100)

# ~1500 passos/tamanho (aprovado via AskUserQuestion, 07/09/2026):
# 15 epocas x (3200/32=100 passos/epoca) = 1500 passos.
DEFAULTS_TREINO = {
    "n_epochs": "15",
    "epoch_size": "3200",
    "batch_size": "32",
    "val_size": "200",
    "eval_batch_size": "200",
    "checkpoint_epochs": "1",
    "baseline": "exponential",
    "seed": "2026",
}


def _valor_final(tokens, flag, default):
    """Varre uma lista de tokens de CLI (ex.: ['--baseline','rollout',...])
    e retorna o valor da ÚLTIMA ocorrência de --flag (mesma semântica do
    argparse ao receber a flag repetida), ou `default` se ausente."""
    valor = default
    for i, tok in enumerate(tokens):
        if tok == f"--{flag}" and i + 1 < len(tokens):
            valor = tokens[i + 1]
    return valor


def _flag_presente(tokens, flag):
    return f"--{flag}" in tokens


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning de Kool et al. (2019) em dados sintéticos estilo Uchoa.",
        add_help=True)
    parser.add_argument("--graph_size", type=int, required=True, choices=TAMANHOS_VALIDOS)
    args_conhecidos, args_extra = parser.parse_known_args()

    checkpoint_pretreinado = KOOL_DIR / "pretrained" / f"cvrp_{args_conhecidos.graph_size}" / "epoch-99.pt"
    if not checkpoint_pretreinado.exists():
        print(f"ERRO: checkpoint pré-treinado não encontrado em {checkpoint_pretreinado}\n"
              f"Rode setup.sh primeiro (ele clona o repositório do Kool et al., que já "
              f"inclui os checkpoints pré-treinados em pretrained/cvrp_{{10,20,50,100}}/).")
        sys.exit(1)

    usa_resume = _flag_presente(args_extra, "resume")

    args_finais = ["--problem", "cvrp", "--graph_size", str(args_conhecidos.graph_size)]
    for flag, valor_padrao in DEFAULTS_TREINO.items():
        args_finais += [f"--{flag}", valor_padrao]
    if not usa_resume:
        args_finais += ["--load_path", str(checkpoint_pretreinado)]
    args_finais += ["--no_tensorboard", "--no_progress_bar"]
    args_finais += ["--output_dir", str(ROOT / "results" / "kool_finetuned"),
                    "--run_name", f"uchoa_ft_{args_conhecidos.graph_size}"]
    args_finais += args_extra  # overrides do usuário sempre por último => sempre vencem

    baseline_final = _valor_final(args_finais, "baseline", "exponential")
    seed_final = int(_valor_final(args_finais, "seed", DEFAULTS_TREINO["seed"]))

    vai_usar_gpu = torch.cuda.is_available() and "--no_cuda" not in args_finais
    print("=== kool_finetune_runner: fine-tuning Kool et al. em dados sintéticos estilo Uchoa ===")
    print(f"  graph_size={args_conhecidos.graph_size}  baseline={baseline_final}  "
          f"resume={'sim' if usa_resume else 'não (carregando ' + str(checkpoint_pretreinado.name) + ')'}")
    print(f"  GPU: {'usando ' + torch.cuda.get_device_name(0) if vai_usar_gpu else 'CPU (nenhuma GPU disponível ou --no_cuda passado)'}")
    print(f"  args finais passados ao Kool et al.: {' '.join(args_finais)}\n")

    from kool_uchoa_monkeypatch import aplicar_monkeypatches
    aplicar_monkeypatches(seed=seed_final, manter_baseline_rollout=(baseline_final == "rollout"))

    from options import get_options
    from run import run

    opts = get_options(args_finais)
    t0 = time.perf_counter()
    run(opts)
    dt = time.perf_counter() - t0
    print(f"\n=== fine-tuning concluído em {dt:.1f}s ({dt/60:.1f}min) -- "
          f"checkpoints em {opts.save_dir} ===")


if __name__ == "__main__":
    main()
