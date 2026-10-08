#!/usr/bin/env python3
"""
sondar_custo_kool.py
=======================
Mede o custo REAL (segundos/passo de treino, na SUA máquina) do
fine-tuning de Kool et al. antes de comprometer o orçamento completo de
~1500 passos/tamanho (decisão aprovada em 07/09/2026). Roda 1 época
pequena (poucos passos) com exatamente as mesmas flags que
`kool_finetune_runner.py` usaria por padrão (mesmo baseline, mesmo
batch_size), e extrapola para o orçamento completo.

Por que isto existe (acha real #30, diário Entrada 40): no sandbox de
pesquisa usado para desenhar este pipeline (2 CPUs, sem GPU), 1 passo de
treino (graph_size=50, batch_size=32) levou ~13,5s -- mas essa medida NÃO
tem nenhuma razão para valer no seu hardware (nº de núcleos, presença de
GPU, etc. são desconhecidos daqui). Rodar este probe primeiro custa
minutos; descobrir a estimativa errada só depois de já ter comprometido
horas de treino custa muito mais.

GPU: detectada automaticamente -- não precisa passar nenhuma flag para
usá-la se existir. Passe `--no_cuda` para forçar CPU mesmo com GPU
disponível.

Uso:
    python sondar_custo_kool.py --graph_size 50
    python sondar_custo_kool.py --graph_size 100 --baseline rollout  # se for usar rollout
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


def main():
    parser = argparse.ArgumentParser(description="Sonda o custo real do fine-tuning de Kool et al. na sua máquina.")
    parser.add_argument("--graph_size", type=int, required=True, choices=TAMANHOS_VALIDOS)
    parser.add_argument("--baseline", default="exponential", choices=("exponential", "rollout", "none"),
                        help="Mesmo baseline que você pretende usar no fine-tuning real (padrão do "
                             "kool_finetune_runner.py é 'exponential' -- ver docstring de lá para o porquê).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--passos_sonda", type=int, default=4,
                        help="Quantos passos de treino medir na sonda (padrão 4 -- rápido e já dá uma "
                             "média razoável).")
    parser.add_argument("--n_epochs_alvo", type=int, default=15, help="n_epochs que você pretende usar de fato.")
    parser.add_argument("--epoch_size_alvo", type=int, default=3200, help="epoch_size que você pretende usar de fato.")
    parser.add_argument("--no_cuda", action="store_true", help="Forçar CPU mesmo com GPU disponível.")
    args = parser.parse_args()

    checkpoint = KOOL_DIR / "pretrained" / f"cvrp_{args.graph_size}" / "epoch-99.pt"
    if not checkpoint.exists():
        print(f"ERRO: {checkpoint} não existe -- rode setup.sh primeiro.")
        sys.exit(1)

    epoch_size_sonda = args.passos_sonda * args.batch_size
    baseline_opt = None if args.baseline == "none" else args.baseline

    args_finais = [
        "--problem", "cvrp", "--graph_size", str(args.graph_size),
        "--load_path", str(checkpoint),
        "--n_epochs", "1", "--epoch_size", str(epoch_size_sonda), "--batch_size", str(args.batch_size),
        "--val_size", str(args.batch_size), "--eval_batch_size", str(args.batch_size),
        "--checkpoint_epochs", "0",
        "--no_tensorboard", "--no_progress_bar",
        "--output_dir", "/tmp/sonda_kool_output", "--run_name", "sonda",
    ]
    if baseline_opt is not None:
        args_finais += ["--baseline", baseline_opt]
    if args.no_cuda:
        args_finais += ["--no_cuda"]

    vai_usar_gpu = torch.cuda.is_available() and not args.no_cuda
    print(f"=== Sondando custo real: graph_size={args.graph_size} baseline={args.baseline} "
          f"batch_size={args.batch_size} ({args.passos_sonda} passos de treino nesta sonda) ===")
    print(f"  GPU: {'usando ' + torch.cuda.get_device_name(0) if vai_usar_gpu else 'CPU (nenhuma GPU disponível ou --no_cuda passado)'}\n")

    from kool_uchoa_monkeypatch import aplicar_monkeypatches
    aplicar_monkeypatches(seed=1, manter_baseline_rollout=(args.baseline == "rollout"))

    from options import get_options
    from run import run

    opts = get_options(args_finais)
    t0 = time.perf_counter()
    run(opts)
    dt_total = time.perf_counter() - t0

    passos_totais_alvo = args.n_epochs_alvo * (args.epoch_size_alvo // args.batch_size)
    epocas_sonda_equiv = epoch_size_sonda / args.epoch_size_alvo  # fração de 1 epoca "alvo" que a sonda cobriu

    # dt_total inclui overhead fixo (carregar checkpoint, criar val_dataset, criar baseline,
    # salvar/validar) que NÃO escala linearmente com passos -- por isso reportamos os dois
    # números (bruto da sonda e um custo/passo médio) em vez de só extrapolar ingenuamente.
    custo_por_passo = dt_total / args.passos_sonda
    estimativa_ingenua_s = custo_por_passo * passos_totais_alvo

    print(f"\n=== Resultado da sonda ===")
    print(f"  Tempo total da sonda (1 época mínima, {args.passos_sonda} passos + overhead "
          f"de carregar/validar/salvar): {dt_total:.1f}s")
    print(f"  Custo médio por passo (grosseiro, overhead fixo incluído/diluído): {custo_por_passo:.2f}s/passo")
    print(f"\n  Orçamento alvo: {args.n_epochs_alvo} épocas x "
          f"{args.epoch_size_alvo // args.batch_size} passos/época = {passos_totais_alvo} passos totais")
    print(f"  Estimativa (linear, ORTODOX -- provavelmente um TETO superior, já que overhead "
          f"fixo por época não escala com nº de passos): "
          f"~{estimativa_ingenua_s:.0f}s (~{estimativa_ingenua_s/60:.1f}min, ~{estimativa_ingenua_s/3600:.2f}h)")
    if args.baseline == "rollout":
        print(f"\n  ATENÇÃO (baseline=rollout): o overhead de rollout (wrap_dataset + validate + "
              f"epoch_callback) escala com epoch_size/val_size, não com batch_size/nº de passos -- "
              f"se --epoch_size_alvo/--val_size no treino real forem diferentes desta sonda, rode "
              f"a sonda de novo com os MESMOS valores para uma estimativa mais fiel.")
    print(f"\n  Recomendação: rode esta sonda com --epoch_size_alvo/--n_epochs_alvo IGUAIS ao que "
          f"pretende usar de fato em kool_finetune_runner.py, para o overhead fixo por época "
          f"(que não é desprezível em épocas curtas) ficar corretamente representado.")


if __name__ == "__main__":
    main()
