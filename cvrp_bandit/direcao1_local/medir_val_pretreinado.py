#!/usr/bin/env python3
"""
medir_val_pretreinado.py
=========================
Fecha a lacuna (ii) do achado #40 (diário, Entrada 61): o custo de validação
do checkpoint PRÉ-TREINADO do Kool et al. (época "-1") nunca tinha sido
medido no conjunto de validação do fine-tuning (200 instâncias estilo
Uchoa, seed 2026), então não se sabia se um custo de validação de ~19,2 era
melhora sobre o modelo original ou só ausência de degradação.

Reconstrói o conjunto de validação do MESMO modo que `run.py` o cria
(random.seed(2026) -> VRPDataset gerado pelo patch Uchoa -> 200 amostras) e
avalia, com `train.rollout` (decodificação gulosa, eval mode):
  - o checkpoint pré-treinado original (epoch-99.pt)
  - as épocas 0 e 14 do fine-tuning ORIGINAL (lr 1e-4) -- CONTROLE do
    harness: os valores têm que bater com os `Validation overall avg_cost`
    já registrados em `log_finetune_{n}.txt`. Se não baterem, o conjunto de
    validação aqui NÃO é o mesmo do treino e a medição do pré-treinado não
    vale.

Uso (no WSL, venv ativado): python medir_val_pretreinado.py [--cuda]
"""
import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
KOOL_DIR = ROOT / "external" / "attention-learn-to-route"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(KOOL_DIR))

SEED = 2026
VAL_SIZE = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

    from kool_uchoa_monkeypatch import aplicar_monkeypatches
    from avaliar_kool_finetunado import carregar_modelo
    from utils.functions import load_problem
    from train import rollout

    problem = load_problem("cvrp")
    opts = SimpleNamespace(device=device, eval_batch_size=200, no_progress_bar=True)

    print(f"{'n':>4} {'checkpoint':<44} {'val_avg_cost':>13}")
    for n in (10, 20, 50, 100):
        # patch de dataset (baseline_rollout=True evita mexer em torch_load_cpu, irrelevante aqui)
        aplicar_monkeypatches(seed=SEED, manter_baseline_rollout=True)
        random.seed(SEED)
        torch.manual_seed(SEED)
        val = problem.make_dataset(size=n, num_samples=VAL_SIZE, filename=None, distribution=None)

        base = ROOT / "results" / "kool_finetuned" / f"cvrp_{n}"
        run_dirs = sorted(d for d in base.glob("uchoa_ft_*") if d.is_dir())
        alvos = [("pre-treinado (epoch-99)", KOOL_DIR / "pretrained" / f"cvrp_{n}" / "epoch-99.pt")]
        if run_dirs:
            alvos += [(f"ft lr1e-4 epoca 0 [controle]", run_dirs[-1] / "epoch-0.pt"),
                      (f"ft lr1e-4 epoca 14", run_dirs[-1] / "epoch-14.pt")]
        for rotulo, ckpt in alvos:
            modelo = carregar_modelo(ckpt, device)
            custos = rollout(modelo, val, opts)
            print(f"{n:>4} {rotulo:<44} {custos.mean().item():>13.4f}", flush=True)


if __name__ == "__main__":
    main()
