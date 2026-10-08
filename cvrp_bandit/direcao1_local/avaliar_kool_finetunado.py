#!/usr/bin/env python3
"""
avaliar_kool_finetunado.py
=============================
Roda o(s) checkpoint(s) fine-tunados de Kool et al. (produzidos por
`kool_finetune_runner.py`) nas 74 instâncias REAIS do benchmark
(A=26/B=23/X=21/CMT=4), decodificação gulosa (greedy -- mesma que
`train.py:rollout` usa para validação, e a mais comum para *reportar*
resultados de inference no paper original), e salva um CSV com o
resultado -- mesmo formato geral dos demais resultados da Direção 1
(`rodar_ortools_74.py`, `rodar_lkh3_74.py`), para permitir comparação
direta.

DUAS RESSALVAS METODOLÓGICAS IMPORTANTES (leia antes de interpretar os
números -- não são bugs, são limitações estruturais do método):

1) NORMALIZAÇÃO DE COORDENADAS: o modelo foi treinado (e fine-tunado) com
   coordenadas em [0,1]². As 74 instâncias reais têm escalas bem
   diferentes (CMT usa coordenadas pequenas tipo 0-100; X/A/B chegam a
   0-1000) -- por isso cada instância é normalizada PELA PRÓPRIA caixa
   delimitadora (bounding box), preservando a proporção largura/altura
   (um único fator de escala = 1/max(largura,altura), não um fator por
   eixo, para não distorcer distâncias relativas). É a abordagem padrão
   para aplicar um modelo treinado em [0,1]² a instâncias reais de escala
   desconhecida -- mas ainda assim é uma transferência de domínio (o
   modelo nunca viu ESSA distribuição espacial exata durante o
   treino/fine-tuning), então alguma perda de qualidade em relação ao
   desempenho "nativo" do modelo é esperada e não indica erro de código.

2) NÚMERO DE VEÍCULOS (K): o modelo não tem NENHUM mecanismo para impor
   "exatamente K rotas" -- ele só aprende a minimizar distância total
   respeitando capacidade e cobertura completa, decidindo quantas vezes
   volta ao depósito livremente (mesma formulação "K não fixo" do próprio
   paper original, que亦 é a convenção nativa dos conjuntos X/CMT deste
   projeto). Para A/B (`vehicles_exact=True`, que EXIGE exatamente K
   rotas), isso quase certamente reprova no `inst.is_feasible()` estrito
   deste projeto -- não por a rota ser ruim, mas por usar um número de
   rotas diferente do exigido. Por isso este script reporta DOIS
   critérios de viabilidade por instância: `viavel_estrito` (o
   `is_feasible()` de sempre, com a checagem de K) e
   `viavel_sem_restricao_k` (só capacidade + cobertura completa, ignorando
   K) -- o segundo é o que de fato mede "o modelo achou uma rota
   correta", o primeiro é o que bate com o critério de aceitação usado
   para os outros solvers desta Direção 1. Ambos são reportados lado a
   lado no CSV; a decisão de como apresentar isso no artigo (ex.: marcar
   a coluna do Kool com uma nota em A/B) fica para depois de ter os
   números reais em mãos.

O custo relatado (coluna `custo`) SEMPRE vem de `inst.solution_cost()`
(distância real, não normalizada, não arredondada para CMT) -- NUNCA do
`cost` retornado pelo próprio modelo (que está em espaço normalizado
[0,1], sem significado direto de distância real) -- mesma disciplina já
aplicada a OR-Tools/LKH3 (ver diário, Entradas 35-36).

Uso:
    python avaliar_kool_finetunado.py
    python avaliar_kool_finetunado.py --checkpoint_dir results/kool_finetuned
    python avaliar_kool_finetunado.py --forcar_tamanho 50   # usa só o checkpoint de 1 tamanho p/ tudo
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
KOOL_DIR = ROOT / "external" / "attention-learn-to-route"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(KOOL_DIR))

TAMANHOS_VALIDOS = (10, 20, 50, 100)
CONJUNTOS = ("A", "B", "X", "CMT")


def descobrir_checkpoints(checkpoint_root):
    """Para cada tamanho (10/20/50/100), acha o checkpoint fine-tunado
    mais recente (maior epoch-N.pt, na run mais recente por nome de
    diretório) dentro de checkpoint_root/cvrp_{tamanho}/*/. Tamanhos sem
    nenhum checkpoint fine-tunado são omitidos (caem de volta pro
    pré-treinado original em `escolher_checkpoint`)."""
    achados = {}
    for tam in TAMANHOS_VALIDOS:
        base = Path(checkpoint_root) / f"cvrp_{tam}"
        if not base.exists():
            continue
        candidatos = sorted(base.glob("*/epoch-*.pt"))
        if not candidatos:
            continue

        def chave(p):
            try:
                n_epoch = int(p.stem.split("-")[1])
            except (IndexError, ValueError):
                n_epoch = -1
            return (p.parent.name, n_epoch)

        achados[tam] = max(candidatos, key=chave)
    return achados


def escolher_checkpoint(n_customers, checkpoints_finetunados, forcar_tamanho=None):
    if forcar_tamanho is not None:
        tam = forcar_tamanho
    else:
        tam = min(TAMANHOS_VALIDOS, key=lambda t: abs(t - n_customers))
    if tam in checkpoints_finetunados:
        return tam, checkpoints_finetunados[tam], True
    return tam, KOOL_DIR / "pretrained" / f"cvrp_{tam}" / "epoch-99.pt", False


def carregar_modelo(checkpoint_path, device):
    import json
    from nets.attention_model import AttentionModel
    from utils.functions import load_problem, torch_load_cpu

    args_json = Path(checkpoint_path).parent / "args.json"
    if args_json.exists():
        with open(args_json) as f:
            train_args = json.load(f)
    else:
        # Checkpoint pré-treinado original do Kool -- usa a arquitetura padrão
        # documentada em options.py (mesma usada para treinar TODOS os
        # checkpoints publicados em pretrained/, confirmado em
        # pretrained/cvrp_50/args.json).
        train_args = {"embedding_dim": 128, "hidden_dim": 128, "n_encode_layers": 3,
                      "normalization": "batch", "tanh_clipping": 10.0}

    problem = load_problem("cvrp")
    model = AttentionModel(
        train_args["embedding_dim"], train_args["hidden_dim"], problem,
        n_encode_layers=train_args["n_encode_layers"], mask_inner=True, mask_logits=True,
        normalization=train_args["normalization"], tanh_clipping=train_args["tanh_clipping"],
    ).to(device)

    dados = torch_load_cpu(checkpoint_path)
    model_ = model
    model_.load_state_dict({**model_.state_dict(), **dados.get("model", {})})
    model.eval()
    from nets.attention_model import set_decode_type
    set_decode_type(model, "greedy")
    return model


def instancia_para_tensores(inst, device):
    """Normaliza pela própria caixa delimitadora (preserva proporção --
    ver ressalva (1) na docstring do módulo) e monta o dict de tensores
    com batch_size=1 esperado por AttentionModel.forward."""
    clientes_ids = inst.customers  # ordem fixa: define o mapeamento indice-Kool <-> id-VRP
    todos_xy = [inst.coords[inst.depot_id]] + [inst.coords[c] for c in clientes_ids]
    xs = [p[0] for p in todos_xy]
    ys = [p[1] for p in todos_xy]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    escala = 1.0 / max(max_x - min_x, max_y - min_y, 1e-9)

    def norm(p):
        return ((p[0] - min_x) * escala, (p[1] - min_y) * escala)

    depot_xy = norm(inst.coords[inst.depot_id])
    loc_xy = [norm(inst.coords[c]) for c in clientes_ids]
    demandas = [inst.demands[c] / float(inst.capacity) for c in clientes_ids]

    entrada = {
        "loc": torch.tensor([loc_xy], dtype=torch.float, device=device),
        "demand": torch.tensor([demandas], dtype=torch.float, device=device),
        "depot": torch.tensor([depot_xy], dtype=torch.float, device=device),
    }
    return entrada, clientes_ids


def pi_para_rotas(pi_tensor, clientes_ids):
    """pi (1D, batch já removido): sequência de ações em {0,...,n_loc},
    0=depósito (separador de rota, nunca no início nem consecutivo -- ver
    `state_cvrp.py:get_mask`), 1..n_loc=cliente (índice 1-based na ordem
    de `clientes_ids`). Retorna lista de rotas em IDs REAIS do .vrp."""
    rotas, atual = [], []
    for a in pi_tensor.tolist():
        if a == 0:
            if atual:
                rotas.append(atual)
                atual = []
        else:
            atual.append(clientes_ids[a - 1])
    if atual:
        rotas.append(atual)
    return rotas


def viavel_sem_restricao_k(inst, routes):
    """Mesma checagem de `inst.is_feasible`, mas SEM a restrição de K --
    só capacidade + cobertura exata (ver ressalva (2) da docstring)."""
    routes_nao_vazias = [r for r in routes if len(r) > 0]
    seen = []
    for r in routes_nao_vazias:
        if inst.route_demand(r) > inst.capacity + 1e-6:
            return False, "rota excede capacidade"
        seen.extend(r)
    if sorted(seen) != sorted(inst.customers):
        return False, "cobertura invalida"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default=str(ROOT / "results" / "kool_finetuned"))
    parser.add_argument("--forcar_tamanho", type=int, default=None, choices=TAMANHOS_VALIDOS,
                        help="Se dado, usa o checkpoint DESTE tamanho para TODAS as instâncias "
                             "(em vez de escolher o mais próximo do n_customers de cada uma).")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--zero_shot", action="store_true",
                        help="Ignora TODOS os checkpoints fine-tunados e usa só os pré-treinados "
                             "originais do Kool (diário, Entrada 62: braço Kool como contexto "
                             "zero-shot). Muda o CSV de saída padrão para kool_zeroshot_74.csv.")
    parser.add_argument("--out_csv", default=None,
                        help="Caminho do CSV de saída (padrão: results/kool_avaliacao_74.csv, ou "
                             "results/kool_zeroshot_74.csv com --zero_shot).")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from instance import load_all  # noqa: E402

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    checkpoints_ft = {} if args.zero_shot else descobrir_checkpoints(args.checkpoint_dir)
    print(f"=== Avaliação do(s) modelo(s) Kool fine-tunado(s) nas 74 instâncias reais ===")
    for tam in TAMANHOS_VALIDOS:
        origem = str(checkpoints_ft.get(tam, KOOL_DIR / 'pretrained' / f'cvrp_{tam}' / 'epoch-99.pt'))
        tipo = "FINE-TUNADO" if tam in checkpoints_ft else "pré-treinado original (sem fine-tuning encontrado)"
        print(f"  tamanho {tam:3d}: {tipo} -- {origem}")
    print()

    modelos_cache = {}

    def obter_modelo(checkpoint_path):
        chave = str(checkpoint_path)
        if chave not in modelos_cache:
            modelos_cache[chave] = carregar_modelo(checkpoint_path, device)
        return modelos_cache[chave]

    linhas_csv = []
    for set_name in CONJUNTOS:
        instancias = load_all(ROOT / "instances" / set_name)
        for inst in instancias:
            tam_escolhido, ckpt_path, era_finetunado = escolher_checkpoint(
                inst.n_customers, checkpoints_ft, forcar_tamanho=args.forcar_tamanho)
            modelo = obter_modelo(ckpt_path)

            entrada, clientes_ids = instancia_para_tensores(inst, device)
            with torch.no_grad():
                _cost_normalizado, _ll, pi = modelo(entrada, return_pi=True)
            rotas = pi_para_rotas(pi[0].cpu(), clientes_ids)

            custo = inst.solution_cost(rotas)
            gap = 100 * (custo - inst.optimum) / inst.optimum
            viavel_estrito, motivo_estrito = inst.is_feasible(rotas)
            viavel_sem_k, motivo_sem_k = viavel_sem_restricao_k(inst, rotas)

            linhas_csv.append({
                "conjunto": set_name, "instancia": inst.name, "n_customers": inst.n_customers,
                "tamanho_checkpoint_usado": tam_escolhido, "checkpoint_fine_tunado": era_finetunado,
                "n_rotas_produzidas": len([r for r in rotas if r]), "n_vehicles_ref": inst.n_vehicles_ref,
                "vehicles_exact": inst.vehicles_exact, "custo": custo, "otimo": inst.optimum, "gap_pct": gap,
                "viavel_estrito": viavel_estrito, "motivo_estrito": motivo_estrito,
                "viavel_sem_restricao_k": viavel_sem_k, "motivo_sem_k": motivo_sem_k,
            })
            marca = "OK" if viavel_estrito else ("(sem-K ok)" if viavel_sem_k else "INVIAVEL")
            print(f"  {inst.name:16s} n={inst.n_customers:3d} ckpt={tam_escolhido:3d}"
                  f"{'*' if era_finetunado else ' '}  rotas={len(rotas):3d}(exige {inst.n_vehicles_ref})"
                  f"  gap={gap:7.2f}%  {marca}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        out_csv = ROOT / "results" / ("kool_zeroshot_74.csv" if args.zero_shot else "kool_avaliacao_74.csv")
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas_csv[0].keys()))
        w.writeheader()
        w.writerows(linhas_csv)
    print(f"\nCSV salvo em: {out_csv}")

    n_estrito = sum(1 for l in linhas_csv if l["viavel_estrito"])
    n_sem_k = sum(1 for l in linhas_csv if l["viavel_sem_restricao_k"])
    print(f"\nViáveis (critério estrito, com restrição de K): {n_estrito}/{len(linhas_csv)}")
    print(f"Viáveis (ignorando K -- só capacidade+cobertura): {n_sem_k}/{len(linhas_csv)}")
    gaps_sem_k = [l["gap_pct"] for l in linhas_csv if l["viavel_sem_restricao_k"]]
    if gaps_sem_k:
        print(f"Gap médio entre as viáveis (sem restrição de K): "
              f"{sum(gaps_sem_k)/len(gaps_sem_k):.2f}%")


if __name__ == "__main__":
    main()
