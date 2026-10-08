#!/usr/bin/env bash
# setup.sh
# ==========
# Prepara TODAS as dependências externas da Direção 1 (diário, Entradas
# 32-40): clona o repositório de Kool et al. (2019), baixa o gerador
# oficial de instâncias do conjunto X (Uchoa et al. 2017 -- só para
# VALIDAR uchoa_gerador_core.py contra a fonte original, não é usado em
# produção), baixa e compila o LKH3, e cria um ambiente virtual Python
# com todas as dependências (torch CPU por padrão, ortools, scipy, tqdm,
# tensorboard_logger, river -- esta última é dependência PRÉ-EXISTENTE do
# próprio HG-UCB em src/bandit.py, não algo novo desta Direção 1).
#
# ASSUME bash + gcc/make/curl-ou-wget disponíveis (Linux ou macOS). Se
# sua máquina principal é Windows, rode isto dentro do WSL (Windows
# Subsystem for Linux) -- não existe uma variante nativa deste script
# para PowerShell/cmd (a compilação do LKH3 e o restante do pipeline
# pressupõem um ambiente tipo Unix). Ver README.md para instruções de
# instalação do WSL, se ainda não tiver.
#
# Uso: bash setup.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
DIRECAO1_DIR="$(pwd)"
ROOT="$(dirname "$DIRECAO1_DIR")"
EXTERNAL="$ROOT/external"
mkdir -p "$EXTERNAL"

echo "=== 1/4: repositório de Kool et al. (attention-learn-to-route) ==="
if [ -d "$EXTERNAL/attention-learn-to-route/.git" ]; then
    echo "  já existe em $EXTERNAL/attention-learn-to-route -- pulando clone."
else
    git clone --depth 1 https://github.com/wouterkool/attention-learn-to-route.git \
        "$EXTERNAL/attention-learn-to-route"
fi
if [ ! -f "$EXTERNAL/attention-learn-to-route/pretrained/cvrp_50/epoch-99.pt" ]; then
    echo "  AVISO: checkpoints pré-treinados não encontrados em pretrained/cvrp_50/ --"
    echo "  o clone pode ter vindo sem eles (repositórios grandes às vezes usam Git LFS "
    echo "  ou o clone --depth 1 pode ter perdido algo). Verifique manualmente."
fi

# Patch pontual em train.py (repo do Kool, não nosso): a linha original cria o
# DataLoader de treino com num_workers=1, o que causou "ValueError: too many
# fds" em pelo menos uma combinação real (WSL2 + Python 3.14 + CUDA, diário
# Entrada 54) ao spawnar até um único processo-worker via multiprocessing.
# num_workers=0 evita processos-worker inteiramente -- sem custo relevante
# aqui, já que a geração de dados (gerar_instancia_uchoa) é Python puro e
# leve, não o gargalo do treino. Idempotente (só aplica se ainda não aplicado,
# então rodar setup.sh de novo não quebra nada).
TRAIN_PY="$EXTERNAL/attention-learn-to-route/train.py"
if [ -f "$TRAIN_PY" ] && grep -q "num_workers=1" "$TRAIN_PY"; then
    echo "  aplicando patch num_workers=1 -> 0 em train.py (evita 'too many fds' em alguns "
    echo "  ambientes -- ver diário, Entrada 54)..."
    sed -i 's/num_workers=1/num_workers=0/' "$TRAIN_PY"
fi

echo ""
echo "=== 2/4: gerador oficial de instâncias Uchoa et al. (só para validação) ==="
mkdir -p "$EXTERNAL/uchoa_generator"
GEN_URL="https://galgos.inf.puc-rio.br/cvrplib/uploads/files/xml100/generator.py"
if [ -f "$EXTERNAL/uchoa_generator/generator.py" ]; then
    echo "  já existe em $EXTERNAL/uchoa_generator/generator.py -- pulando download."
else
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$GEN_URL" -o "$EXTERNAL/uchoa_generator/generator.py"
    else
        wget -q "$GEN_URL" -O "$EXTERNAL/uchoa_generator/generator.py"
    fi
    echo "  baixado de $GEN_URL"
fi

echo ""
echo "=== 3/4: LKH3 (download + compilação) ==="
LKH_URL="http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.7.tgz"
if [ -x "$EXTERNAL/LKH-3.0.7/LKH" ]; then
    echo "  binário já existe em $EXTERNAL/LKH-3.0.7/LKH -- pulando."
else
    if ! command -v gcc >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
        echo "  ERRO: gcc e/ou make não encontrados -- necessários para compilar o LKH3."
        echo "  Debian/Ubuntu: sudo apt install build-essential"
        echo "  macOS: xcode-select --install"
        exit 1
    fi
    TMP_TGZ="$(mktemp)"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$LKH_URL" -o "$TMP_TGZ"
    else
        wget -q "$LKH_URL" -O "$TMP_TGZ"
    fi
    tar -xzf "$TMP_TGZ" -C "$EXTERNAL"
    rm -f "$TMP_TGZ"
    (cd "$EXTERNAL/LKH-3.0.7" && make)
    if [ ! -x "$EXTERNAL/LKH-3.0.7/LKH" ]; then
        echo "  ERRO: compilação terminou mas $EXTERNAL/LKH-3.0.7/LKH não foi criado -- "
        echo "  veja a saída do 'make' acima para o erro real."
        exit 1
    fi
    echo "  compilado com sucesso em $EXTERNAL/LKH-3.0.7/LKH"
fi

echo ""
echo "=== 4/4: ambiente virtual Python (venv/) ==="
if [ -d "$DIRECAO1_DIR/venv" ]; then
    echo "  venv já existe em $DIRECAO1_DIR/venv -- pulando criação (rode "
    echo "  'rm -rf venv' antes se quiser recriar do zero)."
else
    python3 -m venv "$DIRECAO1_DIR/venv"
fi
# shellcheck disable=SC1091
source "$DIRECAO1_DIR/venv/bin/activate"
pip install --upgrade pip --quiet

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  GPU NVIDIA detectada (nvidia-smi presente) -- instalando torch com suporte a CUDA "
    echo "  (build padrão do PyPI, alguns GB de download -- a barra de progresso abaixo é do "
    echo "  próprio pip, não travou se ela estiver parada por alguns segundos entre etapas)."
    echo "  Se isso não for o que você quer, edite este script."
    pip install torch
else
    echo "  nenhuma GPU NVIDIA detectada -- instalando torch CPU-only (build menor/mais rápido "
    echo "  de instalar). Se você TEM GPU mas nvidia-smi não está no PATH, edite este script "
    echo "  para trocar por 'pip install torch' simples."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install ortools scipy tqdm tensorboard_logger river

echo ""
echo "=== Setup concluído ==="
echo "Antes de rodar qualquer script Python deste diretório, ative o venv:"
echo "  source $DIRECAO1_DIR/venv/bin/activate"
echo ""
echo "IMPORTANTE: alguns scripts (kool_finetune_runner.py, sondar_custo_kool.py) precisam "
echo "da variável de ambiente abaixo por causa de um conflito conhecido entre a versão do "
echo "protobuf instalada e o pacote tensorboard_logger (ver diário, Entrada 40 -- não é um "
echo "bug destes scripts, é um workaround documentado do próprio tensorboard_logger):"
echo "  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"
echo ""
echo "Ordem recomendada para rodar tudo -- ver README.md deste diretório para detalhes de "
echo "cada passo, tempos esperados e como interpretar os resultados."
