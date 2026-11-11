#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source slurm/settings.sh

python3 -m venv "${FACENET_VENV}"
source "${FACENET_VENV}/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet "torch==${TORCH_VERSION}" torchvision --index-url "${TORCH_INDEX_URL}"
pip install --quiet pytorch-metric-learning==3.0.2 scikit-learn pydantic pillow pytest
python -c "import torch, torchvision, triton; print(torch.__version__, torchvision.__version__, triton.__version__, torch.cuda.is_available())"

mkdir -p "${FACENET_DATA_DIRECTORY}"
cp -r data/105_classes_pins_dataset data/lfw-deepfunneled "${FACENET_DATA_DIRECTORY}/"
find "${FACENET_DATA_DIRECTORY}" -name "*.jpg" | wc -l
