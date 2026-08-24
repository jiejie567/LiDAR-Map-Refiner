#!/usr/bin/env bash
set -euo pipefail

echo "[manual-loop-closure] Installing Ubuntu 20.04 Python GUI dependencies..."

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  python3-pip \
  python3-venv \
  python3-dev \
  libboost-all-dev \
  libtbb-dev

echo
echo "[manual-loop-closure] Done."
echo "[manual-loop-closure] Next steps:"
echo "  1) Create the Python GUI environment:"
echo "       conda env create -f environment.yml"
echo "       conda activate manual-loop-closure"
echo "  2) Install the GTSAM Python wrapper:"
echo "       bash scripts/install_gtsam_python.sh"
