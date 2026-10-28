# CV_TermProject

Trading-Day-Aligned Vision Transformer for Financial Chart-Based Return Prediction

## Project Overview
Financial chart-based binary return-direction prediction using CNN, Standard ViT, and Trading-Day-Aligned ViT.

## Environment
- Python 3.11
- PyTorch with CUDA 12.6
- Conda environment: cv_termproject

## Setup
conda create -n cv_termproject python=3.11 -y
conda activate cv_termproject
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
