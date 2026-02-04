# Flowers Recognition 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red)

A Deep Learning project for classifying flower species using PyTorch and Transfer Learning with **ResNet18**. This project provides a complete pipeline from data loading (using `ImageFolder`) to training, evaluation, and inference.

## Table of Contents
- [About the Project](#about-the-project)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Setup](#data-setup)
- [Usage](#usage)
  - [Training](#training)
  - [Inference / Prediction](#inference--prediction)
- [Configuration](#configuration)
- [Results](#results)

## 📖 About the Project

This project leverages the power of Convolutional Neural Networks (CNNs) to recognize different types of flowers. We utilize the **ResNet18** architecture, pre-trained on ImageNet, and fine-tune it for a specific flowers dataset.

**Dataset Source:** [Kaggle - Flowers Recognition](https://www.kaggle.com/datasets/nadyana/flowers)

## Project Structure

```bash
FlowersRecognition/
├── config.yaml            # Configuration parameters (paths, hyperparameters)
├── data/                  # Dataset storage (excluded from git)
│   └── raw/               # Raw image data
├── models/                # Saved models and checkpoints
│   └── checkpoints/       # Training checkpoints (.pth files)
├── notebooks/             # Jupyter notebooks for experimentation
├── outputs/               # Training logs and visualizations
│   ├── logs/
│   ├── plots/
│   └── reports/
├── requirements.txt       # Python dependencies
└── src/                   # Source code
    ├── predict.py         # Inference script
    └── train.py           # Main training script
```

## Getting Started

### Prerequisites
*   **OS**: Windows (Tested), Linux, or macOS.
*   **Python**: Version 3.8 or higher.
*   **Compute**: CUDA-enabled GPU is recommended for faster training.

### Installation

1.  **Clone the repository**
    ```bash
    git clone <your-repo-url>
    cd FlowersRecognition
    ```

2.  **Create and activate a virtual environment (Recommended)**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Linux/Mac
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Data Setup

1.  Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nadyana/flowers).
2.  Unzip the content. You should have a folder (e.g., `flowers`) containing subfolders for each class (daisy, rose, etc.).
3.  **Use one of the following methods:**
    *   **Option A**: Move the `flowers` folder into `data/raw/` inside this project (Create `data/raw/` if it doesn't exist).
    *   **Option B**: Edit `config.yaml` and update the `root_dir` to point to your dataset location.

## Usage

### Training
To start training the model, run the `train.py` script. The script takes parameters from `config.yaml`.

```bash
python src/train.py --config config.yaml
```

The script will:
*   Load and augment images using `torchvision`.
*   Train the ResNet18 model for the specified epochs.
*   Save the best performing model to `models/checkpoints/`.

### Inference / Prediction
To classify a new image, use the `predict.py` script.

```bash
python src/predict.py --image "path/to/your/image.jpg" --model models/checkpoints/best_model.pth --threshold 0.7
```

**Arguments:**
*   `--image`: Absolute or relative path to the image you want to classify.
*   `--model`: Path to the trained `.pth` model file.
*   `--threshold`: Confidence threshold (0.0 - 1.0).
    *   *Default (0.7)*: Recommended for high confidence.
    *   *Lower (0.5)*: More lenient predictions.

## Configuration

The `config.yaml` file is the central control for the project. You can modify:

*   **data**: `root_dir`, `img_size`, `batch_size`.
*   **train**: `epochs`, `learning_rate`, `device` (cuda/cpu).
*   **model**: `type`, `pretrained` status.

## Results

During and after training, check the `outputs/` directory for:
*   **Plots**: Training/Validation Loss and Accuracy graphs.
*   **Reports**: Detailed classification metrics and logs.

---

