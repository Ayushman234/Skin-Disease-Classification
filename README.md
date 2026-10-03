# DermAI — Skin Disease Classification

A deep learning-based web application that classifies skin disease images into **5 categories** using a fine-tuned ResNet50 model with two-phase transfer learning. Built with TensorFlow/Keras and deployed via a Flask web interface.

---

## Table of Contents

- [Overview](#overview)
- [Diseases Classified](#diseases-classified)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Outputs](#outputs)
- [Web Application](#web-application)

---

## Overview

DermAI uses **transfer learning** on a pre-trained ResNet50 backbone to perform multi-class skin disease classification. The model is trained in two phases — a warm-up phase where only the classifier head is trained, followed by a fine-tuning phase where the top layers of ResNet50 are unfrozen and adapted to dermatology images.

The trained model is served through a **Flask web application** that allows users to upload a skin image and instantly receive a prediction with confidence scores and disease information.

---

## Diseases Classified

| # | Class | Severity |
|---|---|---|
| 0 | Eczema | Chronic |
| 1 | Psoriasis | Chronic |
| 2 | Melanoma | Critical — Consult a doctor immediately |
| 3 | Basal Cell Carcinoma | High — Medical attention required |
| 4 | Benign Keratosis | Benign — Non-cancerous |

---

## Dataset

| Property | Details |
|---|---|
| **Source** | [Kaggle Skin Disease Dataset](https://www.kaggle.com/) |
| **Classes used** | 5 (see above) |
| **Train / Test split** | 80% / 20% (stratified) |

### Data Augmentation

Applied offline before training to improve generalization:

| Technique | Details |
|---|---|
| Horizontal flip | Left-right mirror |
| Rotation | Random angle in [−20°, +20°] |
| Random crop | 10–20 px from each side, resized back to 224×224 |
| Gaussian blur | Kernel size 3×3 |

---

## Model Architecture

The model uses **ResNet50** (pre-trained on ImageNet) as the feature extraction backbone, with a custom classifier head on top.

```
Input (224 × 224 × 3)
    │
    ▼
ResNet50 Backbone (pre-trained on ImageNet)
    │   175 layers — feature extractor
    ▼
GlobalAveragePooling2D          →  2048-D feature vector
    │
    ▼
Dense(512, ReLU)
BatchNormalization
Dropout(0.4)
    │
    ▼
Dense(256, ReLU)
BatchNormalization
Dropout(0.3)
    │
    ▼
Dense(5, Softmax)               →  5 disease classes
```

### Key Design Choices

| Choice | Reason |
|---|---|
| **224×224 input** | ResNet50 native resolution for optimal feature extraction |
| **ImageNet `preprocess_input`** | Required for pretrained weights — channel-wise mean subtraction instead of simple /255 |
| **GlobalAveragePooling2D** | Reduces spatial feature maps to a compact 2048-D vector |
| **BatchNormalization in head** | Stabilizes training of the classifier on top of frozen features |
| **Two-stage training** | Prevents catastrophic forgetting of ImageNet features |

---

## Training Strategy

Training is split into two phases:

### Phase 1 — Head Warm-Up (15 epochs)

- ResNet50 backbone is **fully frozen** (175 layers locked)
- Only the custom classifier head is trained
- Learning rate: `1e-3`
- Higher LR is safe since pretrained weights are untouched

### Phase 2 — Fine-Tuning (20 epochs)

- Top **30 ResNet50 layers** are unfrozen
- Very small learning rate: `1e-5` to avoid catastrophic forgetting
- Adapts ImageNet features to the dermatology domain

### Callbacks (both phases)

| Callback | Monitors | Config |
|---|---|---|
| `ModelCheckpoint` | `val_accuracy` | Saves best weights only |
| `EarlyStopping` | `val_loss` | patience=5, restores best weights |
| `ReduceLROnPlateau` | `val_loss` | factor=0.5, patience=3, min_lr=1e-7 |

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Batch size | 32 |
| Phase 1 epochs | 15 |
| Phase 2 epochs | 20 |
| Seed | 42 |

---

## Project Structure

```
SkinDiseaseClassification/
├── app/
│   ├── app.py                  ← Flask web application (DermAI)
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/main.js
│   └── templates/
│       └── index.html
├── dataset/
│   └── IMG_CLASSES/
│       ├── 1. Eczema 1677/
│       ├── 7. Psoriasis pictures .../
│       └── ...
├── models/
│   ├── best_model.keras        ← Best checkpoint (used by the web app)
│   ├── final_model.keras       ← Model saved after last training epoch
│   └── history_phase1.json     ← Phase 1 history for resumable training
├── outputs/
│   ├── accuracy.png            ← Phase 1 + Phase 2 combined accuracy plot
│   ├── loss.png                ← Phase 1 + Phase 2 combined loss plot
│   └── confusion_matrix.png    ← 5-class confusion matrix on test set
├── src/
│   ├── config.py               ← Shared configuration (paths, classes, hyperparams)
│   ├── load_dataset.py         ← Dataset loading utility
│   ├── preprocess.py           ← Image loading, augmentation, ResNet50 preprocessing
│   ├── model.py                ← ResNet50 + classifier head architecture
│   ├── train.py                ← Two-phase transfer learning training script
│   ├── evaluate.py             ← Model evaluation + confusion matrix generation
│   └── predict.py              ← Single-image CLI inference script
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repository:**
```bash
git clone https://github.com/debasish07code/Skin-Disease-Classification.git
cd Skin-Disease-Classification
```

**2. (Recommended) Create a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Place dataset:**

Download the skin disease dataset from Kaggle and place the class folders inside:
```
dataset/IMG_CLASSES/
```

---

## Usage

All commands should be run from the **project root** `SkinDiseaseClassification/`.

### Train the Model

```bash
python src/train.py
```

> To skip Phase 1 on subsequent runs (checkpoint already saved), set `SKIP_PHASE1 = True` inside `src/train.py`.

### Evaluate the Model

```bash
python src/evaluate.py
```

Outputs test accuracy, per-class classification report, and saves confusion matrix to `outputs/`.

### Predict on a Single Image

```bash
python src/predict.py "C:/path/to/skin_image.jpg"
```

Prints predicted class, confidence score, and full probability breakdown for all 5 classes.

---

## Outputs

| File | Description |
|---|---|
| `models/best_model.keras` | Best model checkpoint saved during training |
| `models/final_model.keras` | Model saved after the final training epoch |
| `models/history_phase1.json` | Phase 1 training history (enables Phase 1 skip on re-runs) |
| `outputs/accuracy.png` | Training vs validation accuracy across both phases |
| `outputs/loss.png` | Training vs validation loss across both phases |
| `outputs/confusion_matrix.png` | 5-class confusion matrix (counts + normalized %) |

---

## Web Application

DermAI is a Flask-based web interface for real-time skin disease prediction.

### Run the App

```bash
python app/app.py
```

Then open your browser and visit:
```
http://127.0.0.1:5000
```

### Features

- 🖼️ **Drag-and-drop / click-to-upload** image interface
- 🔬 **Real-time prediction** with confidence score
- 📊 **Full probability breakdown** across all 5 disease classes
- 💊 **Disease info cards** — description, symptoms, and severity for each class
- ⚕️ **Medical disclaimer** for responsible AI use

### Supported Upload Formats

`JPG` · `JPEG` · `PNG` · `BMP` · `WEBP` &nbsp;(max 10 MB)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow 2.21 / Keras 3.15 |
| Backbone | ResNet50 (ImageNet pretrained) |
| Image Processing | OpenCV, NumPy |
| Web Framework | Flask 3.1 |
| Evaluation | scikit-learn, seaborn, matplotlib |
| Language | Python 3 |

---

> **Medical Disclaimer:** DermAI is an AI-assisted tool intended for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for any skin concerns.
