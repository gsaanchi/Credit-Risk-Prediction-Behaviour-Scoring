# Automated Research Paper Classification using Transformers

An end-to-end **Natural Language Processing (NLP)** pipeline for automatically assigning research papers to relevant academic categories using their **titles and abstracts**.

The project formulates research-paper tagging as a **multi-label text classification problem across 57 academic categories** and fine-tunes Transformer-based language models including **RoBERTa, DeBERTa and DeBERTa-Large**. A probability-level ensemble of multiple Transformer models is used for the final predictions, achieving a **Weighted F1 Score of 0.70**.

---

## Overview

Academic papers can belong to multiple research areas simultaneously. Manually assigning these subject categories is time-consuming and can be inconsistent, particularly for interdisciplinary work.

This project automates that process by learning semantic representations from paper titles and abstracts and predicting one or more relevant research categories.

### Problem

**Input**

* Research paper title
* Research paper abstract

**Output**

* One or more academic categories from **57 possible labels**

For example:

```text
Title:
An Analysis of Complex-Valued CNNs for RF Data

Predicted Categories:
cs.LG, cs.IT, eess.SP, math.IT
```

---

## Dataset

The dataset contains **51,210 labelled research papers**, with each paper containing:

* `Id`
* `Title`
* `Abstract`
* `Categories`

Since a single paper can belong to multiple categories, the target labels are converted into a **57-dimensional multi-hot representation** using `MultiLabelBinarizer`.

### Dataset Split

To preserve the distribution of labels across the highly imbalanced multi-label dataset, the data is divided using **iterative multi-label stratification**.

| Split      |     Papers |
| ---------- | ---------: |
| Training   |     41,050 |
| Validation |      5,102 |
| Test       |      5,058 |
| **Total**  | **51,210** |

The paper **title and abstract are concatenated** before being passed to the Transformer models.

---

## Model Architecture

Multiple pretrained Transformer architectures were fine-tuned for the classification task.

### Models Experimented With

* **RoBERTa-Base**
* **DeBERTa-Base**
* **DeBERTa-Large**

Each Transformer encoder is followed by:

```text
Transformer Encoder
        ↓
Contextual Representation
        ↓
Dropout (0.3)
        ↓
Linear Classification Layer
        ↓
57 Output Logits
        ↓
Sigmoid
        ↓
Multi-label Predictions
```

Unlike single-label classification, each of the 57 outputs is treated independently because one research paper may belong to multiple categories.

---

## Handling Class Imbalance

The dataset contains a strong **long-tail label distribution**, where some research categories contain thousands of examples while others contain only a few hundred.

Instead of relying only on standard Binary Cross Entropy, the training pipeline implements an **imbalance-aware Resample Loss** incorporating:

* Label-frequency based rebalancing
* Focal-loss weighting
* Logit regularization
* Class-frequency information

This increases the importance of difficult and underrepresented categories during training.

Key loss parameters include:

```text
Reweighting: Rebalance
Focal Alpha: 0.5
Focal Gamma: 2
Negative Scale: 2.0
```

---

## Training Configuration

| Parameter               |      Value |
| ----------------------- | ---------: |
| Maximum sequence length | 256 tokens |
| Training batch size     |          4 |
| Validation batch size   |          4 |
| Optimizer               |      AdamW |
| Learning rate           |       1e-5 |
| Epochs                  |          3 |
| Dropout                 |        0.3 |
| Output classes          |         57 |

Models were trained using **PyTorch and Hugging Face Transformers** with GPU acceleration.

---

## Multi-Model Ensemble

The final prediction pipeline combines **four independently trained Transformer models**:

1. DeBERTa-Base
2. DeBERTa-Base — alternate training run
3. DeBERTa-Large
4. RoBERTa-Base

For every paper, logits from all four models are averaged:

```text
DeBERTa-Base ──────┐
                   │
DeBERTa-Base ──────┤
                   │
DeBERTa-Large ─────┼──► Average Logits
                   │          ↓
RoBERTa-Base ──────┘       Sigmoid
                              ↓
                     Threshold = 0.47
                              ↓
                     Predicted Categories
```

Probability-level ensembling reduces dependence on the errors of any individual Transformer and produces more stable predictions across the 57 categories.

---

## Threshold Optimization

Using a default probability threshold of `0.5` is not necessarily optimal for an imbalanced multi-label problem.

The pipeline therefore evaluates different probability thresholds using **F1 Score, Precision and Recall**, with the final ensemble using:

```python
threshold = 0.47
```

A fallback mechanism is also included for cases where every predicted probability falls below the threshold. In such cases, the category with the **highest predicted probability** is selected so that every paper receives at least one classification.

---

## Performance

The final Transformer ensemble achieved:

### **Weighted F1 Score: 0.70**

across **57 research categories**.

Weighted F1 is used because the label distribution is highly imbalanced and individual papers may contain multiple target categories.

---

## End-to-End Pipeline

```text
Research Paper
      │
      ├── Title
      └── Abstract
             │
             ▼
     Text Concatenation
             │
             ▼
    Transformer Tokenizer
             │
             ▼
 ┌─────────────────────────────┐
 │       Model Ensemble        │
 │                             │
 │  DeBERTa-Base               │
 │  DeBERTa-Base               │
 │  DeBERTa-Large              │
 │  RoBERTa-Base               │
 └─────────────────────────────┘
             │
             ▼
      Average Model Outputs
             │
             ▼
           Sigmoid
             │
             ▼
    Optimized Thresholding
             │
             ▼
  One or More of 57 Categories
```

---

## Repository Structure

```text
Automated-Research-Paper-Classification/
│
├── README.md
│
├── deberta.ipynb
│   └── DeBERTa-Base training pipeline
│
├── deberta6.ipynb
│   └── Alternate DeBERTa training configuration
│
├── debertalarge.ipynb
│   └── DeBERTa-Large training pipeline
│
├── roberta3.ipynb
│   └── RoBERTa-Base training pipeline
│
└── inference.ipynb
    └── Multi-model ensemble and final inference pipeline
```

---

## Tech Stack

### Machine Learning

* PyTorch
* Hugging Face Transformers
* Scikit-learn
* Scikit-multilearn

### Models

* RoBERTa
* DeBERTa
* DeBERTa-Large

### Data Processing

* Pandas
* NumPy
* MultiLabelBinarizer
* Iterative Multi-Label Stratification

### Training

* AdamW
* Resample Loss
* Focal Loss
* GPU-based Transformer fine-tuning

---

## Key Features

* **51K+ research papers**
* **57-class multi-label classification**
* Combined **title + abstract semantic modeling**
* Fine-tuned **RoBERTa and DeBERTa Transformers**
* **DeBERTa-Large** experimentation
* Multi-label stratified train/validation/test splitting
* Long-tail **class imbalance handling**
* Focal and rebalanced loss functions
* **4-model Transformer ensemble**
* Probability-level model aggregation
* F1-based threshold optimization
* End-to-end inference pipeline
* **0.70 Weighted F1 Score**

---

## Future Improvements

Potential extensions include:

* Experimenting with scientific-domain pretrained models such as **SciBERT**
* Per-class threshold optimization instead of a single global threshold
* Calibration of ensemble probabilities
* Knowledge distillation into smaller Transformer models
* FastAPI/Flask deployment for real-time paper classification
* Explainability using attention visualization or SHAP
* Retrieval-based augmentation using related academic papers

---

## Author

**Saanchi Gupta**

GitHub: [@gsaanchi](https://github.com/gsaanchi)
