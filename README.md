# Improving Multi-Class Mental Health Text Classification with BERT

**A Comparison of Model Optimization Techniques with a Focus on Minority Class Detection**

*Nikhil Trivedi — Independent Research, March 2026*

---

## Overview

This repository contains the full research pipeline for a study investigating how hyperparameter tuning, targeted data augmentation, and domain-specific transformer architecture compare in improving BERT-based multi-class mental health text classification across seven mental health categories: Normal, Depression, Anxiety, Stress, Suicidal, Bipolar, and Personality Disorder.

The final optimized model (MentalBERT + augmentation + tuning) achieved a **macro F1 of 86.25%** and **accuracy of 86.19%**, up from a baseline macro F1 of 83.34%.

---

## Repository Structure

```
├── notebook/
│   └── Modeling.ipynb                         # Full experimental pipeline (Google Colab)
│
├── paper/
│   └── Improving_Multi-Class_Mental_Health_Text_Classification_with_BERT.pdf
│
├── results/
│   ├── MacroF1_and_Accuracy_Progression.png   # Model comparison bar chart
│   └── ConfusionMatrix_FinalModel.pdf         # Confusion matrix for the final model
│
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Research Question

> How do the modeling strategies hyperparameter tuning, class-specific data augmentation, and domain-specific architecture compare in maximizing BERT's performance in multi-class mental health text classification, with an emphasis on improving minority class detection?

---

## Experimental Stages

### Stage 0 — Baseline BERT-base
Trained `bert-base-uncased` with default hyperparameters (lr=1e-4, batch size=64, max_len=256) for 2 epochs.

| Metric | Score |
|---|---|
| Accuracy | 84.46% |
| Macro F1 | 83.34% |

### Stage 1 — Hyperparameter Tuning
Grid search over learning rates [1e-5, 5e-5, 1e-4], batch sizes [16, 32, 64], and max sequence lengths [128, 256]. Optimal configuration: lr=5e-5, batch size=32, max_len=256.

| Metric | Score |
|---|---|
| Accuracy | 84.81% |
| Macro F1 | 84.06% |

### Stage 2 — Targeted Data Augmentation
Synonym replacement (NLPaug/WordNet) applied to the Stress class only. Augmenting both Personality Disorder and Stress degraded Personality Disorder performance; augmenting Stress alone yielded strong gains.

| Metric | Score |
|---|---|
| Accuracy | 85.60% |
| Macro F1 | 85.42% |

### Stage 3 — Domain-Specific Architecture (MentalBERT)
Switched to `mental/mental-bert-base-uncased`, pretrained on mental health Reddit data, combined with Stress augmentation and a second grid search (lr ∈ [5e-5, 1e-4], batch size ∈ [16, 32, 64]).

| Metric | Score |
|---|---|
| Accuracy | 86.19% |
| Macro F1 | 86.25% |

---

## Key Findings

- **Targeted augmentation produced the largest single gains.** Augmenting only the Stress class improved its F1 score by ~8.5% on BERT-base and ~7.8% on MentalBERT. Over-augmenting a very small class (Personality Disorder) introduced noise and hurt performance.
- **Domain-specific pretraining matters most when combined with other strategies.** MentalBERT alone offered modest gains, but pairing it with augmentation and tuning produced the best results overall.
- **Hyperparameter tuning improved stability but not semantics.** It could not resolve the persistent misclassification between Depression and Suicidal, which stems from inherent linguistic similarity between these classes.
- **Macro F1 is the more informative metric here.** Accuracy masked weaknesses in minority classes; macro F1 exposed them.

---

## Dataset

The dataset is sourced from Kaggle: [Mental Health Dataset](https://www.kaggle.com/datasets/szegeelim/mental-health/data)

It consists of short text statements collected from public Reddit and Twitter forums, labeled with one of seven mental health statuses. After cleaning (removing NaN rows), the dataset contains 52,681 samples with a highly imbalanced class distribution.

> **Note:** The dataset is not included in this repository. Download it from the Kaggle link above and place `Combined Data.csv` in your working directory before running the notebook.

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
torch
transformers
datasets
pandas
scikit-learn
nlpaug
nltk
matplotlib
tqdm
huggingface-hub
```

The notebook was developed and run on **Google Colab** with GPU acceleration. Running locally requires a CUDA-compatible GPU for reasonable training times (baseline runtime ~58 minutes on Colab T4).

---

## Usage

1. Clone the repository and install requirements.
2. Download the dataset from Kaggle and place `Combined Data.csv` in your working directory.
3. Open `notebook/mental_health_classification.ipynb` in Google Colab or Jupyter.
4. Run cells sequentially. Each stage is clearly labeled and can be run independently after the data loading section is complete.

For MentalBERT (Stage 3), a Hugging Face account and access token are required:

```python
from huggingface_hub import login
login()
```

---

## Results

See the `results/` folder for:
- A bar chart comparing Macro F1 and Evaluation Accuracy across all model configurations
- The confusion matrix for the final optimized MentalBERT model

---

## Citation

If you reference this work, please cite:

```
Nikhil Trivedi. Improving Multi-Class Mental Health Text Classification with BERT: A Comparison of Model Optimization Techniques with a Focus on Minority Class Detection. Zenodo, 6 Apr. 2026, https://doi.org/10.5281/zenodo.19432858.
```

---

## References

- Calvo et al. (2017). Natural language processing in mental health applications using non-clinical texts. *Natural Language Engineering.*
- Chancellor & De Choudhury (2020). Methods in predictive techniques for mental health status on social media. *npj Digital Medicine.*
- Guo et al. (2022). Comparison of pretraining models and strategies for health-related social media text classification. *Healthcare.*
- Ji et al. (2021). MentalBERT: Publicly available pretrained language models for mental healthcare. *arXiv.*
- Martínez-Castaño et al. (2021). BERT-based transformers for early detection of mental health illnesses. *CLEF 2021.*
- Sao & Lim (2024). MIRoBERTa: Mental illness text classification with transfer learning on subreddits. *IEEE Access.*
