# Mobile Phone Price Classification with Neural Networks

PhoneBrain is a machine learning project that predicts mobile phone price categories using PyTorch-based neural networks. The project includes a full data pipeline, a KNN baseline, hyperparameter tuning, Docker support, CI testing, and a clean modular codebase. Designed as a hands-on Data Science project focused on classification, model comparison, and neural network experimentation.

This project trains a PyTorch neural network to predict the **price category**
(low, medium, high) of mobile phones based on hardware and feature data.

We compare the neural network performance against a **K-Nearest-Neighbors baseline**
and then explore hyperparameter tuning to improve accuracy.

Dataset file:
`MobilePhone.csv`

---

## Features and Label

- **Label**: `Price Range` (low / medium / high)
- **Features**: all other numeric columns in the dataset

---

## Repository Structure

- `data/` – dataset CSV placed here  
- `src/data_loading.py` – loading, preprocessing, scaling, train/test split  
- `src/model.py` – PyTorch MLP model  
- `src/train_eval.py` – training + evaluation helpers  
- `src/knn_baseline.py` – KNN baseline training + evaluation  
- `src/hyperparam_search.py` – controlled hyperparameter experiments  
- `scripts/` – runnable scripts for baseline & tuning  
- `tests/` – PyTest unit tests  
- `.github/workflows/ci.yml` – CI pipeline  
- Docker + Makefile included  

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

---

# Usage
## 1) Run Neural Network baseline
python scripts/run_baseline_nn.py

---

# LICENSE (MIT)

---



