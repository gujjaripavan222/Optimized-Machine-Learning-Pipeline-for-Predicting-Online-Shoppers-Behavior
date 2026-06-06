# 🛒 Optimized ML Pipeline — Online Shoppers Behavior Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-87.67%25-blue)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.903-blueviolet)

> **Classify e-commerce user sessions as potential buyers or non-buyers using a fully automated, end-to-end ML pipeline.**

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## 📖 Overview

This project builds an **optimized machine learning pipeline** to predict online shopper purchasing intent using the [Online Shoppers Intention dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset).

The pipeline automates the full ML workflow:
- **Data Preprocessing** — missing value imputation, boolean encoding, ordinal encoding
- **Feature Engineering** — correlation analysis, new feature creation (`Returning_Visitor`)
- **Class Imbalance Handling** — SMOTE oversampling
- **Feature Selection** — SelectKBest with Chi-squared scoring (top 6 features)
- **Model Selection** — 15 classifiers benchmarked with 10-fold cross-validation
- **Evaluation** — Accuracy, F1-Score, ROC-AUC, Confusion Matrix, Classification Report

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Samples | 12,330 sessions |
| Features | 17 (numerical + categorical) |
| Target | `Revenue` (True/False → buyer/non-buyer) |
| Class Imbalance | ~84.5% non-buyers vs ~15.5% buyers |

The dataset is included in this repository: [`online_shoppers_intention.csv`](./online_shoppers_intention.csv)

Key features include: `Administrative`, `Informational`, `ProductRelated` page visits & durations, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`, `Month`, `VisitorType`, `Weekend`.

---

## 📁 Project Structure

```
Optimized-ML-Pipeline-Online-Shoppers/
│
├── src.py                          # Main pipeline script
├── online_shoppers_intention.csv   # Dataset
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── LICENSE                         # MIT License
```

---

## ⚙️ Pipeline Architecture

```
Raw CSV Data
    │
    ▼
Feature Engineering
  ├── Boolean encoding (Weekend, Revenue)
  ├── New column: Returning_Visitor
  └── Ordinal encoding (Month)
    │
    ▼
Train / Test Split (70% / 30%)
    │
    ▼
IMBPipeline (per model)
  ├── ColumnTransformer
  │     ├── Numeric: SimpleImputer → MinMaxScaler
  │     └── Categorical: OneHotEncoder
  ├── SMOTE (handle class imbalance)
  ├── SelectKBest (Chi2, k=6 features)
  └── Classifier
    │
    ▼
10-Fold Cross Validation (ROC-AUC scoring)
    │
    ▼
Best Model → Final Evaluation
```

---

## 🤖 Models Evaluated

| Model | Notes |
|---|---|
| MLPClassifier ✅ | **Best performer** |
| RandomForestClassifier | Strong ensemble |
| XGBClassifier | Gradient boosting |
| LGBMClassifier | Light gradient boosting |
| ExtraTreesClassifier | Fast ensemble |
| AdaBoostClassifier | Boosting |
| BaggingClassifier | Bootstrap aggregation |
| KNeighborsClassifier | Distance-based |
| SVC | Support Vector |
| DecisionTreeClassifier | Interpretable tree |
| SGDClassifier | Stochastic gradient |
| RidgeClassifier | Linear model |
| BernoulliNB | Naive Bayes |
| ExtraTreeClassifier | Single extra tree |
| DummyClassifier | Baseline |

---

## 📈 Results

| Metric | Score |
|---|---|
| **Accuracy** | **87.67%** |
| **ROC-AUC** | **0.903** |
| Best Model | MLPClassifier (hidden_layers=(27,50), relu, adam) |
| Feature Selection | SelectKBest (Chi2, k=6) |
| Imbalance Handling | SMOTE |

The MLPClassifier (configured with architecture from research paper) achieved the highest ROC-AUC of **0.903** across 10-fold cross-validation, indicating excellent ability to distinguish buyers from non-buyers.

---

## 🛠️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/gujjaripavan222/Optimized-Machine-Learning-Pipeline-for-Predicting-Online-Shoppers-Behavior.git
cd Optimized-Machine-Learning-Pipeline-for-Predicting-Online-Shoppers-Behavior
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python src.py
```

The script will:
1. Load and preprocess the dataset
2. Engineer features
3. Run 15 classifiers with 10-fold cross-validation
4. Print ranked model performance (by ROC-AUC)
5. Train the best model (MLPClassifier) and evaluate on the test set
6. Output accuracy, F1-score, ROC-AUC, confusion matrix, and classification report

---

## 🧰 Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, pipelines, metrics |
| `imbalanced-learn` | SMOTE for class imbalance |
| `xgboost` | XGBoost classifier |
| `lightgbm` | LightGBM classifier |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical plots |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Pavan Kumar**
- GitHub: [@gujjaripavan222](https://github.com/gujjaripavan222)

---

⭐ If you found this project useful, please give it a star!
