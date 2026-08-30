# ❤️ Project 3.1: Heart Disease Diagnosis Using Machine Learning

**AI VIET NAM – AI Course 2026**  
*A comprehensive, modular machine learning project for diagnosing coronary artery disease (CAD) using clinical datasets, advanced ensemble techniques, explainability, and interactive web deployment.*

---

## 📌 Table of Contents
- [1. Overview & Problem Definition](#1-overview--problem-definition)
- [2. Repository Architecture](#2-repository-architecture)
- [3. Quickstart & Installation](#3-quickstart--installation)
- [4. Implementation Roadmap (Your Coding Path)](#4-implementation-roadmap-your-coding-path)
- [5. Baseline Methods & Algorithms](#5-baseline-methods--algorithms)
- [6. Advanced Ensemble Techniques](#6-advanced-ensemble-techniques)
- [7. Benchmark Performance Target](#7-benchmark-performance-target)
- [8. Candidate Improvements & Honors Extensions](#8-candidate-improvements--honors-extensions)
- [9. Interactive Web Application & Cloud Deployment](#9-interactive-web-application--cloud-deployment)
- [10. Project Rubric & Self-Assessment](#10-project-rubric--self-assessment)

---

## 1. Overview & Problem Definition

Cardiovascular disease remains the leading cause of mortality worldwide (~20.5 million deaths in 2023, accounting for ~32% of all global fatalities). In this project, we develop an end-to-end Machine Learning system trained on the **Cleveland Heart Disease Dataset** ($N=303$ patients, 13 diagnostic features).

### Primary Objectives:
1. **Data Engineering:** Build clean, leakage-free Scikit-Learn preprocessing pipelines.
2. **Feature Engineering & Selection:** Construct physiological risk ratios and apply Mutual Information and Decision Tree MDI ranking.
3. **Multi-Model Benchmarking:** Implement and compare 9 algorithms across 4 dataset variations.
4. **Interactive Deployment:** Build a Streamlit clinical decision support dashboard deployed via Cloudflared / Docker.

---

## 2. Repository Architecture

```
conquer/
├── README.md                          # Main project guide, setup & instructions (You are here)
├── requirements.txt                   # All dependencies (pandas, scikit-learn, xgboost, streamlit, optuna, shap)
├── data/
│   ├── download_data.py               # Data download script (UCI ML Repo / Google Drive)
│   └── splits/                        # Destination directory for CSV dataset splits
├── src/                               # Modular core library (Scaffolded with hints & TODOs)
│   ├── config.py                      # Global seeds, dataset schemas, search spaces
│   ├── data/
│   │   ├── loader.py                  # Ingestion, coercion, binarization, stratified split
│   │   └── preprocessor.py            # ColumnTransformer, Imputer & Scaler pipelines
│   ├── features/
│   │   ├── engineering.py             # Domain age-ratios & binning transformers
│   │   └── selection.py               # Mutual Information & Decision Tree MDI selectors
│   ├── models/
│   │   ├── base_models.py             # Naive Bayes, KNN, Decision Tree, KMeans, Stacking
│   │   ├── ensemble_models.py         # Random Forest, AdaBoost, Gradient Boosting, XGBoost
│   │   └── advanced_models.py         # [IMPROVEMENTS]: SVM, Logistic Regression, MLP, Weighted KNN
│   ├── evaluation/
│   │   ├── metrics.py                 # Accuracy, Precision, Recall, F1, ROC-AUC
│   │   └── visualization.py           # Heatmaps, CV curves, feature importance, benchmark bar plots
│   └── tuning/
│       └── hyperopt.py                # [IMPROVEMENTS]: Optuna Bayesian search & GridSearchCV
├── app/
│   ├── app.py                         # Streamlit interactive diagnostic dashboard
│   └── deploy_colab.sh                # Automated Cloudflared tunnel for Google Colab
├── notebooks/                         # Step-by-step guided interactive notebooks
│   ├── 01_eda_and_preprocessing.ipynb # Ingestion, EDA, leakage prevention, feature engineering
│   ├── 02_baseline_models.ipynb       # Baseline classifiers (NB, KNN, DT, KMeans, Stacking)
│   ├── 03_ensemble_advancement.ipynb  # Advanced ensemble methods (RF, AdaBoost, GB, XGBoost)
│   └── 04_model_improvements.ipynb    # Extensions (SVM, MLP, RobustScaler, Optuna, SHAP)
├── docker/                            # Containerization
│   ├── Dockerfile
│   └── docker-compose.yml
└── docs/
    ├── Project-3.1.pdf                # Reference course paper
    ├── BASELINE_GUIDE.md              # Detailed mathematical equations & baseline instructions
    ├── IMPROVEMENTS_GUIDE.md          # Comprehensive hints & recipes for all improvements
    └── RUBRIC_CHECKLIST.md            # Self-assessment checklist matching grading criteria
```

---

## 3. Quickstart & Installation

### Step 1: Clone or Open Workspace
Ensure you are in the project root:
```bash
cd ./Module-03/conquer
```

### Step 2: Set up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Fetch Dataset
```bash
python3 data/download_data.py
```

---

## 4. Implementation Roadmap (Your Coding Path)

This repository is designed for active learning. All functions and classes are structured with **type annotations, mathematical docstrings, step-by-step hints, and `TODO` markers**.

```mermaid
graph LR
    N1[1. Data & Preprocessing] --> N2[2. Feature Engineering]
    N2 --> N3[3. Baseline Classifiers]
    N3 --> N4[4. Advanced Ensembles]
    N4 --> N5[5. Web App & Deployment]
    N5 --> N6[6. Advanced Improvements]
```

### Recommended Workflow:
1. **Explore & Prepare Data:** Open [`notebooks/01_eda_and_preprocessing.ipynb`](notebooks/01_eda_and_preprocessing.ipynb) and complete `src/data/loader.py`, `src/data/preprocessor.py`, and `src/features/`.
2. **Implement Baseline Models:** Open [`notebooks/02_baseline_models.ipynb`](notebooks/02_baseline_models.ipynb) and complete `src/models/base_models.py`.
3. **Build Advanced Ensembles:** Open [`notebooks/03_ensemble_advancement.ipynb`](notebooks/03_ensemble_advancement.ipynb) and complete `src/models/ensemble_models.py`.
4. **Deploy Web Interface:** Complete `train_best_decision_tree_pipeline()` in [`app/app.py`](app/app.py) and launch the app.
5. **Explore Improvements:** Open [`notebooks/04_model_improvements.ipynb`](notebooks/04_model_improvements.ipynb) and reference [`docs/IMPROVEMENTS_GUIDE.md`](docs/IMPROVEMENTS_GUIDE.md).

---

## 5. Baseline Methods & Algorithms

1. **Gaussian Naive Bayes (`GaussianNB`):** Closed-form Gaussian likelihood under the conditional independence assumption.
2. **K-Nearest Neighbors (`KNN`):** Majority voting over Euclidean neighborhood, tuned across $K \in [1, 20]$ with Stratified 5-Fold CV.
3. **Decision Tree (`DecisionTreeClassifier`):** Recursive binary recursive partitioning, tuned across `max_depth` $\in [2, 10]$.
4. **K-Means Clustering Classifier:** $k=2$ unsupervised clustering converted into a classifier via training cluster-to-label mode mapping.
5. **Stacking Ensemble:** Base classifiers (`KNN`, `DT`, `NB`) with a `KNN` meta-estimator.

---

## 6. Advanced Ensemble Techniques

1. **Random Forest (Bagging):** Reduces variance by averaging $B$ uncorrelated deep decision trees trained on bootstrap samples with random feature subsampling ($m = \sqrt{p}$).
2. **AdaBoost (Boosting):** Sequentially fits shallow decision stumps, increasing the weight of misclassified training samples at each round.
3. **Gradient Boosting:** Sequentially fits decision trees on the negative gradient (pseudo-residuals) of the logloss loss function with a learning rate $\eta = 0.1$.
4. **XGBoost:** 2nd-order Taylor expansion objective, exact histogram-based split finding (`tree_method='hist'`), and $L_1/L_2$ leaf regularization.

---

## 7. Benchmark Performance Target

Reproduce and verify the 9-model baseline performance table:

| Model | Val (Origin) | Val (FE) | Val (Origin+DT) | Val (FE+DT) | Test (Origin) | Test (FE) | Test (Origin+DT) | Test (FE+DT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes** | 0.90 | 0.90 | 0.93 | 0.93 | 0.84 | 0.84 | 0.84 | 0.84 |
| **KNN** | 0.90 | 0.90 | 0.97 | 0.87 | 0.84 | 0.84 | 0.87 | 0.84 |
| **K-Means** | 0.70 | 0.80 | 0.83 | 0.63 | 0.87 | 0.87 | 0.84 | 0.77 |
| **Decision Tree** | 0.93 | 0.93 | 0.93 | 0.93 | 0.81 | 0.81 | 0.81 | 0.81 |
| **Stacking** | 0.87 | 0.93 | 0.90 | 0.87 | 0.84 | **0.90** | 0.84 | 0.84 |
| **Random Forest** | **0.97** | 0.90 | **0.97** | 0.93 | **0.90** | 0.87 | 0.84 | 0.84 |
| **AdaBoost** | **0.97** | **0.97** | **0.97** | 0.93 | 0.81 | 0.84 | 0.81 | 0.84 |
| **Gradient Boosting**| 0.87 | 0.83 | 0.87 | 0.93 | 0.81 | 0.81 | 0.84 | 0.81 |
| **XGBoost** | 0.90 | 0.87 | 0.93 | 0.90 | 0.84 | 0.87 | 0.81 | 0.87 |

---

## 8. Candidate Improvements & Honors Extensions

Refer to [`docs/IMPROVEMENTS_GUIDE.md`](docs/IMPROVEMENTS_GUIDE.md) for full recipes:
- **Outlier Scaling:** `RobustScaler` on skewed clinical markers (`chol`, `trestbps`).
- **Feature Filtering:** `VarianceThreshold` and correlation-based redundancy pruning.
- **Higher-Order Interactions:** `PolynomialFeatures(degree=2, interaction_only=True)`.
- **Additional Models:** Regularized `LogisticRegression`, Support Vector Machines (`SVC` Linear/RBF), `MLPClassifier`, and `WeightedKNN`.
- **Bayesian Optimization:** Automated multi-parameter tuning with **Optuna** (`src/tuning/hyperopt.py`).
- **Explainable AI:** **SHAP** global summary plots and single-patient waterfall explanation charts.

---

## 9. Interactive Web Application & Cloud Deployment

### Run Locally:
```bash
streamlit run app/app.py
```

### Run via Docker:
```bash
docker-compose -f docker/docker-compose.yml up --build
```

### Deploy on Google Colab:
```bash
bash app/deploy_colab.sh
```

---

## 10. Project Rubric & Self-Assessment

Review [`docs/RUBRIC_CHECKLIST.md`](docs/RUBRIC_CHECKLIST.md) to ensure all criteria across Sections I through VI are satisfied before final submission.
