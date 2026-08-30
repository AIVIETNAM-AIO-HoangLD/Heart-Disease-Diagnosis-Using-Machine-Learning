# Improvements & Extensions Guide

This document provides detailed hints, mathematical formulas, and code recipes for all candidate improvements suggested in the paper and project rubrics.

---

## 1. Advanced Data Preprocessing & Feature Improvements

### 1.1. Outlier-Resistant Scaling with `RobustScaler`
**Problem:** Cholesterol (`chol`) and resting blood pressure (`trestbps`) often have extreme medical outliers (e.g., $chol > 400\text{ mg/dL}$). `StandardScaler` calculates mean and standard deviation, which are sensitive to outliers.  
**Solution:** Use `RobustScaler`, which centers by the median and scales by the Interquartile Range (IQR = Q3 - Q1):
$$x_{\text{scaled}} = \frac{x - \text{median}(x)}{\text{IQR}(x)}$$

**Implementation Hint:**
```python
from sklearn.preprocessing import RobustScaler
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])
```

### 1.2. Low-Variance Feature Filtering with `VarianceThreshold`
**Problem:** Features that have almost identical values across all patients contribute noise rather than predictive signal.  
**Solution:** Calculate sample variance $\text{Var}(X) = \frac{1}{N}\sum (x_i - \bar{x})^2$ and drop features with $\text{Var}(X) < \theta$ (e.g., $\theta = 0.01$).

**Implementation Hint:**
```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X_train)
```

### 1.3. 2nd-Order Interactions via `PolynomialFeatures`
**Problem:** Linear models (Logistic Regression, Linear SVM) cannot naturally capture non-linear feature interactions (e.g., `age * trestbps`).  
**Solution:**
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_numeric)
```

---

## 2. Additional Model Architectures

### 2.1. Support Vector Machines (SVM with RBF Kernel)
**Concept:** Finds the maximum-margin hyperplane separating diseased and non-diseased patients in high-dimensional space:
$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(w^T \phi(x_i) + b) \ge 1 - \xi_i$$
The Radial Basis Function (RBF) Kernel:
$$K(x, z) = \exp\left(-\gamma \|x - z\|^2\right)$$

**Implementation Hint:**
```python
from sklearn.svm import SVC
svm = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)
svm.fit(X_train, y_train)
```

### 2.2. Regularized Logistic Regression (Interpretable Linear Baseline)
**Concept:** Models the log-odds of heart disease:
$$\log\left(\frac{P(y=1|x)}{1 - P(y=1|x)}\right) = w^T x + b$$
With $L_2$ Ridge Penalty: $\min_w -\log L(w) + \frac{1}{2C} \|w\|_2^2$.

**Implementation Hint:**
```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
```

### 2.3. Multi-Layer Perceptron (MLP) on Small Tabular Data
**Caveat on Tabular Overfitting:** Deep networks can easily overfit on small sample sizes ($N=303$). Always include:
- Small hidden layer topology: `(32, 16)`
- Weight decay penalty: `alpha=0.01`
- Early stopping: `early_stopping=True`

**Implementation Hint:**
```python
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    alpha=0.01,
    early_stopping=True,
    n_iter_no_change=15,
    max_iter=500,
    random_state=42
)
mlp.fit(X_train, y_train)
```

### 2.4. Distance-Weighted K-Nearest Neighbors
**Concept:** Standard KNN counts each of the $K$ neighbors equally. Weighted KNN weights each neighbor inversely to its Euclidean distance:
$$w_i = \frac{1}{d(x, x_i) + \epsilon}$$

**Implementation Hint:**
```python
from sklearn.neighbors import KNeighborsClassifier
wknn = KNeighborsClassifier(n_neighbors=7, weights='distance')
wknn.fit(X_train, y_train)
```

---

## 3. Bayesian Hyperparameter Optimization with Optuna

Instead of fixed, slow grid searches, **Optuna** uses Tree-structured Parzen Estimators (TPE) to efficiently find optimal multi-dimensional hyperparameters.

**Implementation Recipe for XGBoost Tuning:**
```python
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 400, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }
    model = XGBClassifier(**params, eval_metric='logloss', random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(f"[✓] Best Params: {study.best_params}")
```

---

## 4. Explainable AI (XAI) with SHAP

In medical applications, black-box predictions are unacceptable. **SHAP** values compute the additive contribution of each clinical feature to a specific patient's predicted risk score:

$$f(x) = \phi_0 + \sum_{j=1}^M \phi_j(x)$$

**Implementation Recipe:**
```python
import shap
explainer = shap.TreeExplainer(rf_model.model)
shap_values = explainer(X_test)

# Global importance summary plot
shap.summary_plot(shap_values, X_test)

# Single-patient waterfall explanation plot
shap.plots.waterfall(shap_values[0])
```

---

## 5. Docker Containerization & Deployment

To run the entire system in an isolated production environment:

```bash
# Build and start container
docker-compose -f docker/docker-compose.yml up --build

# Open browser at:
http://localhost:8501
```
