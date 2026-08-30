# Project 3.1: Self-Assessment Rubric & Evaluation Checklist

Use this checklist to track your progress and verify that all theoretical, algorithmic, and engineering requirements are fulfilled.

---

## 📋 Evaluation Rubric

| Domain | Required Knowledge & Implementation Tasks | Status | Self-Check Notes |
| :--- | :--- | :---: | :--- |
| **I. Domain & Data Context** | - Understand the global cardiovascular clinical context and diagnostic motivation.<br>- Master the 14 attributes in the Cleveland dataset ($N=303$).<br>- Binarize target variable (`y > 0 -> 1`, `y == 0 -> 0`). | [ ] | Check `src/data/loader.py` |
| **II. Preprocessing & Feature Engineering** | - Implement `SimpleImputer` (median for numeric, mode for categorical).<br>- Build strict `ColumnTransformer` (fit on train ONLY to prevent data leakage).<br>- Create domain-specific age ratios (`chol_per_age`, `bps_per_age`, `hr_ratio`).<br>- Rank features via **Mutual Information** and **Decision Tree MDI**.<br>- Produce 4 dataset variants: `Original`, `FE`, `Original + DT`, `FE + DT`. | [ ] | Check `src/features/` |
| **III. Machine Learning Classifiers** | - **Naive Bayes:** Gaussian likelihood with conditional independence.<br>- **KNN:** Stratified 5-Fold CV search for $K \in [1, 20]$.<br>- **Decision Tree:** Stratified CV search for `max_depth` $\in [2, 10]$ and feature importance analysis.<br>- **K-Means:** $k=2$ unsupervised clustering with majority label mapping.<br>- **Stacking Ensemble:** Base models (KNN, DT, NB) with KNN meta-classifier. | [ ] | Check `src/models/base_models.py` |
| **IV. Advanced Ensemble Learning** | - **Random Forest:** Bagging + Feature Randomness ($m=\sqrt{p}$) to minimize variance.<br>- **AdaBoost:** Sequential error-weight boosting with Decision Stumps.<br>- **Gradient Boosting:** Sequential residual gradient descent with learning rate.<br>- **XGBoost:** 2nd-order Taylor objective with $L_1/L_2$ regularization and histogram splitting. | [ ] | Check `src/models/ensemble_models.py` |
| **V. Experiments & Rigorous Evaluation** | - Stratified dataset split ($80:10:10$) with `SEED=42`.<br>- Measure Accuracy, Precision, Recall, F1-score across all 4 dataset variants.<br>- Consolidate master benchmark comparison table (target $\approx 97\%$ Val, $\approx 90\%$ Test). | [ ] | Check `notebooks/03_ensemble_advancement.ipynb` |
| **VI. UI & Cloud Deployment** | - Build interactive Streamlit application (`app/app.py`).<br>- Implement `@st.cache_resource` for instant response.<br>- Construct sidebar input controls (`st.slider` for numeric, `st.selectbox` for categorical).<br>- Render Decision Tree visualization (`plot_tree`) and predicted risk percentages.<br>- Deploy public tunnel via Cloudflared on Colab (`app/deploy_colab.sh`). | [ ] | Check `app/` |
| **VII. Improvements & Extensions (Honors)** | - Test `RobustScaler`, `VarianceThreshold`, and `PolynomialFeatures`.<br>- Implement `SupportVectorMachineModel` (Linear & RBF) and `LogisticRegressionModel`.<br>- Implement `MultiLayerPerceptronModel` with regularization.<br>- Bayesian Hyperparameter Optimization with **Optuna**.<br>- Explainable AI with **SHAP** summary & waterfall plots.<br>- Containerize using **Docker** & **docker-compose**. | [ ] | Check `docs/IMPROVEMENTS_GUIDE.md` & `notebooks/04_model_improvements.ipynb` |
