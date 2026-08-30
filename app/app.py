"""
Interactive Streamlit Web Application for Heart Disease Diagnosis.

Features:
1. Form sidebar for real-time patient clinical parameter entry.
2. Cached model pipeline trained via GridSearchCV (using @st.cache_resource).
3. Risk probability calculation and threshold alert (Disease Likely vs Disease Unlikely).
4. Decision Tree rule visualization using matplotlib and `plot_tree`.
5. [IMPROVEMENT]: SHAP force plot / feature attribution breakdown.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.config import ALL_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, SEED, TARGET, UCI_CLEVELAND_URL

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Heart Disease Diagnosis AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# 2. CACHED MODEL TRAINING PIPELINE
# ==========================================
@st.cache_resource
def train_best_decision_tree_pipeline():
    """
    Download raw data, construct complete preprocessing + Decision Tree pipeline,
    and find optimal max_depth via GridSearchCV.

    HINT:
        1. df = pd.read_csv(UCI_CLEVELAND_URL, header=None, names=ALL_COLUMNS, na_values='?').dropna()
        2. df[TARGET] = (df[TARGET] > 0).astype(int)
        3. X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
        4. y = df[TARGET]
        5. Build num_pipe (SimpleImputer median) & cat_pipe (SimpleImputer most_frequent + OneHotEncoder)
        6. Build ColumnTransformer and Pipeline with DecisionTreeClassifier(random_state=SEED)
        7. Run GridSearchCV with param_grid={'classifier__max_depth': range(3, 11)}, cv=5, scoring='accuracy'
        8. Return fitted grid_search object
    """
    # TODO [USER IMPLEMENTATION]:
    # Construct and fit cached GridSearchCV pipeline
    raise NotImplementedError("Implement `train_best_decision_tree_pipeline`.")


# ==========================================
# 3. APPLICATION UI LAYOUT
# ==========================================
def main():
    st.title("❤️ Cleveland Heart Disease Diagnosis Expert System")
    st.markdown(
        """
        This clinical decision support application predicts the likelihood of coronary artery disease 
        based on demographic attributes, resting clinical indicators, and diagnostic exercise test results.
        """
    )

    try:
        model = train_best_decision_tree_pipeline()
        best_depth = model.best_params_["classifier__max_depth"]
        st.sidebar.success(f"⚡ Optimal Decision Tree Depth: **{best_depth}** (CV Score: {model.best_score_:.2%})")
    except Exception as e:
        st.sidebar.warning(f"Pipeline placeholder active. Implement training logic to load model. Error: {e}")
        model = None

    # Sidebar Clinical Parameter Form
    with st.sidebar.form("patient_input_form"):
        st.header("📋 Patient Clinical Parameters")

        # Numeric sliders
        age = st.slider("Age (years)", min_value=20, max_value=85, value=55)
        trestbps = st.slider("Resting Blood Pressure (trestbps in mmHg)", min_value=80, max_value=220, value=130)
        chol = st.slider("Serum Cholesterol (chol in mg/dl)", min_value=100, max_value=600, value=240)
        thalach = st.slider("Max Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
        oldpeak = st.slider("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)

        st.markdown("---")
        # Categorical selectboxes
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox(
            "Chest Pain Type (cp)",
            options=[
                ("Typical Angina (1)", 1),
                ("Atypical Angina (2)", 2),
                ("Non-anginal Pain (3)", 3),
                ("Asymptomatic (4)", 4),
            ],
            format_func=lambda x: x[0],
        )[1]
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl (fbs)",
            options=[("False (<= 120)", 0), ("True (> 120)", 1)],
            format_func=lambda x: x[0],
        )[1]
        restecg = st.selectbox(
            "Resting ECG (restecg)",
            options=[("Normal (0)", 0), ("ST-T Wave Abnormality (1)", 1), ("Left Ventricular Hypertrophy (2)", 2)],
            format_func=lambda x: x[0],
        )[1]
        exang = st.selectbox(
            "Exercise Induced Angina (exang)",
            options=[("No (0)", 0), ("Yes (1)", 1)],
            format_func=lambda x: x[0],
        )[1]
        slope = st.selectbox(
            "Slope of Peak Exercise ST (slope)",
            options=[("Upsloping (1)", 1), ("Flat (2)", 2), ("Downsloping (3)", 3)],
            format_func=lambda x: x[0],
        )[1]
        ca = st.slider("Number of Major Vessels Colored (ca: 0-3)", min_value=0, max_value=3, value=0)
        thal = st.selectbox(
            "Thalassemia (thal)",
            options=[("Normal (3)", 3), ("Fixed Defect (6)", 6), ("Reversible Defect (7)", 7)],
            format_func=lambda x: x[0],
        )[1]

        submitted = st.form_submit_button("🩺 Run Diagnosis Prediction", use_container_width=True)

    # Prediction & Visual Analysis
    if submitted:
        if model is None:
            st.error("Please complete the model training function in `train_best_decision_tree_pipeline()` first!")
            return

        input_data = pd.DataFrame(
            [
                {
                    "age": age,
                    "sex": sex,
                    "cp": cp,
                    "trestbps": trestbps,
                    "chol": chol,
                    "fbs": fbs,
                    "restecg": restecg,
                    "thalach": thalach,
                    "exang": exang,
                    "oldpeak": oldpeak,
                    "slope": slope,
                    "ca": ca,
                    "thal": thal,
                }
            ]
        )

        # TODO [USER IMPLEMENTATION]:
        # 1. Compute predicted probabilities: `proba = model.predict_proba(input_data)[0]`
        # 2. Extract `prob_disease = proba[1]`
        # 3. Render st.error if prob_disease > 0.5 else st.success
        # 4. Display Decision Tree plot using `plot_tree(model.best_estimator_.named_steps['classifier'])`
        st.info("Form submitted. Complete prediction display logic in `app/app.py`.")


if __name__ == "__main__":
    main()
