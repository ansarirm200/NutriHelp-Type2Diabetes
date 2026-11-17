from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NutriHelp Type 2 Diabetes Enhancement",
    page_icon="🩺",
    layout="wide",
)


# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(
        180deg,
        #f7fbff 0%,
        #f1faf8 100%
    );
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.hero-card {
    background: linear-gradient(
        135deg,
        #176b87 0%,
        #2f9e9a 55%,
        #62c6b8 100%
    );
    padding: 32px 36px;
    border-radius: 22px;
    color: white;
    box-shadow: 0 10px 28px rgba(0,0,0,0.12);
    margin-bottom: 20px;
}

.hero-title {
    color: white;
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 10px 0;
}

.hero-subtitle {
    color: white;
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0 0 10px 0;
}

.hero-text {
    color: white;
    font-size: 1rem;
    line-height: 1.6;
    margin: 0;
}

.prototype-box {
    background: #eaf4ff;
    color: #164e73;
    border-left: 6px solid #3787d8;
    padding: 18px 20px;
    border-radius: 14px;
    min-height: 125px;
    box-shadow: 0 5px 14px rgba(0,0,0,0.05);
}

.framework-box {
    background: white;
    color: #243746;
    border-left: 6px solid #2f9e9a;
    padding: 18px 20px;
    border-radius: 14px;
    min-height: 125px;
    box-shadow: 0 5px 14px rgba(0,0,0,0.05);
}

.section-title {
    font-size: 1.55rem;
    font-weight: 800;
    color: #164e73;
    margin-top: 10px;
    margin-bottom: 12px;
}

.model-card {
    background: white;
    padding: 18px 20px;
    border-radius: 16px;
    border-left: 6px solid #2f9e9a;
    box-shadow: 0 5px 16px rgba(0,0,0,0.06);
    margin-bottom: 14px;
}

.metric-card {
    background: white;
    text-align: center;
    padding: 16px 10px;
    border-radius: 15px;
    box-shadow: 0 5px 14px rgba(0,0,0,0.06);
    margin-bottom: 10px;
}

.metric-label {
    color: #667985;
    font-size: 0.9rem;
}

.metric-value {
    color: #164e73;
    font-size: 1.35rem;
    font-weight: 800;
    margin-top: 5px;
}

.result-healthy {
    background: linear-gradient(
        135deg,
        #dff7e9 0%,
        #effcf4 100%
    );
    border-left: 8px solid #22a66f;
    padding: 24px;
    border-radius: 18px;
    color: #126a46;
    box-shadow: 0 6px 16px rgba(0,0,0,0.06);
    margin-top: 14px;
    margin-bottom: 14px;
}

.result-diabetes {
    background: linear-gradient(
        135deg,
        #fff0dd 0%,
        #fff8ec 100%
    );
    border-left: 8px solid #f39c12;
    padding: 24px;
    border-radius: 18px;
    color: #814907;
    box-shadow: 0 6px 16px rgba(0,0,0,0.06);
    margin-top: 14px;
    margin-bottom: 14px;
}

.result-title {
    font-size: 1.45rem;
    font-weight: 800;
    margin-bottom: 10px;
}

.safety-box {
    background: #fff7df;
    color: #59450e;
    border-left: 6px solid #e8ac00;
    padding: 18px 20px;
    border-radius: 14px;
    margin-top: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}

.responsible-box {
    background: white;
    color: #263942;
    border-left: 6px solid #176b87;
    padding: 20px 24px;
    border-radius: 16px;
    box-shadow: 0 5px 16px rgba(0,0,0,0.05);
}

.footer-text {
    text-align: center;
    color: #77858c;
    font-size: 0.88rem;
    padding: 12px;
}

div.stButton > button {
    background: linear-gradient(
        135deg,
        #ff5964 0%,
        #ff7b54 100%
    );
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.70rem 1.6rem;
    box-shadow: 0 5px 14px rgba(255,89,100,0.28);
}

div.stButton > button:hover {
    color: white;
    border: none;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SVM_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "svm_binary_diabetes.joblib"
)

LOGISTIC_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "logistic_regression_binary_diabetes.joblib"
)

RF_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_binary_diabetes.joblib"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model(model_path):
    saved_object = joblib.load(model_path)
    return saved_object["pipeline"]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def metric_card(label, value):
    html = (
        '<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def healthy_result():
    html = (
        '<div class="result-healthy">'
        '<div class="result-title">'
        '✓ Healthy / No Diabetes Group'
        '</div>'
        '<div>'
        'Based on the information entered, the selected '
        'machine-learning model classified this profile '
        'into the Healthy / No Diabetes study group.'
        '</div>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def diabetes_result():
    html = (
        '<div class="result-diabetes">'
        '<div class="result-title">'
        '⚠ Diabetes Risk Group'
        '</div>'
        '<div>'
        'Based on the information entered, the selected '
        'machine-learning model classified this profile '
        'into the Diabetes study group. This is a research '
        'classification and is not a clinical diagnosis.'
        '</div>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

MODEL_SUMMARY = {
    "Support Vector Machine (SVM)": {
        "Accuracy": "75.1%",
        "Balanced Accuracy": "60.8%",
        "Sensitivity": "38.1%",
        "Specificity": "83.6%",
        "ROC-AUC": "0.629",
        "Note": (
            "Strongest overall performance in the "
            "initial model comparison."
        ),
    },

    "Logistic Regression": {
        "Accuracy": "62.8%",
        "Balanced Accuracy": "58.9%",
        "Sensitivity": "52.8%",
        "Specificity": "65.0%",
        "ROC-AUC": "0.624",
        "Note": (
            "Highest sensitivity for identifying "
            "the Diabetes group."
        ),
    },

    "Random Forest": {
        "Accuracy": "72.0%",
        "Balanced Accuracy": "56.1%",
        "Sensitivity": "30.8%",
        "Specificity": "81.4%",
        "ROC-AUC": "0.588",
        "Note": (
            "Good overall accuracy and specificity, "
            "with lower sensitivity."
        ),
    },
}


# =========================================================
# HERO HEADER
# =========================================================

hero_html = (
    '<div class="hero-card">'
    '<div class="hero-title">'
    'NutriHelp Type 2 Diabetes Enhancement'
    '</div>'
    '<div class="hero-subtitle">'
    'AI Research Prototype for Diabetes Risk Classification'
    '</div>'
    '<div class="hero-text">'
    'Exploring machine-learning classification of '
    '<b>Healthy / No Diabetes</b> and '
    '<b>Diabetes</b> groups using demographic, lifestyle '
    'and health-related information.'
    '</div>'
    '</div>'
)

st.markdown(
    hero_html,
    unsafe_allow_html=True,
)


# =========================================================
# INTRODUCTION CARDS
# =========================================================

intro1, intro2 = st.columns(
    [1.15, 1],
    gap="large",
)


with intro1:

    prototype_html = (
        '<div class="prototype-box">'
        '<b>🔬 Research Prototype</b><br><br>'
        'This application is intended for research, '
        'education and demonstration. It is designed '
        'to explore AI-based diabetes risk classification '
        'and is not a clinical diagnostic system.'
        '</div>'
    )

    st.markdown(
        prototype_html,
        unsafe_allow_html=True,
    )


with intro2:

    framework_html = (
        '<div class="framework-box">'
        '<b>📋 Current Modelling Framework</b><br><br>'
        'Healthy / No Diabetes: <b>HbA1c &lt; 6.0%</b><br>'
        'Diabetes: <b>HbA1c &gt; 6.5%</b><br>'
        'HbA1c 6.0–6.5% is excluded from the current '
        'binary AI training experiment.'
        '</div>'
    )

    st.markdown(
        framework_html,
        unsafe_allow_html=True,
    )


# =========================================================
# STUDY FRAMEWORK
# =========================================================

with st.expander(
    "📘 Study classification framework and modelling assumptions"
):

    st.markdown(
        """
### Binary modelling definition

- **Healthy / No Diabetes:** HbA1c < 6.0%
- **Diabetes:** HbA1c > 6.5%
- **HbA1c 6.0–6.5%:** excluded from the current binary
  machine-learning experiment

### Outcome definition

HbA1c is used to establish the study outcome category.

Therefore, **HbA1c is deliberately excluded from the
predictor variables** to avoid target leakage.

The original `diabetes` field in the source dataset is
also excluded from the predictor variables.

### Current predictors

- Gender
- Age
- Hypertension
- Heart disease
- Smoking history
- BMI
- Blood glucose level
"""
    )


# =========================================================
# MAIN INPUT AND MODEL SELECTION AREA
# =========================================================

left_col, right_col = st.columns(
    [1.2, 0.9],
    gap="large",
)


with left_col:

    st.markdown(
        '<div class="section-title">'
        '🧾 Enter Health Information'
        '</div>',
        unsafe_allow_html=True,
    )

    input1, input2 = st.columns(2)

    with input1:

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male",
                "Other",
            ],
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50,
            step=1,
        )

        smoking_history = st.selectbox(
            "Smoking history",
            [
                "never",
                "No Info",
                "current",
                "former",
                "ever",
                "not current",
            ],
        )

        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=10.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
        )


    with input2:

        hypertension_text = st.selectbox(
            "Hypertension",
            [
                "No",
                "Yes",
            ],
        )

        heart_disease_text = st.selectbox(
            "Heart disease",
            [
                "No",
                "Yes",
            ],
        )

        blood_glucose_level = st.number_input(
            "Blood glucose level",
            min_value=40,
            max_value=500,
            value=100,
            step=1,
        )


with right_col:

    st.markdown(
        '<div class="section-title">'
        '🤖 Select AI Model'
        '</div>',
        unsafe_allow_html=True,
    )

    model_choice = st.selectbox(
        "Machine-learning algorithm",
        [
            "Support Vector Machine (SVM)",
            "Logistic Regression",
            "Random Forest",
        ],
    )

    selected_metrics = MODEL_SUMMARY[
        model_choice
    ]

    model_html = (
        '<div class="model-card">'
        f'<b>{model_choice}</b><br><br>'
        f'{selected_metrics["Note"]}'
        '</div>'
    )

    st.markdown(
        model_html,
        unsafe_allow_html=True,
    )

    mc1, mc2 = st.columns(2)

    with mc1:

        metric_card(
            "Accuracy",
            selected_metrics["Accuracy"],
        )

        metric_card(
            "Sensitivity",
            selected_metrics["Sensitivity"],
        )

        metric_card(
            "ROC-AUC",
            selected_metrics["ROC-AUC"],
        )


    with mc2:

        metric_card(
            "Balanced Accuracy",
            selected_metrics[
                "Balanced Accuracy"
            ],
        )

        metric_card(
            "Specificity",
            selected_metrics[
                "Specificity"
            ],
        )


# =========================================================
# PREPARE INPUT
# =========================================================

hypertension = (
    1
    if hypertension_text == "Yes"
    else 0
)

heart_disease = (
    1
    if heart_disease_text == "Yes"
    else 0
)


input_data = pd.DataFrame(
    {
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [
            smoking_history
        ],
        "bmi": [bmi],
        "blood_glucose_level": [
            blood_glucose_level
        ],
    }
)


# =========================================================
# CLASSIFICATION
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🔎 AI Classification'
    '</div>',
    unsafe_allow_html=True,
)


if st.button(
    "Run Prediction",
    type="primary",
):

    try:

        probability = None
        decision_score = None
        svm_threshold = None


        # -------------------------------------------------
        # SVM
        # -------------------------------------------------

        if model_choice == (
            "Support Vector Machine (SVM)"
        ):

            pipeline = load_model(
                SVM_MODEL_PATH
            )

            decision_score = (
                pipeline.decision_function(
                    input_data
                )[0]
            )

            svm_threshold = -0.50

            if decision_score >= svm_threshold:
                prediction = 1
            else:
                prediction = 0


        # -------------------------------------------------
        # LOGISTIC REGRESSION
        # -------------------------------------------------

        elif model_choice == (
            "Logistic Regression"
        ):

            pipeline = load_model(
                LOGISTIC_MODEL_PATH
            )

            prediction = int(
                pipeline.predict(
                    input_data
                )[0]
            )

            probability = (
                pipeline.predict_proba(
                    input_data
                )[0][1]
            )


        # -------------------------------------------------
        # RANDOM FOREST
        # -------------------------------------------------

        else:

            pipeline = load_model(
                RF_MODEL_PATH
            )

            prediction = int(
                pipeline.predict(
                    input_data
                )[0]
            )

            probability = (
                pipeline.predict_proba(
                    input_data
                )[0][1]
            )


        # -------------------------------------------------
        # DISPLAY SIMPLE RESULT
        # -------------------------------------------------

        if prediction == 0:
            healthy_result()
        else:
            diabetes_result()


        result1, result2 = st.columns(2)

        with result1:

            clean_model_name = (
                model_choice.replace(
                    " (SVM)",
                    "",
                )
            )

            metric_card(
                "Selected AI Model",
                clean_model_name,
            )


        with result2:

            if prediction == 0:

                metric_card(
                    "AI Classification",
                    "Healthy / No Diabetes",
                )

            else:

                metric_card(
                    "AI Classification",
                    "Diabetes Risk Group",
                )


        # -------------------------------------------------
        # TECHNICAL DETAILS
        # -------------------------------------------------

        with st.expander(
            "🔬 Technical Model Details"
        ):

            st.write(
                f"**Selected model:** "
                f"{model_choice}"
            )

            if model_choice == (
                "Support Vector Machine (SVM)"
            ):

                st.write(
                    "**Internal SVM decision score:** "
                    f"{decision_score:.3f}"
                )

                st.write(
                    "**Experimental SVM operating "
                    "threshold:** "
                    f"{svm_threshold:.2f}"
                )

                st.info(
                    """
The SVM decision score is an internal
machine-learning value. It is **not a
probability or clinical risk percentage**.
"""
                )

            else:

                st.write(
                    "**Model-estimated class "
                    "probability:** "
                    f"{probability:.1%}"
                )

                st.info(
                    """
This percentage is a machine-learning model
output and should not be interpreted as an
individual clinical probability of diabetes.
"""
                )


            st.markdown(
                "#### Experimental model performance"
            )

            tech_table = pd.DataFrame(
                {
                    "Metric": [
                        "Accuracy",
                        "Balanced Accuracy",
                        "Sensitivity",
                        "Specificity",
                        "ROC-AUC",
                    ],

                    "Result": [
                        selected_metrics[
                            "Accuracy"
                        ],

                        selected_metrics[
                            "Balanced Accuracy"
                        ],

                        selected_metrics[
                            "Sensitivity"
                        ],

                        selected_metrics[
                            "Specificity"
                        ],

                        selected_metrics[
                            "ROC-AUC"
                        ],
                    ],
                }
            )

            st.dataframe(
                tech_table,
                hide_index=True,
                use_container_width=True,
            )


        safety_html = (
            '<div class="safety-box">'
            '<b>⚠ Important:</b> '
            'This output is a machine-learning '
            'research classification. It does '
            '<b>not diagnose diabetes</b> and '
            'should not be used as a substitute '
            'for laboratory testing or '
            'professional clinical advice.'
            '</div>'
        )

        st.markdown(
            safety_html,
            unsafe_allow_html=True,
        )


    except FileNotFoundError:

        st.error(
            """
The selected trained model could not be found.

Please ensure the corresponding `.joblib`
model file exists inside the `models` folder.
"""
        )


    except Exception as error:

        st.error(
            f"Prediction could not be completed: "
            f"{error}"
        )


# =========================================================
# MODEL COMPARISON
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '📊 Initial Model Comparison'
    '</div>',
    unsafe_allow_html=True,
)


comparison_df = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Support Vector Machine (SVM)",
        ],

        "Accuracy": [
            0.628,
            0.720,
            0.751,
        ],

        "Balanced Accuracy": [
            0.589,
            0.561,
            0.608,
        ],

        "Sensitivity": [
            0.528,
            0.308,
            0.381,
        ],

        "Specificity": [
            0.650,
            0.814,
            0.836,
        ],

        "ROC-AUC": [
            0.624,
            0.588,
            0.629,
        ],
    }
)


st.dataframe(
    comparison_df,
    hide_index=True,
    use_container_width=True,
)


st.caption(
    """
SVM demonstrated the strongest overall initial
classification performance, while Logistic Regression
provided the highest sensitivity for identifying the
Diabetes group.
"""
)


# =========================================================
# RESPONSIBLE AI
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '🛡️ Responsible AI and Clinical Safety'
    '</div>',
    unsafe_allow_html=True,
)


responsible_html = (
    '<div class="responsible-box">'
    '<b>This research prototype:</b>'
    '<ul>'
    '<li>does <b>not diagnose diabetes</b>;</li>'
    '<li>does <b>not prescribe treatment</b>;</li>'
    '<li>does <b>not recommend medication changes</b>;</li>'
    '<li>does <b>not replace HbA1c or other laboratory '
    'testing</b>;</li>'
    '<li>does <b>not replace a GP, endocrinologist, '
    'diabetes educator, dietitian or other qualified '
    'health professional</b>.</li>'
    '</ul>'
    'Anyone concerned about diabetes should seek '
    'appropriate clinical assessment and laboratory '
    'testing.'
    '</div>'
)

st.markdown(
    responsible_html,
    unsafe_allow_html=True,
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

footer_html = (
    '<div class="footer-text">'
    'NutriHelp Type 2 Diabetes Enhancement'
    ' &nbsp; | &nbsp; '
    'Independent AI Research Prototype'
    '</div>'
)

st.markdown(
    footer_html,
    unsafe_allow_html=True,
)
