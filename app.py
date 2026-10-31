import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide",
)

st.markdown("""
<style>
.result-card {
    padding: 24px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid;
}
.benign {
    border-color: #2ecc71;
    background: rgba(46, 204, 113, 0.12);
}
.malignant {
    border-color: #e74c3c;
    background: rgba(231, 76, 60, 0.12);
}
</style>
""", unsafe_allow_html=True)

DISCLAIMER = (
    "⚠️ Educational/research use only. This model is not a medical "
    "diagnostic tool and must not be used to make healthcare decisions."
)

MODEL_PATH = Path(__file__).with_name("knn_breast_cancer_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Put 'knn_breast_cancer_model.pkl' "
        "in the same folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error("The model could not be loaded. Check that the model file is valid.")
    st.exception(exc)
    st.stop()

names = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave points", "symmetry",
    "fractal dimension",
]

defaults = {
    "Mean": [14, 19, 90, 650, 0.096, 0.10, 0.08, 0.05, 0.18, 0.063],
    "Error": [0.4, 1.2, 2.8, 40, 0.007, 0.02, 0.03, 0.01, 0.02, 0.004],
    "Worst": [16, 25, 105, 800, 0.13, 0.25, 0.15, 0.10, 0.25, 0.08],
}
icons = {"Mean": "📏", "Error": "📊", "Worst": "🔬"}

def feature_name(group, name):
    if group == "Mean":
        return f"mean {name}"
    if group == "Error":
        return f"{name} error"
    return f"worst {name}"

features = [
    feature_name(group, name)
    for group in ("Mean", "Error", "Worst")
    for name in names
]

st.title("🩺 Breast Cancer Prediction")
st.caption(
    "Enter the 30 tumor measurements, then click **Predict**. "
    "The model was trained on the Wisconsin Diagnostic Breast Cancer dataset."
)
st.warning(DISCLAIMER)

values = {}
for group, group_defaults in defaults.items():
    st.markdown("---")
    st.subheader(f"{icons[group]} {group} Characteristics")
    cols = st.columns(3)
    for i, (name, default) in enumerate(zip(names, group_defaults)):
        feature = feature_name(group, name)
        values[feature] = cols[i % 3].number_input(
            feature.title(),
            min_value=0.0,
            value=float(default),
            step=0.01 if default >= 1 else 0.001,
            format="%.2f" if default >= 1 else "%.4f",
            key=feature,
        )

st.markdown("---")

if st.button("🔍 Predict", use_container_width=True, type="primary"):
    try:
        input_df = pd.DataFrame(
            [[values[feature] for feature in features]],
            columns=features,
        )

        # Reorder columns to exactly match the order used during training.
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
            missing = [f for f in expected_features if f not in input_df.columns]
            extra = [f for f in input_df.columns if f not in expected_features]
            if missing or extra:
                st.error(
                    "The input features do not match the trained model. "
                    f"Missing: {missing}; unexpected: {extra}"
                )
                st.stop()
            input_df = input_df[expected_features]

        prediction = int(model.predict(input_df)[0])
        class_names = {0: ("Malignant", "🔴", "malignant"),
                       1: ("Benign", "🟢", "benign")}

        if prediction not in class_names:
            st.error(
                f"Unexpected model class: {prediction}. "
                "Expected 0 = Malignant and 1 = Benign."
            )
            st.stop()

        diagnosis, result_icon, css_class = class_names[prediction]
        st.header("Prediction Result")
        st.markdown(
            f"""
            <div class="result-card {css_class}">
                <h1>{result_icon} {diagnosis}</h1>
                <p>The model classified the entered measurements as
                <b>{diagnosis}</b>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if hasattr(model, "predict_proba"):
            st.subheader("📊 Model Probability Estimates")
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_
            cols = st.columns(len(classes))
            for col, class_label, probability in zip(cols, classes, probabilities):
                label = class_names.get(int(class_label), (f"Class {class_label}", "", ""))[0]
                col.metric(label, f"{float(probability) * 100:.2f}%")
                col.progress(float(probability))

        with st.expander("📋 View Entered Measurements"):
            st.dataframe(
                input_df.T.rename(columns={0: "Value"}),
                use_container_width=True,
            )

        st.caption(DISCLAIMER)

    except Exception as exc:
        st.error(
            "Prediction failed. Check that the model file, feature names, "
            "preprocessing, and training configuration match."
        )
        st.exception(exc)
