import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt


# Load and prepare UCI Heart Disease dataset

column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'target'
]

df = pd.read_csv('processed.cleveland.data', names=column_names)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

X = df.drop('target', axis=1)
y = df['target']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI Setup

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide")

# Title and Header
st.title("Heart Disease Risk Predictor")
st.markdown("This tool predicts the likelihood of heart disease using the UCI Cleveland dataset and a Random Forest Classifier.")
st.markdown("---")


# User Input Form

def user_input():
    age = st.slider("Age", 29, 77, 50)
    sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ][x]
    )

    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)

    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    restecg = st.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ][x]
    )

    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)

    exang = st.radio("Exercise-Induced Angina?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)

    slope = st.selectbox("Slope of Peak ST Segment", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])

    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])

    thal = st.selectbox(
        "Thalassemia",
        options=[1, 2, 3],
        format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x - 1]
    )

    user_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
        'ca': ca, 'thal': thal
    }

    return pd.DataFrame(user_data, index=[0])

# Input Section
with st.expander("Enter Your Health Information", expanded=True):
    input_df = user_input()


# Prediction

input_df = input_df[X.columns]
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][1]


# Display Results

st.markdown("---")
st.subheader("Prediction Summary")
col1, col2 = st.columns(2)
col1.metric("Risk Detected", "Yes" if prediction == 1 else "No")
col2.metric("Estimated Probability", f"{proba*100:.2f}%" if prediction == 1 else f"{(1-proba)*100:.2f}%")

if prediction == 1:
    st.warning("You may be at risk of heart disease. Please consult a medical professional.")
    with st.expander("Recommendations"):
        st.markdown("""
        - Consult a cardiologist  
        - Maintain a heart-healthy diet  
        - Engage in regular physical activity  
        - Avoid smoking and limit alcohol  
        - Practice stress-reducing activities
        """)
else:
    st.success("No immediate signs of heart disease detected.")
    st.info("Maintain regular checkups and a healthy lifestyle.")


# SHAP Explanation 

st.markdown("---")
if st.checkbox("Show SHAP Explanation"):
    with st.spinner("Generating model explanation..."):
        explainer = shap.Explainer(model, X)
        shap_values = explainer(input_df)
        st.subheader("Feature Contribution to Prediction")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0, :, 1], max_display=13, show=False)
        st.pyplot(fig)

# Sidebar Info & Credits

with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This application predicts the risk of heart disease using data from the UCI Cleveland dataset and machine learning.
    """)
    st.markdown("### Source")
    st.write("[UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)")
    st.markdown("---")
    st.markdown("Developed by **Mir Abdul Aziz Khan**")
