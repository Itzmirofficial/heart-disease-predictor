# Heart Disease Risk Predictor

This web application predicts the likelihood of heart disease in a patient based on clinical parameters. It uses a machine learning model trained on the **UCI Cleveland Heart Disease Dataset** and provides explainable AI insights using **SHAP**.

---

## 🩺 Project Overview

The Heart Disease Risk Predictor is a user-friendly web app built with **Streamlit** that allows users to:

- Enter key clinical attributes (age, chest pain type, cholesterol, etc.)
- Receive a probability-based risk prediction
- View a model explanation (SHAP waterfall chart)
- Understand which features contributed most to the outcome

---

## ⚙️ Technologies Used

| Layer         | Stack                                                      |
|---------------|------------------------------------------------------------|
| Frontend      | Streamlit (Python-based interactive UI)                    |
| Backend       | Python (RandomForestClassifier from Scikit-learn)          |
| Machine Learning | Random Forest Classifier                               |
| Explainable AI| SHAP (SHapley Additive exPlanations)                       |
| Data Source   | UCI Cleveland Heart Disease Dataset                        |
| Visualization | Matplotlib                                                 |
| Hosting       | Localhost or Streamlit Cloud                               |

---

## 🚀 Getting Started

### 🔧 1. Clone the repository

```bash
git clone https://github.com/yourusername/heart-disease-predictor.git
cd heart-disease-predictor
```

---

### 📦 2. Install required packages

#### Option 1: Using `requirements.txt` (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Option 2: Manual installation

```bash
pip install streamlit pandas numpy matplotlib scikit-learn shap
```

---

### ▶️ 3. Run the Streamlit app

```bash
streamlit run app.py
```

Visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Example Inputs

You’ll be prompted to enter values such as:

- Age  
- Sex  
- Chest Pain Type  
- Resting BP  
- Cholesterol  
- Fasting Blood Sugar  
- Max Heart Rate  
- Exercise-induced Angina  
- and other clinical metrics

Based on your inputs, the app predicts your **heart disease risk level** and displays a probability score.

---

## 🔍 Explainability with SHAP

The model’s predictions are supported with **SHAP waterfall charts** to explain feature contributions — making the prediction **transparent, interpretable, and trustworthy**.

---

## 📎 Dataset Information

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Filename:** `processed.cleveland.data`
- **Features:** 13
- **Samples:** 303

---

## 🧑‍💻 Developed By

**Mir Abdul Aziz Khan**  


---

## 💡 Future Enhancements

- Add login and session history for users  
- Export prediction results as PDF  
- Store predictions in Google Sheets or a secure database  
- Deploy live using [Streamlit Cloud](https://share.streamlit.io)

---
