import streamlit as st  
import joblib  
import numpy as np  
import pandas as pd  
import shap  
import matplotlib.pyplot as plt  
from lime.lime_tabular import LimeTabularExplainer  

# Load the model  
model = joblib.load('RF.pkl')  

# Load test data  
X_test = pd.read_csv('X_test.csv')  

# Define feature names  
feature_names = [  
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",  
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"  
]  

# Streamlit user interface  
st.title("Heart Disease Predictor")  

# Collect user input  
age = st.number_input("Age:", min_value=0, max_value=120, value=41)  
sex = st.selectbox("Sex:", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")  
cp = st.selectbox("Chest Pain Type (CP):", options=[0, 1, 2, 3])  
trestbps = st.number_input("Resting Blood Pressure (trestbps):", min_value=50, max_value=200, value=120)  
chol = st.number_input("Cholesterol (chol):", min_value=100, max_value=600, value=157)  
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (FBS):", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")  
restecg = st.selectbox("Resting ECG (restecg):", options=[0, 1, 2])  
thalach = st.number_input("Maximum Heart Rate Achieved (thalach):", min_value=60, max_value=220, value=182)  
exang = st.selectbox("Exercise Induced Angina (exang):", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")  
oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)  
slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope):", options=[0, 1, 2])  
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca):", options=[0, 1, 2, 3, 4])  
thal = st.selectbox("Thalassemia (thal):", options=[0, 1, 2, 3])  

# Predictive modeling  
if st.button("Predict"):  
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])  

    # Predictions  
    predicted_class = model.predict(features)[0]  
    predicted_proba = model.predict_proba(features)[0]  
    probability = predicted_proba[predicted_class] * 100  

    # Display prediction results  
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")  
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  

    # Generate advice  
    if predicted_class == 1:  
        advice = (  
            f"According to our model, you have a high risk of heart disease. "  
            f"The probability of having heart disease is {probability:.1f}%. "  
            "Consult a healthcare provider for further evaluation."  
        )  
    else:  
        advice = (  
            f"You have a low risk of heart disease. "  
            f"The probability of being healthy is {probability:.1f}%. "  
            "Maintain a healthy lifestyle and regular check-ups."  
        )  
    st.write(advice)  
    
    # SHAP Explanation  
    st.subheader("SHAP Force Plot Explanation")  
    explainer_shap = shap.TreeExplainer(model)  
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))  
    shap.force_plot(explainer_shap.expected_value[predicted_class], shap_values[predicted_class], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)  
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)  
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')  

    # LIME Explanation  
    st.subheader("LIME Explanation")  
    lime_explainer = LimeTabularExplainer(  
        training_data=X_test.values,  
        feature_names=feature_names,  
        class_names=['Not sick', 'Sick'],  # Adjust class names to match your classification task  
        mode='classification'  
    )  
    lime_exp = lime_explainer.explain_instance(  
        data_row=features.flatten(), predict_fn=model.predict_proba  
    )  
    lime_html = lime_exp.as_html(show_table=False)  # Disable feature value table  
    st.components.v1.html(lime_html, height=800, scrolling=True)
