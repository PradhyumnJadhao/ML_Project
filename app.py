import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved model and scaler from the files
try:
    with open('heart_disease_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model or scaler file not found. Make sure they are in the same directory.")
    st.stop()

# --- Page Configuration ---
# THIS IS THE CHANGE: layout="centered" constrains the app to a comfortable width
st.set_page_config(page_title="Heart Disease Dashboard", layout="centered")

# --- UI and Input Fields ---
st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("Enter patient medical data below to get a real-time prediction. This app is for educational purposes only.")

st.divider()

# --- Create a form for all inputs ---
with st.form(key='patient_data_form'):
    st.subheader("Enter Patient Medical Data")

    # --- All inputs are now in a single vertical stack ---
    
    # --- User-enterable number input for 'age' ---
    age = st.number_input('Age 👨‍⚕️', min_value=1, max_value=120, value=50, step=1)
    
    # --- Scrolldown selectbox for 'sex' ---
    sex = st.selectbox('Sex 🚻', ('Male', 'Female'))
    
    # --- Scrolldown selectbox for 'cp' ---
    cp = st.selectbox('Chest Pain Type 💔', ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
    
    # --- User-enterable number input for 'trestbps' ---
    trestbps = st.number_input('Resting Blood Pressure (mm Hg) 🩺', min_value=80, max_value=220, value=120, step=1)
    
    # --- User-enterable number input for 'chol' ---
    chol = st.number_input('Serum Cholestoral (mg/dl) 🔬', min_value=100, max_value=600, value=200, step=1)
    
    # --- Scrolldown selectbox for 'fbs' ---
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl 🩸', ('False', 'True'))
    
    # --- Scrolldown selectbox for 'restecg' ---
    restecg = st.selectbox('Resting ECG Results 📈', ('Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'))

    # --- User-enterable number input for 'thalach' ---
    thalach = st.number_input('Max Heart Rate Achieved ❤️‍🩹', min_value=60, max_value=220, value=150, step=1)

    # --- Scrolldown selectbox for 'exang' ---
    exang = st.selectbox('Exercise Induced Angina 🏃‍♂️', ('No', 'Yes'))
    
    # --- User-enterable number input for 'oldpeak' ---
    oldpeak = st.number_input('ST depression (oldpeak) 📉', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # --- Scrolldown selectbox for 'slope' ---
    slope = st.selectbox('Slope of peak exercise ST segment ⛰️', ('Upsloping', 'Flat', 'Downsloping'))
    
    # --- Scrolldown selectbox for 'ca' ---
    ca = st.selectbox('Number of major vessels (ca) 🪢', (0, 1, 2, 3, 4))
    
    # --- Scrolldown selectbox for 'thal' ---
    thal = st.selectbox('Thalassemia (thal) 🧬', ('Null', 'Fixed defect', 'Normal', 'Reversable defect'))
    
    st.divider()
    # --- Form submit button ---
    submitted = st.form_submit_button('**Get Prediction**', help="Click to run the model")


# --- Prediction Logic (runs only after form submission) ---
if submitted:
    
    # 1. Convert text inputs to the numerical format the model expects
    sex_num = 1 if sex == 'Male' else 0
    fbs_num = 1 if fbs == 'True' else 0
    exang_num = 1 if exang == 'Yes' else 0
    
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_map = {'Null': 0, 'Fixed defect': 1, 'Normal': 2, 'Reversable defect': 3}
    
    # 2. Create the DataFrame for a single prediction
    data = {'age': age, 'sex': sex_num, 'cp': cp_map[cp], 'trestbps': trestbps, 'chol': chol, 'fbs': fbs_num, 
            'restecg': restecg_map[restecg], 'thalach': thalach, 'exang': exang_num, 'oldpeak': oldpeak, 
            'slope': slope_map[slope], 'ca': ca, 'thal': thal_map[thal]}
            
    input_df = pd.DataFrame(data, index=[0])

    # 3. Prepare the input for the model
    # One-hot encode the input dataframe
    training_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex_1', 'cp_1', 'cp_2', 'cp_3', 'fbs_1', 'restecg_1', 'restecg_2', 'exang_1', 'slope_1', 'slope_2', 'ca_1', 'ca_2', 'ca_3', 'ca_4', 'thal_1', 'thal_2', 'thal_3']
    processed_input = pd.get_dummies(input_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'], drop_first=True)
    
    # Align the columns of the input with the training columns
    final_input, _ = processed_input.align(pd.DataFrame(columns=training_cols), join='right', axis=1, fill_value=0)

    # 4. Scale the continuous features
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    final_input[continuous_features] = scaler.transform(final_input[continuous_features])
    
    # 5. Make prediction
    prediction = model.predict(final_input)
    prediction_proba = model.predict_proba(final_input)
    
    # 6. Display the result in a more attractive layout
    st.subheader('Prediction Result')
    res_col1, res_col2 = st.columns([2, 1]) # Make the first column wider

    with res_col1:
        if prediction[0] == 1:
            st.error("### **High Risk of Heart Disease**")
            st.write("The model predicts a high probability that this patient has heart disease. Please consult a medical professional for a formal diagnosis.")
        else:
            st.success("### **Low Risk of Heart Disease**")
            st.write("The- model predicts a low probability that this patient has heart disease. Continue to maintain a healthy lifestyle.")
    
    with res_col2:
        if prediction[0] == 1:
            prob_percent = prediction_proba[0][1] * 100
            st.metric(label="Risk Probability", value=f"{prob_percent:.2f}%", delta="High Risk", delta_color="inverse")
        else:
            prob_percent = prediction_proba[0][0] * 100
            st.metric(label="Safety Probability", value=f"{prob_percent:.2f}%", delta="Low Risk", delta_color="normal")
    