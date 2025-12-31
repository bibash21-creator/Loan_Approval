import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

st.title("🏦 Loan Approval Predictor")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Train Model", "Make Prediction"])

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('loan_data.csv')
    return data

@st.cache_resource
def train_models(data):
    data_imputed = data.copy()
    numeric_cols = data_imputed.select_dtypes(include=[np.number]).columns
    categorical_cols = data_imputed.select_dtypes(include=['object']).columns
    
    # Impute categorical
    for col in categorical_cols:
        if data_imputed[col].isnull().sum() > 0:
            mode_vals = data_imputed[col].mode()
            if not mode_vals.empty:
                data_imputed[col].fillna(mode_vals[0], inplace=True)
    
    # Impute numeric
    for col in numeric_cols:
        if data_imputed[col].isnull().sum() > 0:
            skewness = data_imputed[col].skew()
            if abs(skewness) > 1:
                data_imputed[col].fillna(data_imputed[col].median(), inplace=True)
            else:
                data_imputed[col].fillna(data_imputed[col].mean(), inplace=True)
    
    x = data_imputed.drop('Loan_Status', axis=1)
    y = data_imputed['Loan_Status']
    
    # Encode
    label_encoders = {}
    for col in x.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col].astype(str))
        label_encoders[col] = le
    
    y_le = LabelEncoder()
    y = y_le.fit_transform(y.astype(str))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)
    
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        if name in ['LogisticRegression', 'SVM']:
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)
        else:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
        
        results[name] = accuracy_score(y_test, y_pred)
        trained_models[name] = model
    
    return trained_models, label_encoders, sc, results, x.columns

if page == "Home":
    st.subheader("Welcome to Loan Approval Predictor")
    st.write("This application predicts loan approval using machine learning models.")
    st.info("Navigate to 'Train Model' to train models or 'Make Prediction' to predict loan status.")

elif page == "Train Model":
    st.subheader("Train Models")
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            data = load_data()
            trained_models, label_encoders, sc, results, feature_cols = train_models(data)
            st.session_state.trained_models = trained_models
            st.session_state.label_encoders = label_encoders
            st.session_state.scaler = sc
            st.session_state.feature_cols = feature_cols
            
            st.success("Models trained successfully!")
            col1, col2, col3, col4 = st.columns(4)
            for (name, acc), col in zip(results.items(), [col1, col2, col3, col4]):
                col.metric(name, f"{acc:.4f}")

elif page == "Make Prediction":
    st.subheader("Make Prediction")
    
    if 'trained_models' not in st.session_state:
        st.warning("Please train models first!")
    else:
        model_choice = st.selectbox("Select Model", list(st.session_state.trained_models.keys()))
        st.write("Enter applicant details:")
        
        user_input = {}
        data = load_data()
        
        for col in st.session_state.feature_cols:
            if col in data.select_dtypes(include=['object']).columns:
                unique_vals = data[col].dropna().unique()
                user_input[col] = st.selectbox(col, unique_vals)
            else:
                user_input[col] = st.number_input(col, value=0.0)
        
        if st.button("Predict"):
            input_df = pd.DataFrame([user_input])
            
            for col in input_df.select_dtypes(include=['object']).columns:
                if col in st.session_state.label_encoders:
                    input_df[col] = st.session_state.label_encoders[col].transform(input_df[col].astype(str))
            
            if model_choice in ['LogisticRegression', 'SVM']:
                input_scaled = st.session_state.scaler.transform(input_df)
                prediction = st.session_state.trained_models[model_choice].predict(input_scaled)
            else:
                prediction = st.session_state.trained_models[model_choice].predict(input_df)
            
            result = "✅ Loan Approved" if prediction[0] == 1 else "❌ Loan Rejected"
            st.success(result)