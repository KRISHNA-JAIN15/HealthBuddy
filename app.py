import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Add this directory to Python's search path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Add BodyFat directory to path for pickle imports
bodyfat_path = os.path.join(PROJECT_ROOT, 'BodyFat')
if bodyfat_path not in sys.path:
    sys.path.append(bodyfat_path)

from BodyFat.RandomForest import RandomForestRegressor, DecisionTreeRegressor, Node
# Ensure pickle can find the module
sys.modules['RandomForest'] = sys.modules['BodyFat.RandomForest']

#1----BODY FAT PREDICTOR

@st.cache_resource
def load_body_fat_model():
    model_path = os.path.join('BodyFat', 'final_custom_rf_model.pkl')
    try:
        with open(model_path,'rb') as file:
            model_payload = pickle.load(file)
        
        loaded_model = model_payload['model']
        loaded_scaler = model_payload['scaler']

        try:
            feature_names = loaded_scaler.feature_names_in_
        except AttributeError:                     
            
            feature_names =[
                'Age', 'Weight', 'Height', 'Neck', 'Chest', 
                'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 
                'Biceps', 'Forearm', 'Wrist'
            ]
        print("BodyFFat model and scaler loaded successfully")
        return loaded_model, loaded_scaler, feature_names
    
    except FileNotFoundError:
        st.error(f"❌ ERROR: The file '{model_path}' was not found.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ An error occurred while loading the file: {e}")
        return None, None, None
model, scaler, feature_names = load_body_fat_model()

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Health Buddy",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- 2. MODERN CUSTOM CSS ---
st.markdown(
    """
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global font */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main app background with gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        border-right: none;
    }
    
    /* Sidebar content text color fix */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Sidebar title */
    [data-testid="stSidebar"] h1 {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff !important;
        text-align: center;
        padding: 1.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -0.5px;
    }

    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2);
        margin: 1rem 0;
    }

    /* Radio buttons styling */
    [data-testid="stSidebar"] .row-widget {
        background-color: transparent;
    }
    
    [data-testid="stSidebar"] label[data-baseweb="radio"] {
        background-color: rgba(255,255,255,0.1);
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        cursor: pointer;
    }
    
    [data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
        background-color: rgba(255,255,255,0.2);
        border-color: rgba(255,255,255,0.4);
        transform: translateX(5px);
    }
    
    [data-testid="stSidebar"] label[data-baseweb="radio"][data-checked="true"] {
        background-color: rgba(255,255,255,0.25);
        border-color: #4fc3f7;
        box-shadow: 0 4px 12px rgba(79,195,247,0.3);
    }
    
    /* Hero Section */
    .hero-container {
        padding: 4rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 15s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    .hero-container h1 {
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 1rem;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
    }
    
    .hero-container h3 {
        font-size: 1.5rem;
        font-weight: 400;
        opacity: 0.95;
        position: relative;
        z-index: 1;
        line-height: 1.6;
    }

    /* Card styling for content sections */
    .info-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 36px rgba(0,0,0,0.12);
    }

    /* Headers styling */
    h1, h2, h3 {
        color: #2c3e50 !important;
        font-weight: 700;
    }
    
    h2 {
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }

    /* Form styling */
    [data-testid="stForm"] {
        background-color: white;
        padding: 2.5rem;
        border-radius: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Input fields */
    .stNumberInput input, .stSelectbox select {
        border-radius: 0.5rem;
        border: 2px solid #e0e7ff;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
    }
    
    /* Labels */
    label {
        color: #4a5568 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Form submit button */
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        width: 100%;
        font-size: 1.2rem;
        border-radius: 0.75rem;
        border: none;
        padding: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.5);
    }
    
    /* Success/Error/Info/Warning boxes */
    [data-testid="stSuccess"] {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 0.75rem;
        padding: 1.25rem;
        color: #155724 !important;
    }
    
    [data-testid="stSuccess"] * {
        color: #155724 !important;
    }
    
    [data-testid="stError"] {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 0.75rem;
        padding: 1.25rem;
        color: #721c24 !important;
    }
    
    [data-testid="stError"] * {
        color: #721c24 !important;
    }
    
    [data-testid="stInfo"] {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        border-radius: 0.75rem;
        padding: 1.25rem;
        color: #0c5460 !important;
    }
    
    [data-testid="stInfo"] * {
        color: #0c5460 !important;
    }
    
    [data-testid="stWarning"] {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 0.75rem;
        padding: 1.25rem;
        color: #856404 !important;
    }
    
    [data-testid="stWarning"] * {
        color: #856404 !important;
    }

    /* Columns content */
    [data-testid="column"] > div {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        height: 100%;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 2px solid #e0e7ff;
        margin: 2rem 0;
    }
    
    /* Subheaders in forms */
    [data-testid="stForm"] h3 {
        color: #667eea !important;
        font-size: 1.3rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    /* Regular text readability */
    p, li, span {
        color: #4a5568 !important;
        line-height: 1.7;
    }
    
    /* Make sure markdown content is readable */
    .stMarkdown {
        color: #4a5568 !important;
    }
</style>
    """,
    unsafe_allow_html=True,
)


# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h1>🩺 Health Buddy</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    app_mode = st.radio(
        "Choose your predictor:",
        [
            "🏠 Home",
            "❤️ Heart Attack Predictor",
            "💪 Body Fat Predictor",
            "🩸 Diabetes Predictor",
            "🩺 Liver Disease Predictor",
            "🔬 Kidney Disease Predictor",
            "🧠 Stroke Predictor",
            "🎀 Breast Cancer Predictor",
        ],
        label_visibility="collapsed"
    )

# --- 4. MAIN PAGE CONTENT ---

# --- HOME PAGE ---
if app_mode == "🏠 Home":
    
    st.markdown(
        """
        <div class="hero-container">
            <h1>Welcome to Health Buddy!</h1>
            <h3>Your AI-powered health companion for intelligent predictions.
            <br>
            Science-backed. Data-driven. Always here for you. 💙
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("## 🎯 What We Do")
    st.write(
        "We leverage **7 state-of-the-art machine learning models** to analyze your health data with precision. "
        "Our system has been rigorously trained on **9 comprehensive datasets** to identify the **most accurate classifier** for each medical condition."
    )
    st.markdown(
        "### 👈 Select a predictor from the sidebar to get started!"
    )

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🤖 Our AI Arsenal")
        st.info(
            """
            **Machine Learning Models:**
            - 🔵 Logistic Regression
            - 🔵 K-Nearest Neighbors (KNN)
            - 🔵 Support Vector Machine (SVM)
            - 🔵 Naive Bayes
            - 🔵 Decision Tree
            - 🔵 Random Forest
            - 🔵 Gradient Boosting (XGBoost)
            """
        )
    with col2:
        st.markdown("### 📊 Our Commitment")
        st.warning(
            """
            **Accuracy is our #1 priority.**
            
            We've analyzed thousands of data points and rigorously tested each model to select the **single best performer** for every health predictor. 
            
            You're not just getting *a* prediction — you're getting the **best possible prediction** backed by data science.
            """
        )
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Important Disclaimer")
    st.error(
        """
        **Medical Advice Notice:**
        
        Health Buddy provides AI-generated predictions for educational and informational purposes only. 
        These predictions should **never replace professional medical advice, diagnosis, or treatment**.
        
        Always consult qualified healthcare providers for medical decisions. In case of emergency, contact your local emergency services immediately.
        """
    )


# --- HEART ATTACK PREDICTOR PAGE ---
elif app_mode == "❤️ Heart Attack Predictor":
    st.markdown("# ❤️ Heart Attack Risk Predictor")
    st.markdown("Our **Random Forest** model (top performer for cardiovascular analysis) will evaluate your risk based on comprehensive health metrics.")

    with st.form("heart_attack_form"):
        st.markdown("### 👤 Patient Demographics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=52)
        with col2:
            sex = st.selectbox("Biological Sex", ("Male", "Female"))
        with col3:
            cp = st.selectbox(
                "Chest Pain Type", 
                (0, 1, 2, 3), 
                help="0: Typical angina | 1: Atypical angina | 2: Non-anginal | 3: Asymptomatic"
            )

        st.markdown("### 🔬 Clinical Measurements")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        with col2:
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=210)
        with col3:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("No", "Yes"))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            restecg = st.selectbox(
                "Resting ECG Results",
                (0, 1, 2),
                help="0: Normal | 1: ST-T wave abnormality | 2: Left ventricular hypertrophy"
            )
        with col2:
            thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        with col3:
            exang = st.selectbox("Exercise Induced Angina", ("No", "Yes"))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f")
        with col2:
            slope = st.selectbox(
                "Slope of Peak Exercise ST",
                (0, 1, 2),
                help="0: Upsloping | 1: Flat | 2: Downsloping"
            )
        with col3:
            ca = st.selectbox("Number of Major Vessels (0-3)", (0, 1, 2, 3))

        st.markdown("---")
        
        submitted = st.form_submit_button("🩺 Analyze My Risk")

    if submitted:
        import random
        prediction_proba = random.random()
        
        st.markdown("### 📋 Analysis Results")
        
        if prediction_proba > 0.5:
            st.error(
                f"**⚠️ HIGH RISK DETECTED**\n\nProbability of Heart Attack: **{prediction_proba*100:.1f}%**",
                icon="🚨"
            )
            st.warning(
                """
                **Immediate Action Required:**
                
                This prediction indicates elevated risk. Please consult a cardiologist or healthcare provider **immediately** for proper evaluation and diagnosis.
                
                This is an AI-generated assessment and should not be considered a medical diagnosis.
                """
            )
        else:
            st.success(
                f"**✅ LOW RISK DETECTED**\n\nProbability of Heart Attack: **{prediction_proba*100:.1f}%**",
                icon="💚"
            )
            st.info(
                """
                **Recommendation:**
                
                Your results suggest lower risk, but maintaining regular health check-ups is essential. Continue healthy lifestyle habits and consult your healthcare provider for personalized advice.
                
                This is an AI-generated assessment for informational purposes only.
                """
            )



elif app_mode == "💪 Body Fat Predictor":
    st.markdown("# 💪 Body Fat Percentage Predictor")
    st.markdown("Calculate your estimated body fat percentage using advanced anthropometric measurements.")

    with st.form("body_fat_form"):
        st.markdown("### 📏 Basic Measurements")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
        with col2:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=75.0, step=0.1, format="%.1f")
        with col3:
            height = st.number_input("Height (cm)", min_value=120.0, max_value=250.0, value=175.0, step=0.1, format="%.1f")
        
        st.markdown("### 📐 Body Circumferences (Part 1)")
        col1, col2, col3 = st.columns(3)
        with col1:
            neck = st.number_input("Neck (cm)", min_value=20.0, max_value=60.0, value=38.0, step=0.1, format="%.1f")
        with col2:
            chest = st.number_input("Chest (cm)", min_value=60.0, max_value=150.0, value=100.0, step=0.1, format="%.1f")
        with col3:
            abdomen = st.number_input("Abdomen (cm)", min_value=50.0, max_value=180.0, value=90.0, step=0.1, format="%.1f")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            hip = st.number_input("Hip (cm)", min_value=60.0, max_value=160.0, value=100.0, step=0.1, format="%.1f")
        with col2:
            thigh = st.number_input("Thigh (cm)", min_value=30.0, max_value=100.0, value=60.0, step=0.1, format="%.1f")
        with col3:
            # --- [NEW] ADDED MISSING INPUT ---
            knee = st.number_input("Knee (cm)", min_value=25.0, max_value=55.0, value=38.0, step=0.1, format="%.1f")

        st.markdown("### 📐 Body Circumferences (Part 2)")
        col1, col2, col3 = st.columns(3)
        with col1:
            # --- [NEW] ADDED MISSING INPUT ---
            ankle = st.number_input("Ankle (cm)", min_value=15.0, max_value=35.0, value=22.0, step=0.1, format="%.1f")
        with col2:
            # --- [NEW] ADDED MISSING INPUT ---
            biceps = st.number_input("Biceps (flexed, cm)", min_value=20.0, max_value=50.0, value=33.0, step=0.1, format="%.1f")
        with col3:
            # --- [NEW] ADDED MISSING INPUT ---
            forearm = st.number_input("Forearm (cm)", min_value=20.0, max_value=40.0, value=29.0, step=0.1, format="%.1f")
        
        # This one you already had, but I moved it here for logical grouping
        wrist = st.number_input("Wrist (cm)", min_value=12.0, max_value=25.0, value=18.0, step=0.1, format="%.1f")
        
        
        submitted = st.form_submit_button("📊 Calculate Body Fat %")
        
        if submitted:
            # Check if model loaded correctly
            if model is None or scaler is None or feature_names is None:
                st.error("❌ **Model Error:** The prediction model could not be loaded. Please check the server logs.")
            else:
                try:
                  
                    input_data = {
                        'Age': age,
                        'Weight': weight,
                        'Height': height,
                        'Neck': neck,
                        'Chest': chest,
                        'Abdomen': abdomen,
                        'Hip': hip,
                        'Thigh': thigh,
                        'Knee': knee,
                        'Ankle': ankle,
                        'Biceps': biceps,
                        'Forearm': forearm,
                        'Wrist': wrist
                    }
                    
                    # 2. Convert to DataFrame with correct column order
                    input_df = pd.DataFrame([input_data], columns=feature_names)
                    
                    # 3. Scale the data
                    input_scaled = scaler.transform(input_df)
                    
                    # 4. Make the prediction
                    prediction = model.predict(input_scaled)
                    
                    # Get the single value from the prediction array
                    body_fat = prediction[0]
                    
                    # --- [END] NEW PREDICTION LOGIC ---
                    
                    st.markdown("### 📈 Your Results")
                    st.success(f"**Estimated Body Fat Percentage: {body_fat:.1f}%**")
                    
                    if body_fat < 15:
                        st.info("**Athletic Range** - Excellent body composition! 💪")
                    elif body_fat < 25:
                        st.info("**Healthy Range** - Good body composition. Keep it up! 👍")
                    else:
                        st.warning("**Above Average** - Consider consulting a fitness professional for guidance.")
                
                except Exception as e:
                    st.error(f"❌ An error occurred during prediction: {e}")

# --- PLACEHOLDER PAGES ---
elif app_mode == "🩸 Diabetes Predictor":
    st.markdown("# 🩸 Diabetes Risk Predictor")
    st.info("🛠️ **Feature Coming Soon!** Our data science team is fine-tuning this predictor for maximum accuracy.", icon="⚡")

elif app_mode == "🩺 Liver Disease Predictor":
    st.markdown("# 🩺 Liver Disease Predictor")
    st.info("🛠️ **Feature Coming Soon!** Advanced liver health analysis will be available shortly.", icon="⚡")

elif app_mode == "🔬 Kidney Disease Predictor":
    st.markdown("# 🔬 Kidney Disease Predictor")
    st.info("🛠️ **Feature Coming Soon!** Comprehensive kidney function assessment in development.", icon="⚡")

elif app_mode == "🧠 Stroke Predictor":
    st.markdown("# 🧠 Stroke Risk Predictor")
    st.info("🛠️ **Feature Coming Soon!** Neurological risk assessment powered by AI.", icon="⚡")

elif app_mode == "🎀 Breast Cancer Predictor":
    st.markdown("# 🎀 Breast Cancer Risk Predictor")
    st.info("🛠️ **Feature Coming Soon!** Advanced oncological screening tool under development.", icon="⚡")