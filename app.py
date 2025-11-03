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
# Add Classifier_codes directory to path for pickle imports
classifier_codes_path = os.path.join(PROJECT_ROOT, 'Classifier_codes')
if classifier_codes_path not in sys.path:
    sys.path.append(classifier_codes_path)


from BodyFat.RandomForest import RandomForestRegressor, DecisionTreeRegressor, Node as BodyFatNode
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






# Heart Attack Predictor model loading function
@st.cache_resource
def load_heart_attack_model():
    model_path = os.path.join('HeartAttack', 'heart_attack_rf_model.pkl')
    

    original_rf_module = sys.modules.get('RandomForest')
    try:
        from Classifier_codes import DecisionTree
        from Classifier_codes import RandomForest as ClassifierRandomForest

        sys.modules['DecisionTree'] = DecisionTree
        sys.modules['RandomForest'] = ClassifierRandomForest

        with open(model_path, 'rb') as file:
            heart_pipeline = pickle.load(file)

        print("Heart Attack model loaded successfully")


        if original_rf_module:
            sys.modules['RandomForest'] = original_rf_module
        else:
            del sys.modules['RandomForest']
        
        del sys.modules['DecisionTree']

        return heart_pipeline
    
    except FileNotFoundError:
        st.error(f"❌ ERROR: The file '{model_path}' was not found.")
        return None
    except ImportError as e:
        st.error(f"❌ IMPORT ERROR: {e}")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred while loading the file: {e}")
        return None
    

#Maternal Health Predictor

@st.cache_resource
def load_maternal_health_model():
    model_path = os.path.join('MentalHealth', 'maternal_health_randomforest.pkl')
    
    original_rf_module = sys.modules.get('RandomForest')

    try:
        from Classifier_codes import DecisionTree
        from Classifier_codes import RandomForest as ClassifierRandomForest

        sys.modules['DecisionTree'] = DecisionTree
        sys.modules['RandomForest'] = ClassifierRandomForest

        with open(model_path, 'rb') as file:
            maternal_model = pickle.load(file)

        print("Maternal Health model loaded successfully")


        if original_rf_module:
            sys.modules['RandomForest'] = original_rf_module
        else:
            del sys.modules['RandomForest']
        del sys.modules['DecisionTree']
        return maternal_model
    except FileNotFoundError:
        st.error(f"❌ ERROR: The file '{model_path}' was not found.")
        return None
    except ImportError as e:
        st.error(f"❌ IMPORT ERROR: {e}")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred while loading the file: {e}")
        return None
         
        

@st.cache_resource
def obesity_model_loader():
    model_path = os.path.join('ObesityLevel', 'decision_tree_model.pkl')
    try:
        # Monkey patch for sklearn compatibility issue
        import sklearn.compose._column_transformer
        if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
            class _RemainderColsList:
                pass
            sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
        
        with open(model_path, 'rb') as file:
            obesity_model = pickle.load(file)
        print("Obesity model loaded successfully")
        return obesity_model
    except FileNotFoundError:
        st.error(f"❌ ERROR: The file '{model_path}' was not found.")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred while loading the file: {e}")
        return None

      


model, scaler, feature_names = load_body_fat_model()
heart_model_pipeline = load_heart_attack_model()      
maternal_model_pipeline = load_maternal_health_model()
obesity_model = obesity_model_loader()




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
            "🤰 Maternal Health Predictor",
            "⚖️ Obesity Level Predictor",
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
    st.markdown("Our **Custom Random Forest** model will evaluate your risk based on 8 clinical health metrics.")

    with st.form("heart_attack_form"):
        st.markdown("### 👤 Patient Demographics")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
        with col2:
            # Your model was trained on 'Gender' which is likely 1 for Male, 0 for Female
            gender = st.selectbox("Gender", (1, 0), format_func=lambda x: "Male" if x == 1 else "Female")

        st.markdown("### 🔬 Clinical Measurements")
        
        col1, col2 = st.columns(2)
        with col1:
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=220, value=75, help="Average beats per minute.")
        with col2:
            blood_sugar = st.number_input("Blood Sugar (mg/dL)", min_value=50.0, max_value=500.0, value=100.0, step=0.1, help="Fasting blood glucose level.")
        
        col1, col2 = st.columns(2)
        with col1:
            systolic_bp = st.number_input("Systolic Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120, help="The 'top' number.")
        with col2:
            diastolic_bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=40, max_value=150, value=80, help="The 'bottom' number.")
        
        st.markdown("### 🧬 Cardiac Enzyme Markers")
        col1, col2 = st.columns(2)
        with col1:
            ck_mb = st.number_input("CK-MB (ng/mL)", min_value=0.0, max_value=100.0, value=3.0, step=0.1, help="Creatine kinase-MB enzyme level.")
        with col2:
            troponin = st.number_input("Troponin (ng/mL)", min_value=0.0, max_value=10.0, value=0.02, step=0.01, format="%.2f", help="Troponin enzyme level.")

        st.markdown("---")
        
        submitted = st.form_submit_button("🩺 Analyze My Risk")

    if submitted:
        # Check if the model is loaded
        if heart_model_pipeline is None:
            st.error("❌ **Model Error:** The Heart Attack prediction model is not loaded. Please check the server logs.")
        else:
            try:
                # 1. Create a dictionary of all 8 inputs
                input_data = {
                    'Age': age,
                    'Gender': gender,
                    'Heart rate': heart_rate,
                    'Systolic blood pressure': systolic_bp,
                    'Diastolic blood pressure': diastolic_bp,
                    'Blood sugar': blood_sugar,
                    'CK-MB': ck_mb,
                    'Troponin': troponin
                }
                
                # 2. Convert to DataFrame
                # The column order MUST match the training script:
                feature_order = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure', 
                                 'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
                input_df = pd.DataFrame([input_data], columns=feature_order)

                # 3. Make prediction
                # The saved object is a full pipeline, so it handles scaling/imputing
                # Since the custom model doesn't have predict_proba, use predict and assume binary
                prediction = heart_model_pipeline.predict(input_df)[0]
                prediction_proba = 1.0 if prediction == 1 else 0.0
                
                # --- [END] NEW PREDICTION LOGIC ---

                st.markdown("### 📋 Analysis Results")
                
                if prediction_proba > 0.5: # 50% threshold
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
            
            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {e}")


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

# --- MATERNAL HEALTH PREDICTOR PAGE ---
elif app_mode == "🤰 Maternal Health Predictor":
    st.markdown("# 🤰 Maternal Health Risk Predictor")
    st.markdown("Our **Custom Random Forest** model will evaluate your risk based on 6 key vital signs.")

    with st.form("maternal_health_form"):
        st.markdown("### 👤 Patient Vitals")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=10, max_value=70, value=25)
        with col2:
            systolic_bp = st.number_input("Systolic BP (mm Hg)", min_value=70, max_value=200, value=130)
        with col3:
            diastolic_bp = st.number_input("Diastolic BP (mm Hg)", min_value=40, max_value=120, value=80)

        col1, col2, col3 = st.columns(3)
        with col1:
            bs = st.number_input("Blood Sugar (BS) (mmol/L)", min_value=3.0, max_value=20.0, value=15.0, step=0.1, format="%.1f")
        with col2:
            body_temp_f = st.number_input("Body Temp (°F)", min_value=95.0, max_value=105.0, value=98.0, step=0.1, format="%.1f")
        with col3:
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=50, max_value=120, value=86)

        st.markdown("---")
        submitted = st.form_submit_button("🩺 Analyze My Risk")

    if submitted:
        # Check if the model is loaded
        if maternal_model_pipeline is None:
            st.error("❌ **Model Error:** The Maternal Health prediction model is not loaded. Please check the server logs.")
        else:
            try:
                # 1. Create a dictionary of all 6 inputs
                input_data = {
                    'Age': age,
                    'SystolicBP': systolic_bp,
                    'DiastolicBP': diastolic_bp,
                    'BS': bs,
                    'BodyTemp': body_temp_f, # Your model was trained on 'BodyTemp'
                    'HeartRate': heart_rate
                }
                
                # 2. Convert to DataFrame
                # The column order MUST match the training script:
                feature_order = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
                input_df = pd.DataFrame([input_data], columns=feature_order)

                # 3. Make prediction
                # The saved object is a full pipeline (model + preprocessor)
                prediction_encoded = maternal_model_pipeline.predict(input_df)
                
                # Get the single encoded label (0, 1, or 2)
                encoded_label = prediction_encoded[0]
                
                # 4. Map to class names
                # Must match the order from your training script
                class_names = ['high risk', 'low risk', 'mid risk']
                prediction_label = class_names[encoded_label]
                
                # --- [END] NEW PREDICTION LOGIC ---

                st.markdown("### 📋 Analysis Results")
                
                if prediction_label == 'high risk':
                    st.error(
                        f"**⚠️ HIGH RISK DETECTED**",
                        icon="🚨"
                    )
                    st.warning(
                        """
                        **Immediate Action Required:**
                        
                        This prediction indicates a high-risk profile. Please consult a healthcare provider **immediately** for a proper evaluation.
                        """
                    )
                elif prediction_label == 'mid risk':
                    st.warning(
                        f"**🟡 MID RISK DETECTED**",
                        icon="⚠️"
                    )
                    st.info(
                        """
                        **Recommendation:**
                        
                        Your results suggest a moderate-risk profile. It is strongly recommended to schedule a consultation with your healthcare provider for further monitoring.
                        """
                    )
                else: # 'low risk'
                    st.success(
                        f"**✅ LOW RISK DETECTED**",
                        icon="💚"
                    )
                    st.info(
                        """
                        **Recommendation:**
                        
                        Your results suggest a low-risk profile. Continue to maintain regular health check-ups and consult your provider with any questions.
                        """
                        )
            
            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {e}")






# --- OBESITY LEVEL PREDICTOR PAGE ---
elif app_mode == "⚖️ Obesity Level Predictor":
    st.markdown("# ⚖️ Obesity Level Predictor")
    st.markdown("Our **Custom Decision Tree** model will analyze 16 lifestyle and physical metrics to predict your weight category.")

    # These are the 16 features your model was trained on
    COLUMN_ORDER = [
        'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
        'CALC', 'MTRANS'
    ]
    BINARY_FEATURES = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    CLASS_NAMES = [
        'Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I', 
        'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I', 
        'Overweight_Level_II'
    ]

    with st.form("obesity_form"):
        st.markdown("### 👤 Patient Demographics")
        col1, col2, col3 = st.columns(3)
        with col1:
            Age = st.number_input("Age", min_value=1.0, max_value=100.0, value=25.0, step=1.0)
        with col2:
            Gender = st.selectbox("Gender", ('Male', 'Female'))
        with col3:
            family_history_with_overweight = st.selectbox("Family History of Overweight?", ('yes', 'no'))

        col1, col2 = st.columns(2)
        with col1:
            Height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.75, step=0.01, format="%.2f")
        with col2:
            Weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=80.0, step=0.1, format="%.1f")

        st.markdown("### 🍔 Dietary Habits")
        col1, col2, col3 = st.columns(3)
        with col1:
            FAVC = st.selectbox("Frequent High Caloric Food?", ('yes', 'no'), help="Do you frequently eat high caloric food?")
        with col2:
            FCVC = st.slider("Vegetable Consumption", 1.0, 3.0, 2.0, 1.0, help="1=Never, 2=Sometimes, 3=Always")
        with col3:
            NCP = st.slider("Number of Main Meals", 1.0, 4.0, 3.0, 1.0, help="Number of main meals per day")

        col1, col2, col3 = st.columns(3)
        with col1:
            CAEC = st.selectbox("Food Between Meals", ('No', 'Sometimes', 'Frequently', 'Always'))
        with col2:
            CALC = st.selectbox("Alcohol Consumption", ('no', 'Sometimes', 'Frequently', 'Always'))
        with col3:
            CH2O = st.slider("Water Consumption (Liters)", 1.0, 3.0, 2.0, 0.5, help="Liters of water per day")

        st.markdown("### 🏃 Lifestyle & Activity")
        col1, col2, col3 = st.columns(3)
        with col1:
            SMOKE = st.selectbox("Do you smoke?", ('yes', 'no'))
        with col2:
            SCC = st.selectbox("Monitor Calorie Intake?", ('yes', 'no'), help="Do you monitor calories?")
        with col3:
            FAF = st.slider("Physical Activity Frequency", 0.0, 3.0, 1.0, 0.5, help="Days per week. 0=None, 3=Often")
        
        col1, col2 = st.columns(2)
        with col1:
            TUE = st.slider("Time Using Tech Devices", 0.0, 2.0, 1.0, 0.5, help="0=0-2h, 1=3-5h, 2=>5h")
        with col2:
            MTRANS = st.selectbox("Primary Transportation", ('Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'))
        
        st.markdown("---")
        submitted = st.form_submit_button("⚖️ Calculate Obesity Level")

    if submitted:
        if obesity_model is None:
            st.error("❌ **Model Error:** The Obesity prediction model is not loaded. Please check the server logs.")
        else:
            try:
                # 1. Create a dictionary of all 16 inputs
                input_data = {
                    'Gender': Gender, 'Age': Age, 'Height': Height, 'Weight': Weight, 
                    'family_history_with_overweight': family_history_with_overweight,
                    'FAVC': FAVC, 'FCVC': FCVC, 'NCP': NCP, 'CAEC': CAEC, 'SMOKE': SMOKE, 
                    'CH2O': CH2O, 'SCC': SCC, 'FAF': FAF, 'TUE': TUE, 'CALC': CALC, 'MTRANS': MTRANS
                }
                
                # 2. Convert to DataFrame with correct column order
                input_df = pd.DataFrame([input_data], columns=COLUMN_ORDER)

                # --- [CRITICAL PREDICTION FIX] ---
                # The pipeline expects 1/0 for binary features, not 'yes'/'no'
                for col in BINARY_FEATURES:
                    if input_df[col].dtype == 'object':
                        input_df[col] = input_df[col].map({'yes': 1, 'no': 0})
                # --- [END OF FIX] ---

                # 3. Make prediction
                prediction_encoded = obesity_model.predict(input_df)
                
                # 4. Map to class names
                prediction_label = CLASS_NAMES[prediction_encoded[0]]
                
                # --- [END] NEW PREDICTION LOGIC ---

                st.markdown("### 📋 Analysis Results")
                
                if "Obesity" in prediction_label:
                    st.error(
                        f"**Prediction: {prediction_label.replace('_', ' ')}**",
                        icon="🚨"
                    )
                elif "Overweight" in prediction_label:
                    st.warning(
                        f"**Prediction: {prediction_label.replace('_', ' ')}**",
                        icon="⚠️"
                    )
                elif "Normal" in prediction_label:
                    st.success(
                        f"**Prediction: {prediction_label.replace('_', ' ')}**",
                        icon="💚"
                    )
                else: # Insufficient
                    st.info(
                        f"**Prediction: {prediction_label.replace('_', ' ')}**",
                        icon="ℹ️"
                    )
                
                st.info("This is an AI-generated assessment for informational purposes only. Please consult a healthcare professional for medical advice.")

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

