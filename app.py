import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import tempfile

import sys

# Define missing classes for pickle compatibility
class OvRWrapper:
    pass

class _RemainderColsList:
    pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

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

@st.cache_resource
def gallstone_model_loader():
    model_path = os.path.join('GallStone', 'gallstone_adaboost_model.pkl')
    try:
        with open(model_path, 'rb') as file:
            gallstone_model = pickle.load(file)
        print("GallStone model loaded successfully")
        return gallstone_model
    except FileNotFoundError:
        st.error(f"❌ ERROR: The file '{model_path}' was not found.")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred while loading the file: {e}")
        return None     

@st.cache_resource
def load_kidney_disease_model():
    # Load the new KNN model
    model_path = os.path.join('ChronicKidneyDisease', 'knn_ckd_model.pkl')
    try:
        with open(model_path, 'rb') as file:
            kidney_pipeline = pickle.load(file)
        print("Kidney Disease (KNN) model loaded successfully")
        return kidney_pipeline
    except Exception as e:
        st.error(f"❌ An error occurred while loading the Kidney Disease model: {e}")
        return None
    
@st.cache_resource
def load_diabetes_model():
    # Load the model from the Diabetes folder
    model_path = os.path.join('Diabetes', 'logreg_ckd_model.pkl')
    try:
        with open(model_path, 'rb') as file:
            diabetes_pipeline = pickle.load(file)
        print("Diabetes (Logistic Regression) model loaded successfully")
        return diabetes_pipeline
    except Exception as e:
        st.error(f"❌ An error occurred while loading the Diabetes model: {e}")
        return None
# Brain Tumor


@st.cache_data
def extract_hog(img):
    hog_features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), block_norm='L2-Hys',
                          visualize=True, transform_sqrt=True)
    return hog_features

@st.cache_data
def extract_glcm_features(img):
    glcm = graycomatrix(img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]
    return features

@st.cache_data
def extract_lbp_features(img):
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, 10),
                             range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

@st.cache_data
def extract_statistical_features(img):
    mean = np.mean(img)
    var = np.var(img)
    skew = np.mean((img - mean)**3) / (np.std(img)**3 + 1e-6)
    kurt = np.mean((img - mean)**4) / (np.std(img)**4 + 1e-6)
    entropy = -np.sum((img/255) * np.log2(img/255 + 1e-6))
    return [mean, var, skew, kurt, entropy]

def extract_features_from_image(img_path):
    """Main preprocessing function for a single image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image {img_path}.")
        return None
    
    img = cv2.resize(img, (128, 128))

    hog_f = extract_hog(img)
    glcm_f = extract_glcm_features(img)
    lbp_f = extract_lbp_features(img)
    stat_f = extract_statistical_features(img)

    combined = np.hstack([hog_f, glcm_f, lbp_f, stat_f])
    return combined

@st.cache_resource
def load_brain_tumor_model():
    model_path = os.path.join('Brain Tumor', 'knn_pipeline.pkl')
    try:
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
        
        # Unpack the dictionary
        bt_model = pipeline['model']
        bt_le = pipeline['label_encoder']
        bt_scaler = pipeline['scaler']
        bt_pca = pipeline['pca']
        bt_class_names = pipeline['class_names']
        
        print("Brain Tumor (KNN) pipeline loaded successfully")
        return bt_model, bt_le, bt_scaler, bt_pca, bt_class_names
    except Exception as e:
        st.error(f"❌ An error occurred while loading the Brain Tumor model: {e}")
        return None, None, None, None, None

model, scaler, feature_names = load_body_fat_model()
heart_model_pipeline = load_heart_attack_model()      
maternal_model_pipeline = load_maternal_health_model()
obesity_model = obesity_model_loader()
gallstone_model = gallstone_model_loader()
kidney_model_pipeline = load_kidney_disease_model()
diabetes_model_pipeline = load_diabetes_model()
bt_model, bt_le, bt_scaler, bt_pca, bt_class_names = load_brain_tumor_model()
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
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@300;400;500;600;700;800&display=swap');

    /* Global reset and base styles */
    * {
        box-sizing: border-box;
    }

    /* Main app container */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
        min-height: 100vh;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar styling - Modern glassmorphism effect */
    [data-testid="stSidebar"] {
        background: linear-gradient(145deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(148,163,184,0.1);
        box-shadow: 4px 0 24px rgba(0,0,0,0.08);
        border-radius: 0 24px 24px 0;
        padding: 2rem 1rem;
    }

    /* Sidebar title - Elegant typography */
    [data-testid="stSidebar"] h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.5px;
        text-shadow: none;
    }

    /* Sidebar navigation buttons - Modern pill design */
    [data-testid="stSidebar"] .row-widget {
        margin: 0.5rem 0;
    }

    [data-testid="stSidebar"] label[data-baseweb="radio"] {
        background: rgba(255,255,255,0.8);
        border: 2px solid rgba(148,163,184,0.2);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        margin: 0.4rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }

    [data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
        background: rgba(59,130,246,0.1);
        border-color: rgba(59,130,246,0.3);
        transform: translateX(8px) scale(1.02);
        box-shadow: 0 8px 25px rgba(59,130,246,0.15);
    }

    [data-testid="stSidebar"] label[data-baseweb="radio"][data-checked="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border-color: #1d4ed8;
        color: white !important;
        box-shadow: 0 8px 25px rgba(59,130,246,0.4);
        transform: translateX(8px);
    }

    [data-testid="stSidebar"] label[data-baseweb="radio"][data-checked="true"]:before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        width: 4px;
        height: 100%;
        background: rgba(255,255,255,0.8);
        border-radius: 0 4px 4px 0;
    }

    /* Hero section - Dramatic and modern */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 24px;
        color: white;
        text-align: center;
        padding: 4rem 3rem;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
        backdrop-filter: blur(20px);
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 20s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translate(-50%, -50%) rotate(0deg); }
        50% { transform: translate(-50%, -50%) rotate(180deg); }
    }

    .hero-container h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 3.8rem;
        font-weight: 900;
        margin-bottom: 1.5rem;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        letter-spacing: -1px;
        position: relative;
        z-index: 2;
    }

    .hero-container h3 {
        font-size: 1.6rem;
        font-weight: 400;
        opacity: 0.95;
        line-height: 1.6;
        position: relative;
        z-index: 2;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Content cards - Clean and professional */
    .content-card {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(148,163,184,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .content-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    }

    /* Form containers - Modern glass effect */
    [data-testid="stForm"] {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 3rem;
        border: 1px solid rgba(148,163,184,0.1);
        box-shadow: 0 12px 40px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }

    /* Form sections */
    .form-section {
        background: rgba(248,250,252,0.8);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(148,163,184,0.1);
    }

    .form-section h3 {
        color: #1e293b !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        margin-bottom: 1.5rem;
        font-size: 1.4rem;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
    }

    /* Input fields - Modern design */
    .stNumberInput input, .stTextInput input {
        border-radius: 12px;
        border: 2px solid #cbd5e1;
        padding: 0.875rem 1rem;
        font-size: 1rem;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: white;
        color: #1e293b;
    }

    /* Specific styling for select dropdowns to ensure contrast */
    .stSelectbox select {
        border-radius: 12px;
        border: 2px solid #cbd5e1;
        padding: 0.875rem 1rem;
        font-size: 1rem;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: #374151 !important;
        color: white !important;
    }

    .stSelectbox select option {
        background: #374151 !important;
        color: white !important;
    }

    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
        background: white;
        outline: none;
    }

    .stSelectbox select:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
        background: #374151;
        outline: none;
    }

    /* Labels - Clean typography */
    label {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
        font-family: 'Inter', sans-serif;
    }

    /* Submit button - Professional single color */
    [data-testid="stFormSubmitButton"] button {
        background: #3b82f6;
        color: black !important;
        font-weight: 700;
        width: 100%;
        font-size: 1.25rem;
        border-radius: 16px;
        border: none;
        padding: 1.25rem 2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(59,130,246,0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'Poppins', sans-serif;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        background: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59,130,246,0.4);
    }

    /* Regular buttons */
    .stButton button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.75rem 3rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
        background: #10b981;
        color: black !important;
        box-shadow: 0 4px 15px rgba(16,185,129,0.3);
    }

    .stButton button:hover {
        background: #059669;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(16,185,129,0.4);
    }

    /* Alert boxes - Modern design */
    [data-testid="stSuccess"] {
        background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(16,185,129,0.05) 100%);
        border: 1px solid rgba(16,185,129,0.2);
        border-left: 4px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        color: #065f46 !important;
        backdrop-filter: blur(10px);
    }

    [data-testid="stError"] {
        background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(239,68,68,0.05) 100%);
        border: 1px solid rgba(239,68,68,0.2);
        border-left: 4px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        color: #991b1b !important;
        backdrop-filter: blur(10px);
    }

    [data-testid="stWarning"] {
        background: linear-gradient(135deg, rgba(245,158,11,0.1) 0%, rgba(245,158,11,0.05) 100%);
        border: 1px solid rgba(245,158,11,0.2);
        border-left: 4px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        color: #92400e !important;
        backdrop-filter: blur(10px);
    }

    [data-testid="stInfo"] {
        background: linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(59,130,246,0.05) 100%);
        border: 1px solid rgba(59,130,246,0.2);
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        color: #1e40af !important;
        backdrop-filter: blur(10px);
    }

    /* Headers - Modern typography */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #1e293b !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #1e293b 0%, #374151 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }

    h2 {
        font-size: 2rem;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.75rem;
        margin-bottom: 2rem;
    }

    h3 {
        font-size: 1.25rem;
        color: #374151 !important;
    }

    /* Columns - Better spacing */
    [data-testid="column"] {
        padding: 0 1rem;
    }

    [data-testid="column"]:first-child {
        padding-left: 0;
    }

    [data-testid="column"]:last-child {
        padding-right: 0;
    }

    /* File uploader - Modern design */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.9);
        border: 2px dashed rgba(148,163,184,0.3);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
        background: rgba(59,130,246,0.02);
    }

    /* Images - Rounded corners */
    .stImage img {
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    /* Progress bars - Modern styling */
    .stProgress > div > div {
        background: #3b82f6;
        border-radius: 10px;
    }

    /* Spinners - Custom colors */
    .stSpinner > div {
        border-color: #3b82f6;
        border-right-color: transparent;
    }

    /* General text improvements */
    p, li, span, div {
        color: #4b5563 !important;
        line-height: 1.7;
        font-weight: 400;
    }

    /* Markdown content */
    .stMarkdown {
        color: #4b5563 !important;
    }

    /* Tables - Modern styling */
    .stTable {
        background: rgba(255,255,255,0.9);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    .stTable table {
        border-collapse: separate;
        border-spacing: 0;
    }

    .stTable th {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        font-weight: 700;
        color: #374151 !important;
        padding: 1rem;
        border-bottom: 2px solid #e5e7eb;
    }

    .stTable td {
        padding: 1rem;
        border-bottom: 1px solid #f3f4f6;
    }

    /* Scrollbars - Modern design */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.05);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }

    /* Responsive design improvements */
    @media (max-width: 768px) {
        .hero-container h1 {
            font-size: 2.5rem;
        }

        .hero-container {
            padding: 2rem 1.5rem;
        }

        [data-testid="stSidebar"] {
            border-radius: 0 16px 16px 0;
        }

        [data-testid="stForm"] {
            padding: 1.5rem;
        }

        .form-section {
            padding: 1rem;
        }
    }
</style>
    """,
    unsafe_allow_html=True,
)


# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h1>Health Buddy</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    app_mode = st.radio(
        "Choose your predictor:",
        [
            "Home",
            "Heart Attack Predictor",
            "Body Fat Predictor",
            "Maternal Health Predictor",
            "Obesity Level Predictor",
            "GallStone Predictor",
            "Diabetes Predictor",
            "Kidney Disease Predictor",
            "Brain Tumor Predictor",
            "Breast Cancer Predictor",
        ],
        label_visibility="collapsed"
    )

# --- 4. MAIN PAGE CONTENT ---

# --- HOME PAGE ---
if app_mode == "Home":
    
    st.markdown(
        """
        <div class="hero-container">
            <h1>Welcome to Health Buddy!</h1>
            <h3>Your AI-powered health companion for intelligent predictions.
            <br>
            Science-backed. Data-driven. Always here for you.
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("## What We Do")
    st.markdown(
        """
        <div class="content-card">
        <p>We leverage <strong>7 state-of-the-art machine learning models</strong> to analyze your health data with precision. 
        Our system has been rigorously trained on <strong>9 comprehensive datasets</strong> to identify the <strong>most accurate classifier</strong> for each medical condition.</p>
        <p><strong>Select a predictor from the sidebar to get started!</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="home-grid">', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="home-card">
        <h3>Our AI Arsenal</h3>
        <p><strong>Machine Learning Models:</strong></p>
        <ul>
        <li>Logistic Regression</li>
        <li>K-Nearest Neighbors (KNN)</li>
        <li>Support Vector Machine (SVM)</li>
        <li>Naive Bayes</li>
        <li>Decision Tree</li>
        <li>Random Forest</li>
        <li>Gradient Boosting (XGBoost)</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div class="home-card">
        <h3>Our Commitment</h3>
        <p><strong>Accuracy is our #1 priority.</strong></p>
        <p>We've analyzed thousands of data points and rigorously tested each model to select the <strong>single best performer</strong> for every health predictor.</p>
        <p>You're not just getting <em>a</em> prediction — you're getting the <strong>best possible prediction</strong> backed by data science.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("### Important Disclaimer")
    st.error(
        """
        **Medical Advice Notice:**
        
        Health Buddy provides AI-generated predictions for educational and informational purposes only. 
        These predictions should **never replace professional medical advice, diagnosis, or treatment**.
        
        Always consult qualified healthcare providers for medical decisions. In case of emergency, contact your local emergency services immediately.
        """
    )


# --- HEART ATTACK PREDICTOR PAGE ---

elif app_mode == "Heart Attack Predictor":
    st.markdown("# Heart Attack Risk Predictor")
    st.markdown("Our Custom Random Forest model will evaluate your risk based on 8 clinical health metrics.")

    with st.form("heart_attack_form"):
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("### 👤 Patient Demographics")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
        with col2:
            # Your model was trained on 'Gender' which is likely 1 for Male, 0 for Female
            gender = st.selectbox("Gender", (1, 0), format_func=lambda x: "Male" if x == 1 else "Female")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("### 🧬 Cardiac Enzyme Markers")
        col1, col2 = st.columns(2)
        with col1:
            ck_mb = st.number_input("CK-MB (ng/mL)", min_value=0.0, max_value=100.0, value=3.0, step=0.1, help="Creatine kinase-MB enzyme level.")
        with col2:
            troponin = st.number_input("Troponin (ng/mL)", min_value=0.0, max_value=10.0, value=0.02, step=0.01, format="%.2f", help="Troponin enzyme level.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        
        submitted = st.form_submit_button("Analyze My Risk")

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


elif app_mode == "Body Fat Predictor":
    st.markdown("# Body Fat Percentage Predictor")
    st.markdown("Calculate your estimated body fat percentage using advanced anthropometric measurements.")

    with st.form("body_fat_form"):
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown("### 📏 Basic Measurements")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
        with col2:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=75.0, step=0.1, format="%.1f")
        with col3:
            height = st.number_input("Height (cm)", min_value=120.0, max_value=250.0, value=175.0, step=0.1, format="%.1f")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)
        
        
        submitted = st.form_submit_button("Calculate Body Fat %")
        
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
elif app_mode == "Maternal Health Predictor":
    st.markdown("# Maternal Health Risk Predictor")
    st.markdown("Our Custom Random Forest model will evaluate your risk based on 6 key vital signs.")

    with st.form("maternal_health_form"):
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        submitted = st.form_submit_button("Analyze My Risk")

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
elif app_mode == "Obesity Level Predictor":
    st.markdown("# Obesity Level Predictor")
    st.markdown("Our Custom Decision Tree model will analyze 16 lifestyle and physical metrics to predict your weight category.")

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
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        submitted = st.form_submit_button("Calculate Obesity Level")

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




# --- GALLSTONE PREDICTOR PAGE ---
elif app_mode == "🪨 GallStone Predictor":
    st.markdown("# GallStone Risk Predictor")
    st.markdown("Our **Custom AdaBoost** model will analyze 38 health and body composition metrics to predict gallstone risk.")

    # Define the 38 features and their order
    COLUMN_ORDER = [
        'Age', 'Gender', 'Height', 'Weight', 'Body_Mass_Index_BMI', 'Total_Body_Water_TBW', 
        'Extracellular_Water_ECW', 'Intracellular_Water_ICW', 'Extracellular_Fluid/Total_Body_Water_ECF/TBW', 
        'Total_Body_Fat_Ratio_TBFR', 'Lean_Mass_LM', 'Body_Protein_Content_Protein', 
        'Visceral_Fat_Rating_VFR', 'Bone_Mass_BM', 'Muscle_Mass_MM', 'Obesity', 
        'Total_Fat_Content_TFC', 'Visceral_Fat_Area_VFA', 'Visceral_Muscle_Area_VMA_Kg', 
        'Glucose', 'Total_Cholesterol_TC', 'Low_Density_Lipoprotein_LDL', 
        'High_Density_Lipoprotein_HDL', 'Triglyceride', 'Aspartat_Aminotransferaz_AST', 
        'Alanin_Aminotransferaz_ALT', 'Alkaline_Phosphatase_ALP', 'Creatinine', 
        'Glomerular_Filtration_Rate_GFR', 'C-Reactive_Protein_CRP', 'Hemoglobin_HGB', 
        'Vitamin_D', 'Coronary_Artery_Disease_CAD', 'Hypothyroidism', 'Hyperlipidemia', 
        'Diabetes_Mellitus_DM', 'Comorbidity', 'Hepatic_Fat_Accumulation_HFA'
    ]

    with st.form("gallstone_form"):
        st.markdown("### 👤 Patient Demographics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            Age = st.number_input("Age", min_value=1, max_value=120, value=50)
        with col2:
            Gender = st.selectbox("Gender", ('Male', 'Female'))
        with col3:
            Height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=165.0, step=0.1)
        with col4:
            Weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.1)
        
        st.markdown("### 📊 Body Composition")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            Body_Mass_Index_BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0, step=0.1)
        with col2:
            Total_Body_Water_TBW = st.number_input("Total Body Water (TBW)", value=30.0, step=0.1)
        with col3:
            Extracellular_Water_ECW = st.number_input("Extracellular Water (ECW)", value=12.0, step=0.1)
        with col4:
            Intracellular_Water_ICW = st.number_input("Intracellular Water (ICW)", value=18.0, step=0.1)

        with col1:
            Total_Body_Fat_Ratio_TBFR = st.number_input("Total Body Fat Ratio (%)", value=35.0, step=0.1)
        with col2:
            Lean_Mass_LM = st.number_input("Lean Mass (kg)", value=45.0, step=0.1)
        with col3:
            Body_Protein_Content_Protein = st.number_input("Protein (kg)", value=9.0, step=0.1)
        with col4:
            Visceral_Fat_Rating_VFR = st.number_input("Visceral Fat Rating", value=10, step=1)
            
        with col1:
            Bone_Mass_BM = st.number_input("Bone Mass (kg)", value=2.0, step=0.1)
        with col2:
            Muscle_Mass_MM = st.number_input("Muscle Mass (kg)", value=42.0, step=0.1)
        with col3:
            Total_Fat_Content_TFC = st.number_input("Total Fat Content (kg)", value=25.0, step=0.1)
        with col4:
            Visceral_Fat_Area_VFA = st.number_input("Visceral Fat Area (cm²)", value=90.0, step=0.1)
            
        with col1:
             Visceral_Muscle_Area_VMA_Kg = st.number_input("Visceral Muscle Area (kg)", value=40.0, step=0.1)
        with col2:
            Obesity = st.number_input("Obesity (0=No, 1=Yes)", 0, 1, 0)
            
        
        st.markdown("### 🧪 Blood Labs")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            Glucose = st.number_input("Glucose", value=100.0, step=0.1)
        with col2:
            Total_Cholesterol_TC = st.number_input("Total Cholesterol (TC)", value=220.0, step=0.1)
        with col3:
            Low_Density_Lipoprotein_LDL = st.number_input("LDL", value=130.0, step=0.1)
        with col4:
            High_Density_Lipoprotein_HDL = st.number_input("HDL", value=40.0, step=0.1)

        with col1:
            Triglyceride = st.number_input("Triglyceride", value=160.0, step=0.1)
        with col2:
            Aspartat_Aminotransferaz_AST = st.number_input("AST", value=30.0, step=0.1)
        with col3:
            Alanin_Aminotransferaz_ALT = st.number_input("ALT", value=35.0, step=0.1)
        with col4:
            Alkaline_Phosphatase_ALP = st.number_input("ALP", value=100.0, step=0.1)
            
        with col1:
            Creatinine = st.number_input("Creatinine", value=0.9, step=0.1)
        with col2:
            Glomerular_Filtration_Rate_GFR = st.number_input("GFR", value=80.0, step=0.1)
        with col3:
            C_Reactive_Protein_CRP = st.number_input("C-Reactive Protein (CRP)", value=2.0, step=0.1)
        with col4:
            Hemoglobin_HGB = st.number_input("Hemoglobin (HGB)", value=13.0, step=0.1)
        
        with col1:
            Vitamin_D = st.number_input("Vitamin D", value=20.0, step=0.1)

        st.markdown("### 🩺 Clinical History")
        col1, col2, col3 = st.columns(3)
        with col1:
            Coronary_Artery_Disease_CAD = st.number_input("CAD (0=No, 1=Yes)", 0, 1, 0)
        with col2:
            Hypothyroidism = st.number_input("Hypothyroidism (0=No, 1=Yes)", 0, 1, 0)
        with col3:
            Hyperlipidemia = st.number_input("Hyperlipidemia (0=No, 1=Yes)", 0, 1, 0)

        with col1:
            Diabetes_Mellitus_DM = st.number_input("Diabetes (0=No, 1=Yes)", 0, 1, 0)
        with col2:
            Comorbidity = st.selectbox("Comorbidity", ('No', 'Yes'))
        with col3:
            Hepatic_Fat_Accumulation_HFA = st.selectbox("Hepatic Fat Accumulation", ('None', 'Mild', 'Moderate', 'Severe'))

        st.markdown("---")
        submitted = st.form_submit_button("Analyze GallStone Risk")

    if submitted:
        if gallstone_model is None:
            st.error("❌ **Model Error:** The GallStone prediction model is not loaded. Please check the server logs.")
        else:
            try:
                # 1. Create a dictionary of all 38 inputs
                input_data = {
                    'Age': Age, 'Gender': Gender, 'Height': Height, 'Weight': Weight, 
                    'Body_Mass_Index_BMI': Body_Mass_Index_BMI, 'Total_Body_Water_TBW': Total_Body_Water_TBW, 
                    'Extracellular_Water_ECW': Extracellular_Water_ECW, 'Intracellular_Water_ICW': Intracellular_Water_ICW, 
                    'Extracellular_Fluid/Total_Body_Water_ECF/TBW': Extracellular_Water_ECW / Total_Body_Water_TBW if Total_Body_Water_TBW else 0, # Calculate ratio
                    'Total_Body_Fat_Ratio_TBFR': Total_Body_Fat_Ratio_TBFR, 'Lean_Mass_LM': Lean_Mass_LM, 
                    'Body_Protein_Content_Protein': Body_Protein_Content_Protein, 
                    'Visceral_Fat_Rating_VFR': Visceral_Fat_Rating_VFR, 'Bone_Mass_BM': Bone_Mass_BM, 
                    'Muscle_Mass_MM': Muscle_Mass_MM, 'Obesity': Obesity, 
                    'Total_Fat_Content_TFC': Total_Fat_Content_TFC, 'Visceral_Fat_Area_VFA': Visceral_Fat_Area_VFA, 
                    'Visceral_Muscle_Area_VMA_Kg': Visceral_Muscle_Area_VMA_Kg, 
                    'Glucose': Glucose, 'Total_Cholesterol_TC': Total_Cholesterol_TC, 
                    'Low_Density_Lipoprotein_LDL': Low_Density_Lipoprotein_LDL, 
                    'High_Density_Lipoprotein_HDL': High_Density_Lipoprotein_HDL, 'Triglyceride': Triglyceride, 
                    'Aspartat_Aminotransferaz_AST': Aspartat_Aminotransferaz_AST, 
                    'Alanin_Aminotransferaz_ALT': Alanin_Aminotransferaz_ALT, 'Alkaline_Phosphatase_ALP': Alkaline_Phosphatase_ALP, 
                    'Creatinine': Creatinine, 'Glomerular_Filtration_Rate_GFR': Glomerular_Filtration_Rate_GFR, 
                    'C-Reactive_Protein_CRP': C_Reactive_Protein_CRP, 'Hemoglobin_HGB': Hemoglobin_HGB, 
                    'Vitamin_D': Vitamin_D, 'Coronary_Artery_Disease_CAD': Coronary_Artery_Disease_CAD, 
                    'Hypothyroidism': Hypothyroidism, 'Hyperlipidemia': Hyperlipidemia, 
                    'Diabetes_Mellitus_DM': Diabetes_Mellitus_DM, 'Comorbidity': Comorbidity, 
                    'Hepatic_Fat_Accumulation_HFA': Hepatic_Fat_Accumulation_HFA
                }
                
                # 2. Convert to DataFrame with correct column order
                input_df = pd.DataFrame([input_data], columns=COLUMN_ORDER)

                # 3. Make prediction (pipeline handles all preprocessing)
                prediction = gallstone_model.predict(input_df)
                
                # Get the single prediction ('Yes' or 'No')
                result_label = prediction[0]
                
                # --- [END] NEW PREDICTION LOGIC ---

                st.markdown("### 📋 Analysis Results")
                
                if result_label == 'Yes':
                    st.error(
                        f"**Prediction: {result_label} (High Risk for GallStones)**",
                        icon="🚨"
                    )
                    st.warning("This prediction indicates an elevated risk for gallstones. Please consult a healthcare provider for further evaluation.")
                else: # 'No'
                    st.success(
                        f"**Prediction: {result_label} (Low Risk for GallStones)**",
                        icon="💚"
                    )
                    st.info("Your results suggest a low risk. Continue to maintain a healthy lifestyle and regular check-ups.")
            
            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {e}")





# --- PLACEHOLDER PAGES ---
# --- DIABETES PREDICTOR PAGE ---
elif app_mode == "🩸 Diabetes Predictor":
    st.markdown("# Diabetes Risk Predictor")
    st.markdown("This predictor analyzes **24 clinical metrics** to assess risk, based on our **Custom Logistic Regression** model.")

    # These are the 24 features your model was trained on
    COLUMN_ORDER = [
        'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
        'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 
        'cad', 'appet', 'pe', 'ane'
    ]
    
    with st.form("diabetes_form"):
        st.markdown("### 👤 Patient Vitals")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
        with col2:
            bp = st.number_input("Blood Pressure (bp, mm/Hg)", min_value=40.0, max_value=200.0, value=80.0, step=1.0)
        with col3:
            bgr = st.number_input("Blood Glucose (bgr, mgs/dl)", min_value=20.0, max_value=500.0, value=100.0, step=1.0)

        st.markdown("### 🧪 Lab Results (Blood)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            bu = st.number_input("Blood Urea (bu, mgs/dl)", min_value=1.0, max_value=400.0, value=40.0, step=0.1)
        with col2:
            sc = st.number_input("Serum Creatinine (sc, mgs/dl)", min_value=0.1, max_value=80.0, value=1.2, step=0.1)
        with col3:
            sod = st.number_input("Sodium (sod, mEq/L)", min_value=100.0, max_value=180.0, value=138.0, step=0.1)
        with col4:
            pot = st.number_input("Potassium (pot, mEq/L)", min_value=2.0, max_value=10.0, value=4.5, step=0.1)

        st.markdown("### 🩸 Lab Results (Complete Blood Count)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            hemo = st.number_input("Hemoglobin (hemo, gms/dl)", min_value=3.0, max_value=20.0, value=15.0, step=0.1)
        with col2:
            pcv = st.number_input("Packed Cell Vol (pcv, %)", min_value=10.0, max_value=60.0, value=45.0, step=1.0)
        with col3:
            wc = st.number_input("WBC Count (wc, cells/cumm)", min_value=2000.0, max_value=30000.0, value=7500.0, step=100.0)
        with col4:
            rc = st.number_input("RBC Count (rc, millions/cmm)", min_value=2.0, max_value=9.0, value=5.2, step=0.1)

        st.markdown("### 💧 Urinalysis (Nominal Values)")
        # --- IMPORTANT: These must be strings or numbers as defined in your training script ---
        col1, col2, col3 = st.columns(3)
        with col1:
            # Your training script treated these as numeric, so we use numbers
            sg = st.selectbox("Specific Gravity (sg)", (1.005, 1.010, 1.015, 1.020, 1.025), index=2)
        with col2:
            al = st.selectbox("Albumin (al)", (0.0, 1.0, 2.0, 3.0, 4.0, 5.0), index=0)
        with col3:
            su = st.selectbox("Sugar (su)", (0.0, 1.0, 2.0, 3.0, 4.0, 5.0), index=0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            rbc = st.selectbox("Red Blood Cells (rbc)", ('normal', 'abnormal'), index=0)
        with col2:
            pc = st.selectbox("Pus Cell (pc)", ('normal', 'abnormal'), index=0)
        with col3:
            pcc = st.selectbox("Pus Cell Clumps (pcc)", ('notpresent', 'present'), index=0)
        
        with col1: # Re-using first column
             ba = st.selectbox("Bacteria (ba)", ('notpresent', 'present'), index=0)

        st.markdown("### 🩺 Clinical History & Symptoms (Nominal)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            htn = st.selectbox("Hypertension (htn)", ('no', 'yes'), index=0)
        with col2:
            dm = st.selectbox("Diabetes Mellitus (dm)", ('no', 'yes'), index=0, help="Note: This is a historical symptom, not the prediction target.")
        with col3:
            cad = st.selectbox("Coronary Artery Disease (cad)", ('no', 'yes'), index=0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            appet = st.selectbox("Appetite (appet)", ('good', 'poor'), index=0)
        with col2:
            pe = st.selectbox("Pedal Edema (pe)", ('no', 'yes'), index=0)
        with col3:
            ane = st.selectbox("Anemia (ane)", ('no', 'yes'), index=0)

        st.markdown("---")
        submitted = st.form_submit_button("Analyze My Risk")

    if submitted:
        if diabetes_model_pipeline is None:
            st.error("❌ **Model Error:** The Diabetes prediction model is not loaded. Please check the server logs.")
        else:
            try:
                # 1. Create a dictionary of all 24 inputs
                input_data = {
                    'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su, 'rbc': rbc,
                    'pc': pc, 'pcc': pcc, 'ba': ba, 'bgr': bgr, 'bu': bu, 'sc': sc,
                    'sod': sod, 'pot': pot, 'hemo': hemo, 'pcv': pcv, 'wc': wc,
                    'rc': rc, 'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet,
                    'pe': pe, 'ane': ane
                }
                
                # 2. Convert to DataFrame with correct column order
                input_df = pd.DataFrame([input_data], columns=COLUMN_ORDER)

                # 3. Make prediction
                prediction_encoded = diabetes_model_pipeline.predict(input_df)
                
                # 4. Map to class names (from your predict.py)
                # 0 = ckd, 1 = notckd
                class_names = ['High Risk', 'Low Risk'] 
                prediction_label = class_names[prediction_encoded[0]]
                
                # --- [END] NEW PREDICTION LOGIC ---

                st.markdown("### 📋 Analysis Results")
                
                if prediction_label == 'High Risk':
                    st.error(
                        f"**⚠️ HIGH RISK DETECTED**",
                        icon="🚨"
                    )
                    st.warning(
                        """
                        **Medical Note:**
                        
                        This prediction indicates a high-risk profile based on clinical markers. Please consult an endocrinologist or healthcare provider **immediately** for proper evaluation.
                        """
                    )
                else: # 'Low Risk'
                    st.success(
                        f"**✅ LOW RISK DETECTED**",
                        icon="💚"
                    )
                    st.info(
                        """
                        **Recommendation:**
                        
                        Your results suggest a low-risk profile. Continue to maintain regular health check-ups and a healthy lifestyle.
                        """
                        )
            
            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {e}")

elif app_mode == "🩺 Liver Disease Predictor":
    st.markdown("# 🩺 Liver Disease Predictor")
    st.info("🛠️ **Feature Coming Soon!** Advanced liver health analysis will be available shortly.", icon="⚡")

# --- KIDNEY DISEASE PREDICTOR PAGE ---
elif app_mode == "🔬 Kidney Disease Predictor":
    st.markdown("# Chronic Kidney Disease (CKD) Predictor")
    st.markdown("Our **Custom KNN** model will analyze 24 clinical metrics to predict your risk for CKD.")

    # These are the 24 features your model was trained on
    COLUMN_ORDER = [
        'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
        'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 
        'cad', 'appet', 'pe', 'ane'
    ]
    
    with st.form("kidney_disease_form"):
        st.markdown("### 👤 Patient Vitals")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
        with col2:
            bp = st.number_input("Blood Pressure (bp, mm/Hg)", min_value=40.0, max_value=200.0, value=80.0, step=1.0)
        with col3:
            bgr = st.number_input("Blood Glucose (bgr, mgs/dl)", min_value=20.0, max_value=500.0, value=100.0, step=1.0)

        st.markdown("### 🧪 Lab Results (Blood)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            bu = st.number_input("Blood Urea (bu, mgs/dl)", min_value=1.0, max_value=400.0, value=40.0, step=0.1)
        with col2:
            sc = st.number_input("Serum Creatinine (sc, mgs/dl)", min_value=0.1, max_value=80.0, value=1.2, step=0.1)
        with col3:
            sod = st.number_input("Sodium (sod, mEq/L)", min_value=100.0, max_value=180.0, value=138.0, step=0.1)
        with col4:
            pot = st.number_input("Potassium (pot, mEq/L)", min_value=2.0, max_value=10.0, value=4.5, step=0.1)

        st.markdown("### 🩸 Lab Results (Complete Blood Count)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            hemo = st.number_input("Hemoglobin (hemo, gms/dl)", min_value=3.0, max_value=20.0, value=15.0, step=0.1)
        with col2:
            pcv = st.number_input("Packed Cell Vol (pcv, %)", min_value=10.0, max_value=60.0, value=45.0, step=1.0)
        with col3:
            wc = st.number_input("WBC Count (wc, cells/cumm)", min_value=2000.0, max_value=30000.0, value=7500.0, step=100.0)
        with col4:
            rc = st.number_input("RBC Count (rc, millions/cmm)", min_value=2.0, max_value=9.0, value=5.2, step=0.1)

        st.markdown("### 💧 Urinalysis (Nominal Values)")
        # --- IMPORTANT: These must be strings, as per your predict.py ---
        col1, col2, col3 = st.columns(3)
        with col1:
            sg = st.selectbox("Specific Gravity (sg)", ('1.005', '1.010', '1.015', '1.020', '1.025'), index=2)
        with col2:
            al = st.selectbox("Albumin (al)", ('0.0', '1.0', '2.0', '3.0', '4.0', '5.0'), index=0)
        with col3:
            su = st.selectbox("Sugar (su)", ('0.0', '1.0', '2.0', '3.0', '4.0', '5.0'), index=0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            rbc = st.selectbox("Red Blood Cells (rbc)", ('normal', 'abnormal'), index=0)
        with col2:
            pc = st.selectbox("Pus Cell (pc)", ('normal', 'abnormal'), index=0)
        with col3:
            pcc = st.selectbox("Pus Cell Clumps (pcc)", ('notpresent', 'present'), index=0)
        
        with col1: # Re-using first column
             ba = st.selectbox("Bacteria (ba)", ('notpresent', 'present'), index=0)

        st.markdown("### 🩺 Clinical History & Symptoms (Nominal)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            htn = st.selectbox("Hypertension (htn)", ('no', 'yes'), index=0)
        with col2:
            dm = st.selectbox("Diabetes Mellitus (dm)", ('no', 'yes'), index=0)
        with col3:
            cad = st.selectbox("Coronary Artery Disease (cad)", ('no', 'yes'), index=0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            appet = st.selectbox("Appetite (appet)", ('good', 'poor'), index=0)
        with col2:
            pe = st.selectbox("Pedal Edema (pe)", ('no', 'yes'), index=0)
        with col3:
            ane = st.selectbox("Anemia (ane)", ('no', 'yes'), index=0)

        st.markdown("---")
        submitted = st.form_submit_button("Analyze My Risk")

    if submitted:
        if kidney_model_pipeline is None:
            st.error("❌ **Model Error:** The Kidney Disease prediction model is not loaded. Please check the server logs.")
        else:
            try:
                # 1. Create a dictionary of all 24 inputs
                input_data = {
                    'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su, 'rbc': rbc,
                    'pc': pc, 'pcc': pcc, 'ba': ba, 'bgr': bgr, 'bu': bu, 'sc': sc,
                    'sod': sod, 'pot': pot, 'hemo': hemo, 'pcv': pcv, 'wc': wc,
                    'rc': rc, 'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet,
                    'pe': pe, 'ane': ane
                }
                
                # 2. Convert to DataFrame with correct column order
                input_df = pd.DataFrame([input_data], columns=COLUMN_ORDER)

                # 3. Make prediction
                prediction_encoded = kidney_model_pipeline.predict(input_df)
                
                # 4. Map to class names (from your predict.py)
                class_names = ['ckd', 'notckd']
                prediction_label = class_names[prediction_encoded[0]]
                
                # --- [END] NEW PREDICTION LOGIC ---

                st.markdown("### 📋 Analysis Results")
                
                if prediction_label == 'ckd':
                    st.error(
                        f"**⚠️ HIGH RISK DETECTED (Result: ckd)**",
                        icon="🚨"
                    )
                    st.warning(
                        """
                        **Immediate Action Required:**
                        
                        This prediction indicates a high risk for Chronic Kidney Disease. Please consult a nephrologist or healthcare provider **immediately** for proper evaluation.
                        """
                    )
                else: # 'notckd'
                    st.success(
                        f"**✅ LOW RISK DETECTED (Result: notckd)**",
                        icon="💚"
                    )
                    st.info(
                        """
                        **Recommendation:**
                        
                        Your results suggest a low risk for Chronic Kidney Disease. Continue to maintain regular health check-ups and a healthy lifestyle.
                        """
                        )
            
            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {e}")

# --- BRAIN TUMOR PREDICTOR PAGE ---
elif app_mode == "🧠 Brain Tumor Predictor":
    st.markdown("# Brain Tumor Predictor (MRI)")
    st.markdown("Our **Custom KNN** model will analyze a brain MRI scan to classify the tumor type.")
    st.info("This model uses advanced feature extraction (HOG, GLCM, LBP) on the image provided.")
    
    # 1. Create the file uploader
    uploaded_file = st.file_uploader(
        "Upload a Brain MRI (jpg, png, bmp)...", 
        type=["jpg", "png", "jpeg", "bmp"]
    )
    
    temp_image_path = None
    
    if uploaded_file is not None:
        # 2. Display the uploaded image
        st.image(uploaded_file, caption="Uploaded MRI Scan.", use_column_width=True)
        
        # 3. Create a temporary file to save the image
        # This is necessary because extract_features_from_image expects a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_image_path = tmp.name
        
        # 4. Create the prediction button
        if st.button("Analyze MRI Scan", use_container_width=True):
            # Check if all models loaded
            if not all([bt_model, bt_le, bt_scaler, bt_pca]):
                st.error("❌ **Model Error:** The Brain Tumor prediction pipeline is not fully loaded. Check server logs.")
            else:
                try:
                    with st.spinner("Analyzing image and extracting features..."):
                        # 5. Run the full extraction and prediction pipeline
                        
                        # 5a. Extract features from the temp file path
                        features = extract_features_from_image(temp_image_path)
                        
                        if features is not None:
                            # 5b. Reshape features
                            features_2d = features.reshape(1, -1)
                            
                            # 5c. Apply scaler
                            scaled_features = bt_scaler.transform(features_2d)
                            
                            # 5d. Apply PCA
                            pca_features = bt_pca.transform(scaled_features)
                            
                            # 5e. Predict
                            prediction_index = bt_model.predict(pca_features)[0]
                            
                            # 5f. Decode the result
                            prediction_name = bt_le.inverse_transform([prediction_index])[0]
                            
                            # --- [END] NEW PREDICTION LOGIC ---
                            
                            st.markdown("### 📋 Analysis Results")
                            st.success(f"**Prediction: {prediction_name}**")
                            st.info(f"The model has classified the MRI scan as **{prediction_name}**.")
                        
                        else:
                            st.error("Could not read or process the uploaded image.")

                except Exception as e:
                    st.error(f"❌ An error occurred during prediction: {e}")
                
                finally:
                    # 7. Clean up (delete) the temporary file
                    if temp_image_path and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)