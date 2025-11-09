# HealthBuddy 🏥

A comprehensive machine learning-powered health prediction application built with Streamlit. HealthBuddy provides AI-driven analysis for various health conditions including heart disease, diabetes, brain tumors, EEG analysis, and more.

## 🌟 Features

HealthBuddy offers multiple predictive models for different health conditions:

- **Heart Attack Risk Predictor** - Analyzes cardiac biomarkers and patient demographics
- **Body Fat Percentage Predictor** - Estimates body fat using anthropometric measurements
- **Maternal Health Risk Predictor** - Assesses pregnancy-related health risks
- **Obesity Level Predictor** - Classifies obesity levels based on lifestyle factors
- **Chronic Kidney Disease Predictor** - Early detection of kidney disease
- **Diabetes Predictor** - Type 2 diabetes risk assessment
- **Gallstone Predictor** - Gallbladder health analysis
- **Mental Health Predictor** - Mental health risk assessment
- **Brain Tumor Detection** - MRI scan analysis for tumor detection
- **EEG Stress Analysis** - EEG signal processing for stress detection

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/KRISHNA-JAIN15/HealthBuddy.git
   cd HealthBuddy
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv

   # On Linux/Mac:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` doesn't exist, install the dependencies manually:

   ```bash
   pip install streamlit pandas numpy scikit-learn opencv-python
   pip install scipy mne scikit-image matplotlib seaborn
   pip install plotly pillow
   ```

### 🏃‍♂️ Running the Application

1. **Start the Streamlit application**

   ```bash
   python -m streamlit run app.py
   ```

2. **Access the application**
   - Open your web browser and navigate to `http://localhost:8501`
   - The HealthBuddy interface will load automatically

## 📋 Requirements

### Core Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=9.5.0
plotly>=5.15.0
```

### Optional Dependencies (for advanced features)

```
mne>=1.4.0              # For EEG analysis
scikit-image>=0.21.0    # For advanced image processing
```

### Installing Optional Dependencies

For full functionality, especially EEG analysis and advanced image processing:

```bash
pip install mne scikit-image
```

**Note**: If you encounter issues with MNE installation, some features may be disabled but the core application will still function.

## 🗂️ Project Structure

```
HealthBuddy/
├── app.py                          # Main Streamlit application
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── venv/                          # Virtual environment
├── BodyFat/                       # Body fat prediction model
│   ├── main.py                    # Training script
│   ├── predict.py                 # Prediction utilities
│   ├── RandomForest.py            # Custom RF implementation
│   ├── bodyfat.csv               # Dataset
│   └── README.md
├── Brain Tumor/                   # Brain tumor detection
│   ├── predict.py
│   ├── train.py
│   ├── Training/                  # Training images
│   └── Testing/                   # Test images
├── ChronicKidneyDisease/          # CKD prediction
│   ├── main.py
│   ├── predict.py
│   ├── chronic_kidney_disease.csv
│   └── README.md
├── Classifier_codes/              # Reusable ML algorithms
│   ├── DecisionTree.py
│   ├── RandomForest.py
│   ├── LogisticRegression.py
│   ├── SVM.py
│   ├── KNN.py
│   └── ...
├── Diabetes/                      # Diabetes prediction
│   ├── main.py
│   ├── predict.py
│   ├── Diabetes.csv
│   └── README.md
├── EEG/                          # EEG analysis
│   ├── extract.py                # Feature extraction
│   ├── model.py                  # Model training
│   ├── predict.py                # Prediction script
│   ├── *.edf                     # EEG data files
│   └── README.txt
├── GallStone/                    # Gallstone prediction
├── HeartAttack/                  # Heart attack risk
├── MentalHealth/                 # Mental health assessment
└── ObesityLevel/                 # Obesity classification
```

## 🎯 Usage Guide

### 1. Web Interface

1. Launch the application using `streamlit run app.py`
2. Select a predictor from the sidebar
3. Input the required health parameters or upload medical files
4. Click "Analyze" or "Predict" to get results
5. Review the AI-generated health assessment

### 2. Command Line Usage (Individual Predictors)

Each module can be run independently:

```bash
# Diabetes prediction
cd Diabetes
python main.py

# Body fat analysis
cd BodyFat
python main.py

# EEG analysis
cd EEG
python predict.py your_eeg_file.edf
```

### 3. Supported File Formats

- **Images**: JPG, JPEG, PNG (for brain tumor detection)
- **EEG Data**: EDF files (European Data Format)
- **CSV**: For bulk data analysis

## 🧠 Machine Learning Models

HealthBuddy implements various ML algorithms:

### Classification Algorithms

- **Random Forest**: Primary classifier for most models
- **Decision Trees**: Interpretable classification
- **Logistic Regression**: Linear classification
- **Support Vector Machine (SVM)**: Non-linear classification
- **K-Nearest Neighbors (KNN)**: Instance-based learning
- **Naive Bayes**: Probabilistic classification
- **Multi-Layer Perceptron**: Neural network approach

### Image Processing

- **HOG (Histogram of Oriented Gradients)**: Feature extraction
- **GLCM (Gray-Level Co-occurrence Matrix)**: Texture analysis
- **OpenCV**: Image preprocessing

### Signal Processing (EEG)

- **MNE-Python**: EEG data handling
- **SciPy**: Signal processing and statistical analysis
- **Frequency Domain Analysis**: Power spectral density
- **Time Domain Features**: Statistical moments

## 🔧 Configuration

### Environment Variables

No environment variables required for basic functionality.

### Model Files

Pre-trained models are automatically loaded from their respective directories. Ensure all `.pkl` files are present in their corresponding folders.

### Custom Models

To train custom models:

1. Navigate to the specific condition directory
2. Run the training script:
   ```bash
   python main.py
   ```
3. The trained model will be saved automatically

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Missing dependencies
   pip install --upgrade streamlit pandas numpy scikit-learn
   ```

2. **MNE/SciPy Import Failed**

   ```bash
   # Install additional dependencies
   pip install mne scipy
   ```

   _Note: EEG predictor will be disabled if these fail_

3. **scikit-image Import Failed**

   ```bash
   # Install image processing library
   pip install scikit-image
   ```

4. **Model Files Not Found**

   - Ensure you've run the training scripts in each directory
   - Check that `.pkl` files exist in the model directories

5. **Permission Errors**
   ```bash
   # On Linux/Mac, try:
   sudo chmod +x app.py
   ```

### Performance Issues

- For large EEG files, processing may take several minutes
- Brain tumor analysis requires sufficient RAM for image processing
- Consider using smaller image sizes if memory issues occur

## 📊 Data Sources

- **Heart Disease**: Clinical cardiac biomarkers dataset
- **Body Fat**: Anthropometric measurements dataset
- **Diabetes**: BRFSS (Behavioral Risk Factor Surveillance System) data
- **Chronic Kidney Disease**: UCI ML Repository dataset
- **EEG**: PhysioNet EEG database (European Data Format)
- **Brain Tumor**: Medical imaging datasets




## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



**Happy Health Monitoring! 🌟**
