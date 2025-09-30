"""
Application Settings and Configuration
"""
import streamlit as st

def setup_page_config():
    """Set up Streamlit page configuration"""
    st.set_page_config(
        page_title="Employee Salary Analytics",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Navigation options
NAVIGATION_OPTIONS = [
    "📊 Data Overview", 
    "🔮 Salary Prediction", 
    "📈 Model Comparison", 
    "🧮 Feature Analysis", 
    "⚙️ Model Tuning"
]

# Model configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5
}

# File paths
DATA_FILE = "employee_salaries_india.csv"