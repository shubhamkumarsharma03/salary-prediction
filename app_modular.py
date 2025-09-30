"""
Main Application Entry Point
Employee Salary Analytics Dashboard - Modular Version
"""
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import configuration and components
from config.settings import setup_page_config
from config.styles import load_custom_styles
from data.loader import load_data
from components.sidebar import render_sidebar

# Import pages
from pages.data_overview import render_data_overview_page
from pages.prediction import render_prediction_page
from pages.model_comparison import render_model_comparison_page
from pages.feature_analysis import render_feature_analysis_page
from pages.model_tuning import render_model_tuning_page

def main():
    """Main application function"""
    # Setup page configuration
    setup_page_config()
    
    # Load custom styles
    load_custom_styles()
    
    # Main header
    st.markdown('<h1 class="main-header">💼 Employee Salary Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data first
    df = load_data()
    if df is None:
        st.stop()
    
    # Render sidebar and get selected page
    page = render_sidebar()

    # Route to appropriate page
    if page == "📊 Data Overview":
        render_data_overview_page(df)
    elif page == "🔮 Salary Prediction":
        render_prediction_page(df)
    elif page == "📈 Model Comparison":
        render_model_comparison_page(df)
    elif page == "🧮 Feature Analysis":
        render_feature_analysis_page(df)
    elif page == "⚙️ Model Tuning":
        render_model_tuning_page(df)

if __name__ == "__main__":
    main()