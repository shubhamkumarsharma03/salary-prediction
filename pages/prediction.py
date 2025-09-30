"""
Salary Prediction Page
"""
import streamlit as st
import pandas as pd
from models.model_trainer import train_models

def render_prediction_page(df):
    """Render the salary prediction page"""
    st.markdown("## 🔮 Salary Prediction")
    
    # Load trained models
    results, pipelines, _, _, _, _ = train_models(df)
    
    if not pipelines:
        st.error("No models were trained successfully!")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Enter Employee Details")
        
        experience = st.slider("Years of Experience", 0, 40, 5)
        education = st.selectbox("Education Level", df["education"].unique())
        location = st.selectbox("Location", df["location"].unique())
        company_size = st.selectbox("Company Size", df["company_size"].unique())
        industry = st.selectbox("Industry", df["industry"].unique())
        job_role = st.selectbox("Job Role", df["job_role"].unique())
        remote = st.radio("Remote Work?", ["Yes", "No"])
        
        model_choice = st.selectbox("Choose Model", list(pipelines.keys()))

    with col2:
        st.subheader("📊 Input Summary")
        st.write(f"**Experience:** {experience} years")
        st.write(f"**Education:** {education}")
        st.write(f"**Location:** {location}")
        st.write(f"**Company Size:** {company_size}")
        st.write(f"**Industry:** {industry}")
        st.write(f"**Job Role:** {job_role}")
        st.write(f"**Remote:** {remote}")

    # Prediction
    if st.button("🎯 Predict Salary", type="primary"):
        input_data = pd.DataFrame({
            "experience": [experience],
            "education": [education],
            "location": [location],
            "company_size": [company_size],
            "industry": [industry],
            "job_role": [job_role],
            "remote": [remote]
        })
        
        try:
            prediction = pipelines[model_choice].predict(input_data)[0]
            model_metrics = results[model_choice]
            
            st.markdown(f"""
            <div class="prediction-result">
                <h2>💰 Predicted Salary: ₹{prediction:,.0f}</h2>
                <p>Model: {model_choice}</p>
                <p>Test R² Score: {model_metrics['test_r2']:.3f}</p>
                <p>Test RMSE: ₹{model_metrics['test_rmse']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")