"""
Feature Analysis Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from models.model_trainer import train_models
from components.charts import create_feature_importance_chart

def render_feature_analysis_page(df):
    """Render the feature analysis page"""
    st.markdown("## 🧮 Feature Analysis")
    
    results, pipelines, _, _, _, _ = train_models(df)
    
    if not pipelines:
        st.error("No trained models available!")
        return
    
    st.subheader("🎯 Feature Importance Analysis")
    
    try:
        # Get feature importance from the best performing model
        best_model_name = "Scikit-Learn RF" if "Scikit-Learn RF" in pipelines else list(pipelines.keys())[0]
        model = pipelines[best_model_name].named_steps["model"]
        
        # Get feature names after preprocessing
        categorical_cols = df.drop("salary_in_inr", axis=1).select_dtypes(include=["object"]).columns
        numerical_cols = df.drop("salary_in_inr", axis=1).select_dtypes(include=["int64", "float64"]).columns
        
        preprocessor = pipelines[best_model_name].named_steps["preprocessor"]
        
        # Get feature names
        feature_names = (
            numerical_cols.tolist() + 
            preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
        )
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = create_feature_importance_chart(importance_df, best_model_name)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error analyzing feature importance: {e}")

    # Statistical analysis
    st.subheader("📈 Statistical Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Experience analysis
        experience_bins = pd.cut(df['experience'], bins=5, labels=['0-8', '8-16', '16-24', '24-32', '32-40'])
        exp_salary = df.groupby(experience_bins)['salary_in_inr'].mean()
        
        fig = px.bar(x=exp_salary.index, y=exp_salary.values,
                    title="Average Salary by Experience Range")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Education impact
        education_impact = df.groupby('education')['salary_in_inr'].mean().sort_values(ascending=False)
        
        fig = px.bar(x=education_impact.index, y=education_impact.values,
                    title="Average Salary by Education Level")
        st.plotly_chart(fig, use_container_width=True)