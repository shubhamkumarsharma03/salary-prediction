"""
Model Comparison Page
"""
import streamlit as st
import pandas as pd
from models.model_trainer import train_models
from components.charts import create_r2_comparison_chart, create_rmse_comparison_chart

def render_model_comparison_page(df):
    """Render the model comparison page"""
    st.markdown("## 📈 Model Performance Comparison")
    
    results, _, _, _, _, _ = train_models(df)
    
    if not results:
        st.error("No model results available!")
        return
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results).T
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 R² Score Comparison")
        fig = create_r2_comparison_chart(comparison_df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📉 RMSE Comparison")
        fig = create_rmse_comparison_chart(comparison_df)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics table
    st.subheader("📊 Detailed Performance Metrics")
    display_df = comparison_df.copy()
    for col in display_df.columns:
        if 'rmse' in col or 'mae' in col:
            display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.0f}")
        else:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True)