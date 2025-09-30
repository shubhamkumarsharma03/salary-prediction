"""
Data Overview Page
"""
import streamlit as st
import numpy as np
from components.charts import (
    create_salary_histogram, create_experience_scatter,
    create_salary_by_location_bar, create_salary_by_role_bar,
    create_correlation_heatmap
)

def render_data_overview_page(df):
    """Render the data overview page"""
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Avg Salary", f"₹{df['salary_in_inr'].mean():,.0f}")
    with col4:
        st.metric("Salary Range", f"₹{df['salary_in_inr'].max() - df['salary_in_inr'].min():,.0f}")

    # Data distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Salary Distribution")
        fig = create_salary_histogram(df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Experience vs Salary")
        fig = create_experience_scatter(df)
        st.plotly_chart(fig, use_container_width=True)

    # Categorical analysis
    st.subheader("📊 Categorical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_salary_by_location_bar(df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_salary_by_role_bar(df)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap for numeric data
    st.subheader("🔥 Correlation Analysis")
    fig = create_correlation_heatmap(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation analysis.")