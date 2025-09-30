"""
Reusable Chart Components
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_salary_histogram(df):
    """Create salary distribution histogram"""
    fig = px.histogram(df, x="salary_in_inr", nbins=25, 
                      title="Distribution of Salaries")
    fig.update_layout(showlegend=False)
    return fig

def create_experience_scatter(df):
    """Create experience vs salary scatter plot"""
    fig = px.scatter(df, x="experience", y="salary_in_inr", 
                    color="education", size="salary_in_inr",
                    title="Experience vs Salary by Education")
    return fig

def create_salary_by_location_bar(df):
    """Create average salary by location bar chart"""
    avg_salary_by_location = df.groupby('location')['salary_in_inr'].mean().sort_values(ascending=False)
    fig = px.bar(x=avg_salary_by_location.index, y=avg_salary_by_location.values,
                title="Average Salary by Location")
    return fig

def create_salary_by_role_bar(df):
    """Create average salary by job role bar chart"""
    avg_salary_by_role = df.groupby('job_role')['salary_in_inr'].mean().sort_values(ascending=False)
    fig = px.bar(x=avg_salary_by_role.index, y=avg_salary_by_role.values,
                title="Average Salary by Job Role")
    fig.update_layout(xaxis={'tickangle': 45})
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap for numeric features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        fig = px.imshow(correlation_matrix, 
                        title="Feature Correlation Heatmap",
                        aspect="auto", 
                        color_continuous_scale="RdBu_r")
        return fig
    return None

def create_r2_comparison_chart(comparison_df):
    """Create R² score comparison chart"""
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Training R²', x=comparison_df.index, y=comparison_df['train_r2']))
    fig.add_trace(go.Bar(name='Test R²', x=comparison_df.index, y=comparison_df['test_r2']))
    fig.update_layout(barmode='group', title="R² Score: Higher is Better")
    return fig

def create_rmse_comparison_chart(comparison_df):
    """Create RMSE comparison chart"""
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Training RMSE', x=comparison_df.index, y=comparison_df['train_rmse']))
    fig.add_trace(go.Bar(name='Test RMSE', x=comparison_df.index, y=comparison_df['test_rmse']))
    fig.update_layout(barmode='group', title="RMSE: Lower is Better")
    return fig

def create_feature_importance_chart(importance_df, model_name):
    """Create feature importance chart"""
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                title=f"Top 15 Most Important Features ({model_name})")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

def create_prediction_vs_actual_chart(y_test, y_pred):
    """Create predictions vs actual values scatter plot"""
    fig = px.scatter(x=y_test, y=y_pred, 
                    title="Predictions vs Actual Values")
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                           y=[y_test.min(), y_test.max()], 
                           mode='lines', name='Perfect Prediction'))
    return fig