"""
Model Tuning Page
"""
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from components.charts import create_prediction_vs_actual_chart

def render_model_tuning_page(df):
    """Render the model tuning page"""
    st.markdown("## ⚙️ Model Hyperparameter Tuning")
    
    if df is None:
        return
        
    X = df.drop("salary_in_inr", axis=1)
    y = df["salary_in_inr"].values
    
    st.subheader("🎛️ Tune Random Forest Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
        max_depth = st.slider("Maximum Depth", 5, 30, 15, 1)
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2, 1)
    
    with col2:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5, 1)
    
    if st.button("🚀 Train Custom Model"):
        try:
            # Preprocessing
            categorical_cols = X.select_dtypes(include=["object"]).columns
            numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
            
            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
            ])
            
            # Custom model with user parameters
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='r2')
            
            # Fit and predict
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("CV R² Score", f"{cv_scores.mean():.4f}")
            with col2:
                st.metric("Test R² Score", f"{r2_score(y_test, y_pred):.4f}")
            with col3:
                st.metric("Test RMSE", f"₹{np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
            with col4:
                st.metric("Test MAE", f"₹{mean_absolute_error(y_test, y_pred):,.0f}")
            
            # Prediction vs Actual plot
            fig = create_prediction_vs_actual_chart(y_test, y_pred)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error training custom model: {e}")