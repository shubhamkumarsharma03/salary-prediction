"""
Model Training and Evaluation Logic
"""
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from models.custom_rf import EnhancedRandomForestRegressor

@st.cache_data
def train_models(df, test_size=0.2, random_state=42):
    """Train multiple models and return results"""
    if df is None:
        return {}, {}, None, None, None, None
        
    X = df.drop("salary_in_inr", axis=1)
    y = df["salary_in_inr"].values

    # Preprocessing
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    # Models
    models = {
        "Custom Random Forest": EnhancedRandomForestRegressor(n_estimators=30, max_depth=15),
        "Scikit-Learn RF": RandomForestRegressor(n_estimators=50, max_depth=15, random_state=random_state),
        "Decision Tree": DecisionTreeRegressor(max_depth=15, random_state=random_state),
        "Linear Regression": LinearRegression()
    }

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    results = {}
    pipelines = {}
    
    for name, model in models.items():
        try:
            # Create pipeline
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)
            
            # Metrics
            results[name] = {
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "train_r2": r2_score(y_train, y_pred_train),
                "test_r2": r2_score(y_test, y_pred_test),
                "train_mae": mean_absolute_error(y_train, y_pred_train),
                "test_mae": mean_absolute_error(y_test, y_pred_test)
            }
            
            pipelines[name] = pipeline
        except Exception as e:
            st.warning(f"Error training {name}: {e}")
            continue

    return results, pipelines, X_train, X_test, y_train, y_test