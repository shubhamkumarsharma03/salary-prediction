import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Employee Salary Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-result {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Random Forest with proper error handling
class EnhancedRandomForestRegressor:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features="sqrt"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.features_idx = []
        self.feature_importances_ = None

    def _get_max_features(self, n_features):
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_features

    def fit(self, X, y):
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
            
        n_samples, n_features = X.shape
        max_feats = self._get_max_features(n_features)
        
        # Use scikit-learn's DecisionTreeRegressor for reliability
        from sklearn.tree import DecisionTreeRegressor

        for _ in range(self.n_estimators):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]

            feat_idx = np.random.choice(n_features, max_feats, replace=False)
            self.features_idx.append(feat_idx)

            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, random_state=42)
            tree.fit(X_sample[:, feat_idx], y_sample)
            self.trees.append(tree)
        
        # Calculate ensemble feature importance
        self.feature_importances_ = np.zeros(n_features)
        for i, tree in enumerate(self.trees):
            for j, feat_idx in enumerate(self.features_idx[i]):
                if j < len(tree.feature_importances_):
                    self.feature_importances_[feat_idx] += tree.feature_importances_[j]
        
        if self.n_estimators > 0:
            self.feature_importances_ /= self.n_estimators

    def predict(self, X):
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
            
        preds = np.zeros((self.n_estimators, X.shape[0]))
        for i, tree in enumerate(self.trees):
            if i < len(self.features_idx):
                preds[i] = tree.predict(X[:, self.features_idx[i]])
        return np.mean(preds, axis=0)

# Load and cache data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("employee_salaries_india.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset file 'employee_salaries_india.csv' not found!")
        return None

# Model training with error handling
@st.cache_data
def train_models(test_size=0.2, random_state=42):
    df = load_data()
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

# Main App
def main():
    st.markdown('<h1 class="main-header">💼 Employee Salary Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data first
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox("Choose Analysis Type", 
                               ["📊 Data Overview", "🔮 Salary Prediction", "📈 Model Comparison", 
                                "🧮 Feature Analysis", "⚙️ Model Tuning"])

    if page == "📊 Data Overview":
        data_overview_page(df)
    elif page == "🔮 Salary Prediction":
        prediction_page(df)
    elif page == "📈 Model Comparison":
        model_comparison_page()
    elif page == "🧮 Feature Analysis":
        feature_analysis_page(df)
    elif page == "⚙️ Model Tuning":
        model_tuning_page()

def data_overview_page(df):
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
        fig = px.histogram(df, x="salary_in_inr", nbins=25, 
                          title="Distribution of Salaries")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Experience vs Salary")
        fig = px.scatter(df, x="experience", y="salary_in_inr", 
                        color="education", size="salary_in_inr",
                        title="Experience vs Salary by Education")
        st.plotly_chart(fig, use_container_width=True)

    # Categorical analysis
    st.subheader("📊 Categorical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        avg_salary_by_location = df.groupby('location')['salary_in_inr'].mean().sort_values(ascending=False)
        fig = px.bar(x=avg_salary_by_location.index, y=avg_salary_by_location.values,
                    title="Average Salary by Location")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        avg_salary_by_role = df.groupby('job_role')['salary_in_inr'].mean().sort_values(ascending=False)
        fig = px.bar(x=avg_salary_by_role.index, y=avg_salary_by_role.values,
                    title="Average Salary by Job Role")
        # Fix the tickangle issue by using update_layout
        fig.update_layout(xaxis={'tickangle': 45})
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap for numeric data
    st.subheader("🔥 Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        fig = px.imshow(correlation_matrix, 
                        title="Feature Correlation Heatmap",
                        aspect="auto", 
                        color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation analysis.")

def prediction_page(df):
    st.markdown("## 🔮 Salary Prediction")
    
    # Load trained models
    results, pipelines, _, _, _, _ = train_models()
    
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

def model_comparison_page():
    st.markdown("## 📈 Model Performance Comparison")
    
    results, _, _, _, _, _ = train_models()
    
    if not results:
        st.error("No model results available!")
        return
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results).T
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 R² Score Comparison")
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Training R²', x=comparison_df.index, y=comparison_df['train_r2']))
        fig.add_trace(go.Bar(name='Test R²', x=comparison_df.index, y=comparison_df['test_r2']))
        fig.update_layout(barmode='group', title="R² Score: Higher is Better")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📉 RMSE Comparison")
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Training RMSE', x=comparison_df.index, y=comparison_df['train_rmse']))
        fig.add_trace(go.Bar(name='Test RMSE', x=comparison_df.index, y=comparison_df['test_rmse']))
        fig.update_layout(barmode='group', title="RMSE: Lower is Better")
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

def feature_analysis_page(df):
    st.markdown("## 🧮 Feature Analysis")
    
    results, pipelines, _, _, _, _ = train_models()
    
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
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title=f"Top 15 Most Important Features ({best_model_name})")
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
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

def model_tuning_page():
    st.markdown("## ⚙️ Model Hyperparameter Tuning")
    
    df = load_data()
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
            fig = px.scatter(x=y_test, y=y_pred, 
                            title="Predictions vs Actual Values")
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                   y=[y_test.min(), y_test.max()], 
                                   mode='lines', name='Perfect Prediction'))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error training custom model: {e}")

if __name__ == "__main__":
    main()