import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------
# Custom Decision Tree Regressor (simplified)
# -----------------------------
class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def mse(self, y):
        return np.var(y) * len(y)

    def best_split(self, X, y):
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
            
        best_feat, best_thresh, best_score = None, None, float("inf")
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for thresh in thresholds:
                left_idx = X[:, feature] <= thresh
                right_idx = ~left_idx
                if sum(left_idx) < self.min_samples_split or sum(right_idx) < self.min_samples_split:
                    continue
                left_mse = self.mse(y[left_idx])
                right_mse = self.mse(y[right_idx])
                score = (left_mse + right_mse)

                if score < best_score:
                    best_feat, best_thresh, best_score = feature, thresh, score
        return best_feat, best_thresh

    def build_tree(self, X, y, depth=0):
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
            
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split:
            return np.mean(y)

        feat, thresh = self.best_split(X, y)
        if feat is None:
            return np.mean(y)

        left_idx = X[:, feat] <= thresh
        right_idx = ~left_idx

        return {
            "feature": feat,
            "threshold": thresh,
            "left": self.build_tree(X[left_idx], y[left_idx], depth + 1),
            "right": self.build_tree(X[right_idx], y[right_idx], depth + 1)
        }

    def fit(self, X, y):
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        self.tree = self.build_tree(X, y)

    def predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree["feature"]] <= tree["threshold"]:
            return self.predict_one(x, tree["left"])
        else:
            return self.predict_one(x, tree["right"])

    def predict(self, X):
        return np.array([self.predict_one(sample, self.tree) for sample in X])


# -----------------------------
# Custom Random Forest Regressor
# -----------------------------
class RandomForestRegressorScratch:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features="sqrt"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.features_idx = []

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

        for _ in range(self.n_estimators):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]

            feat_idx = np.random.choice(n_features, max_feats, replace=False)
            self.features_idx.append(feat_idx)

            tree = DecisionTreeRegressorScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, feat_idx], y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Convert sparse matrix to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
            
        preds = np.zeros((self.n_estimators, X.shape[0]))
        for i, tree in enumerate(self.trees):
            preds[i] = tree.predict(X[:, self.features_idx[i]])
        return np.mean(preds, axis=0)


# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("employee_salaries_india.csv")
    return df

df = load_data()

# Features & target
X = df.drop("salary_in_inr", axis=1)
y = df["salary_in_inr"].values

# Separate categorical & numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Preprocessing
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Custom model
model = RandomForestRegressorScratch(n_estimators=20, max_depth=10)

# Pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", model)])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🇮🇳 Employee Salary Prediction App (India)")

st.write("Enter employee details to predict *salary in INR (₹)*:")

experience = st.slider("Years of Experience", 0, 40, 3)
education = st.selectbox("Education Level", df["education"].unique())
location = st.selectbox("Location", df["location"].unique())
company_size = st.selectbox("Company Size", df["company_size"].unique())
industry = st.selectbox("Industry", df["industry"].unique())
job_role = st.selectbox("Job Role", df["job_role"].unique())
remote = st.radio("Remote Work?", ["Yes", "No"])

input_data = pd.DataFrame({
    "experience": [experience],
    "education": [education],
    "location": [location],
    "company_size": [company_size],
    "industry": [industry],
    "job_role": [job_role],
    "remote": [remote]
})

if st.button("Predict Salary"):
    prediction = pipeline.predict(input_data)[0]
    st.success(f"💰 Estimated Salary: ₹{prediction:,.0f}")