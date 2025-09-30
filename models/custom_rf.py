"""
Enhanced Random Forest Regressor Implementation
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor

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