from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, GroupKFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from xgboost import XGBRegressor
import json

# --------------------------------------------------------------------------- #
# GLOBAL VARIABLES
# --------------------------------------------------------------------------- #

hf_token = ""   # Insert the access token here
RANDOM_STATE = 0
PCA_COMPONENTS = 40 #variance > 90 
PCA_VARIANCE_THRESHOLD = 0.90

class TextEmbedder:

    def __init__(self, embed_model: str):
        self.embed_model = embed_model

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        vectorizer = SentenceTransformer(self.embed_model, trust_remote_code=True)
        embeddings = vectorizer.encode(texts)
        return embeddings 

class ModelTrainer:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def _define_pipeline(self, pca, model) -> Pipeline:
        steps = []
        steps.append(('scaler', StandardScaler()))
        if pca:
            steps.append(('pca', PCA(n_components=PCA_VARIANCE_THRESHOLD, svd_solver='full')))
        steps.append(('model', model))
        pipeline = Pipeline(steps)
        return pipeline

    def _create_model(self):
        models = {
            "SVC": svm.SVC(probability=True, random_state=RANDOM_STATE),
            "RandomForestClassifier": RandomForestClassifier(random_state=RANDOM_STATE),
            "MultinomialNB": MultinomialNB(),
            "LogisticRegression": LogisticRegression(),
            "GaussianNB": GaussianNB(),
            "SVR": svm.SVR(),
            "RandomForestRegressor": RandomForestRegressor(random_state=RANDOM_STATE),
            "Ridge": Ridge(),
            "XGBoost": XGBRegressor(random_state=RANDOM_STATE),
            "LightGBM": lgb.LGBMRegressor(random_state=RANDOM_STATE)
        }
        if self.model_name not in models:
            raise ValueError(f"Model {self.model_name} not supported")
        return models[self.model_name]

    def train(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           hyperparameter_path: str, scoring: str,
                           use_pca: bool, pca_type, probs = False) -> Tuple[np.ndarray, np.ndarray]:
        model = self._create_model()
        param_grid = self._load_hyperparameters(self.model_name, hyperparameter_path)
        print(f"N. Folds: {len(np.unique(groups))}")

        # Leave one patient out cross-validation
        logo = LeaveOneGroupOut()      
        inner = GroupKFold(n_splits=5)
   
        predictions = np.full_like(y, -1)
        probabilities = np.zeros_like(y, dtype=float)
        fold = 0
        for train_idx, test_idx in logo.split(X, y, groups):
            fold += 1
            print(f"Fold {fold}")
            print(f"  Test patient(s): {groups[test_idx].values[0]} ({len(test_idx)} samples)")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            pipeline = self._define_pipeline(use_pca, model)
            groups_train = groups[train_idx]
            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,scoring=scoring, cv=inner.split(X_train, y_train, groups_train))
            grid_search.fit(X_train, y_train)
            prediction = grid_search.best_estimator_.predict(X_test)
            predictions[test_idx] = prediction
            if probs:
                probabilities[test_idx] = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
            else:
                probabilities[test_idx] = np.nan

        #### hyperparameters logo whole data#####
        #grid = GridSearchCV(estimator=pipeline, param_grid=param_grid,cv=logo.split(X, y, groups), scoring=scoring)
        #grid.fit(X, y)
        #print("Best parameters found: ", grid.best_params_)

        return predictions, probabilities

    def _load_hyperparameters(self, model_name: str, hyperparameter_path: str) -> dict:
        with open(hyperparameter_path, 'r') as f:
            return json.load(f)[model_name]
