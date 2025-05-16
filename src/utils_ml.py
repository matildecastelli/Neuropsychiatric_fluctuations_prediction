from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.base import clone
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from xgboost import XGBRegressor
import lightgbm as lgb
import json

# --------------------------------------------------------------------------- #
# GLOBAL VARIABLES
# --------------------------------------------------------------------------- #

RANDOM_STATE = 0
NGRAM_RANGE = (1, 3) # min and max n-grams
PCA_COMPONENTS = 60 #variance > 90 for tfidf and CountVectorizer
PCA_VARIANCE_THRESHOLD = 0.90

class TextEmbedder:
    """
    Handles text embedding computations using various embedding models (CountVectorizer, TfidfVectorizer,
    and SentenceTransformer) with optional dimensionality reduction.

    Attributes:
        embed_model: The embedding model to be used for text embedding.
        use_pca: Boolean flag to indicate whether dimensionality reduction
            should be applied.
    """
    def __init__(self, embed_model: str, use_pca: bool = False):
        self.embed_model = embed_model
        self.use_pca = use_pca

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.embed_model == "CountVectorizer":
            return self._compute_count_vectorizer(texts)
        elif self.embed_model == "TfidfVectorizer":
            return self._compute_tfidf(texts)
        else:
            return self._compute_sentence_transformer(texts)

    def _compute_count_vectorizer(self, texts: List[str]) -> np.ndarray:
        vectorizer = CountVectorizer(ngram_range=NGRAM_RANGE)
        embeddings = vectorizer.fit_transform(texts)
        return self._apply_pca(embeddings, TruncatedSVD)

    def _compute_tfidf(self, texts: List[str]) -> np.ndarray:
        vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE)
        embeddings = vectorizer.fit_transform(texts)
        return self._apply_pca(embeddings, TruncatedSVD)

    def _compute_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        vectorizer = SentenceTransformer(self.embed_model, trust_remote_code=True)
        embeddings = vectorizer.encode(texts)
        scaled_data = StandardScaler().fit_transform(embeddings)
        return self._apply_pca(scaled_data, PCA)

    def _apply_pca(self, data: np.ndarray, pca_class) -> np.ndarray:
        if not self.use_pca:
            return data
        n_components = PCA_COMPONENTS if pca_class == TruncatedSVD else PCA_VARIANCE_THRESHOLD
        dim_reducer = pca_class(n_components=n_components, random_state=RANDOM_STATE)
        return dim_reducer.fit_transform(data)


class ModelTrainer:
    @staticmethod
    def create_model(model_name: str):
        models = {
            "SVC": svm.SVC(probability=True, random_state=RANDOM_STATE),
            "RandomForestClassifier": RandomForestClassifier(random_state=RANDOM_STATE),
            "MultinomialNB": MultinomialNB(),
            "LogisticRegression": LogisticRegression(),
            "GaussianNB": GaussianNB(),
            "SVR": svm.SVR(),
            "RandomForestRegressor": RandomForestRegressor(random_state=RANDOM_STATE),
            "Ridge": Ridge(),
            "XGBoost": XGBRegressor(random_state=RANDOM_STATE)
            "LightGBM": lgb.LGBMRegressor(random_state=RANDOM_STATE)
        }
        if model_name not in models:
            raise ValueError(f"Model {model_name} not supported")
        return models[model_name]

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           model_name: str, hyperparameter_path: str, scoring: str,
                           probs: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        model = self.create_model(model_name)
        param_grid = self._load_hyperparameters(model_name, hyperparameter_path)

        # Train with cross-validation
        logo = LeaveOneGroupOut()
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                   cv=logo.split(X, y, groups), scoring=scoring)
        grid_search.fit(X, y)

        print("Best parameters found: ", grid_search.best_params_)
        print("Best cross-validation score: ", grid_search.best_score_)

        # Predict using best model
        if probs:
            predictions, probabilities = self._cross_validate_predictions(
                X, y, groups, model, grid_search.best_params_, probs
            )
            return predictions, probabilities

        predictions = self._cross_validate_predictions(
            X, y, groups, model, grid_search.best_params_, probs
        )
        return predictions

    def _load_hyperparameters(self, model_name: str, hyperparameter_path: str) -> dict:
        with open(hyperparameter_path, 'r') as f:
            return json.load(f)[model_name]

    def _cross_validate_predictions(self, X: np.ndarray, y: np.ndarray,
                                    groups: np.ndarray, model,
                                    best_params: dict, probs) -> Tuple[np.ndarray, np.ndarray]:
        predictions = np.zeros_like(y)
        probabilities = np.zeros_like(y, dtype=float)
        logo = LeaveOneGroupOut()

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            model = clone(model).set_params(**best_params)
            model.fit(X_train, y_train)

            predictions[test_idx] = model.predict(X_test)
            if probs:
                probabilities[test_idx] = model.predict_proba(X_test)[:, 1]
                return predictions, probabilities

        return predictions