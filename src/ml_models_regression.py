from sentence_transformers import SentenceTransformer
import utils_text_processing
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from scipy import stats
import argparse
import json

RANDOM_STATE = 0
N_GRAM_MIN = 1
N_GRAM_MAX = 3

#def train_and_predict_best_model(x,y,groups):
def compute_embeddings(X,embed_model: str,pca_flag: bool):
    match embed_model:
        case "CountVectorizer":
            vectorizer = CountVectorizer(ngram_range=(N_GRAM_MIN,N_GRAM_MAX))
            X_embed = vectorizer.fit_transform(X)
            if pca_flag:
                #dim_red = TruncatedSVD(n_components=60,random_state=RANDOM_STATE)
                dim_red = PCA(n_components=60, svd_solver='arpack')
                X_embed = dim_red.fit_transform(X_embed)

        case "TfidfVectorizer":
            vectorizer = TfidfVectorizer(ngram_range=(N_GRAM_MIN,N_GRAM_MAX))
            X_embed = vectorizer.fit_transform(X)
            print(f"Before dim reduction:{X_embed.shape}")
            if pca_flag:
                #dim_red = TruncatedSVD(n_components=60, random_state=RANDOM_STATE)
                dim_red = PCA(n_components=60, svd_solver='arpack')
                X_embed = dim_red.fit_transform(X_embed)
        case _:
            vectorizer = SentenceTransformer(embed_model, trust_remote_code=True)
            X_embed = vectorizer.encode(X)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(X_embed)
            if pca_flag:
                dim_red = PCA(n_components=41)# svd_solver='full')  # keep an explaiined variance > 90
                X_embed = dim_red.fit_transform(scaled_data)
                print("Explained Variance Ratio:\n", dim_red.explained_variance_ratio_.sum())
    return X_embed


def main():
    data_path, output_folder, vector_model, pca, model_name = parse_arguments()
    df = pd.read_csv(data_path)

    # extract only ids without MedOFF and MedON to group them
    df["id"] = df["ID"].apply(lambda x: str(x).split('_')[0])
    pipeline = utils_text_processing.Text_Processing('Transcription')

    # TEXT PROCESSING
    # Version 1: Apply normalization steps and remove stopwords
    df_norm = (df.pipe(pipeline.create_processed_column)
               .pipe(pipeline.lowercasing)
               .pipe(pipeline.remove_punctuation)
               .pipe(pipeline.spacy_tokenizer_custom)
               .pipe(pipeline.stop_words_removal))
    print(df_norm.head())

    X = df_norm['processed_text'] #transcriptions

    # EMBEDDINGS COMPUTATION
    X_embed = compute_embeddings(X,vector_model,pca)

    #if pca: print("Explained Variance Ratio:\n", dim_red.explained_variance_ratio_.sum())
    print(f'\nembeddings model: {vector_model}\n')
    print(f"emb shape: {X_embed.shape}")
    print(f"ML model: {model_name}")

    # ML MODEL TRAINING
    # output column
    y = df_norm["NFS_score"]
    y = y.values

    # groups for Leave-One-Group-Out cross-validation
    groups = df_norm["id"]

    # HYPERPARAMETERS TUNING
    # Dictionary for hyperparameters tuning
    with open('./input_files/hyperparamters_ml_regression.json', 'r') as json_file:
        param_grid = json.load(json_file)

    # Initialize Leave-One-Group-Out cross-validator
    logo = LeaveOneGroupOut()

    # Initialize the model
    match model_name:
        case "SVR":
            reg = svm.SVR()
        case "RandomForestRegressor":
            reg = RandomForestRegressor(random_state=RANDOM_STATE)
        case "Ridge":
            reg = Ridge()
        case "XGBoost":
            reg = XGBRegressor(random_state=RANDOM_STATE)
        case "LightGBM":
            reg = lgb.LGBMRegressor(random_state=RANDOM_STATE)
        case _:
            raise ValueError(f"Model {model_name} not supported")


    # Hyperperameter Tuning vis GridSearch
    grid_search = GridSearchCV(estimator=reg, param_grid=param_grid[model_name], cv=logo.split(X_embed, y, groups), scoring='neg_root_mean_squared_error')
    # Fit the model
    grid_search.fit(X_embed, y)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    # Best model predictions
    # Double check and predict
    reg.set_params(**grid_search.best_params_)

    # Store predictions
    predictions = np.zeros_like(y)
    pred_prob = np.zeros_like(y, dtype=float)

    # Perform Leave-One-Group-Out cross-validation
    for train_idx, test_idx in logo.split(X_embed, y, groups):
        X_train, X_test = X_embed[train_idx], X_embed[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        #print(df_norm.iloc[test_idx]['ID'])

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        #print(f"prediction: {y_pred}, real: {y_test}")

        # Store the predictions
        predictions[test_idx] = y_pred


    #Clip negative values to 0 and values >60 to 60
    predictions = np.clip(predictions, 0, 60)
    df_norm["prediction"] = predictions
    print(df_norm[["ID", "State", "NFS_score","prediction"]])
    absolute_errors = np.abs(df['NFS_score'] - df['prediction'])
    std_absolute_errors = absolute_errors.std()
    quartiles = absolute_errors.quantile([0.25, 0.5, 0.75])
    # correlation
    p,p_value = stats.spearmanr(y, predictions)
    print(f"MAE: {mean_absolute_error(y, predictions)}, sd {std_absolute_errors:.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y, predictions))}")
    print(f"R^2: {r2_score(y, predictions)}")
    print(f"Spearman: {p}, p-value: {p_value}")
    print(f"Median,1-3 Quartile of Absolute Errors: {quartiles[0.50]:.4f},{quartiles[0.25]:.4f},{quartiles[0.75]:.4f}")

    vector_model = vector_model.split('/')[-1]
    #df_norm[["ID", "State", "NFS_score","prediction"]].to_csv(f"./output/regression_{model_name}_{vector_model}.csv", index=False)
def parse_arguments():
    """
    Retrieve the arguments given as input to the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcriptions_path',
                        default='../../data/Parkinson/Levodopa_transcriptions/Levodopa_NFS_en.csv',
                        type=Path,
                        help='Path of the csv file containing the transcriptions.')

    parser.add_argument('--output_path',
                        default='./output',
                        type=Path,
                        help='Path of the folder containing output files')

    parser.add_argument('--vector_model',
                        default='multi-qa-mpnet-base-dot-v1',
                        type=str,
                        help='Embedding model: CountVectorizer, TfidfVectorizer, multi-qa-mpnet-base-dot-v1, Alibaba-NLP/gte-Qwen2-1.5B-instruct, dunzhang/stella_en_1.5B_v5')

    parser.add_argument('--pca',
                        action='store_true',
                        help='Do dimensionality reduction')

    parser.add_argument('--model_name',
                        default='RandomForestRegressor',
                        type=str,
                        help='ML model: RandomForestRegressor, Ridge, SVR')

    args = parser.parse_args()
    return args.transcriptions_path, args.output_path, args.vector_model, args.pca, args.model_name

if __name__ == "__main__":
    main()