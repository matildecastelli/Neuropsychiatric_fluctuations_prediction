from sentence_transformers import SentenceTransformer
import utils_text_processing
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
import argparse
import json

RANDOM_STATE = 0
N_GRAM_MIN = 1
N_GRAM_MAX = 3

def compute_embeddings(X,embed_model: str,pca_flag: bool):
    match embed_model:
        case "CountVectorizer":
            vectorizer = CountVectorizer(ngram_range=(N_GRAM_MIN,N_GRAM_MAX))
            X_embed = vectorizer.fit_transform(X)
            if pca_flag:
                #dim_red = TruncatedSVD(n_components=60,random_state=RANDOM_STATE)
                dim_red = TruncatedSVD(n_components=60,random_state=RANDOM_STATE)
                X_embed = dim_red.fit_transform(X_embed)

        case "TfidfVectorizer":
            vectorizer = TfidfVectorizer(ngram_range=(N_GRAM_MIN,N_GRAM_MAX))
            X_embed = vectorizer.fit_transform(X)
            print(f"Before dim reduction:{X_embed.shape}")
            if pca_flag:
                dim_red = TruncatedSVD(n_components=60, random_state=RANDOM_STATE)
                X_embed = dim_red.fit_transform(X_embed)
        case _:
            vectorizer = SentenceTransformer(embed_model, trust_remote_code=True)
            X_embed = vectorizer.encode(X)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(X_embed)
            if pca_flag:
                dim_red = PCA(n_components=0.91, svd_solver='full')  # keep an explaiined variance > 90
                X_embed = dim_red.fit_transform(scaled_data)
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
    y = (df_norm["State"] == "ON").astype(int)
    y = y.values

    # groups for Leave-One-Group-Out cross-validation
    groups = df_norm["id"]

    # HYPERPARAMETERS TUNING
    # Dictionary for hyperparameters tuning
    with open('./input_files/hyperparamters_ml.json', 'r') as json_file:
        param_grid = json.load(json_file)

    # Initialize Leave-One-Group-Out cross-validator
    logo = LeaveOneGroupOut()

    # Initialize the model
    match model_name:
        case "SVC":
            clf = svm.SVC(probability=True, random_state=RANDOM_STATE)
        case "RandomForestClassifier":
            clf = RandomForestClassifier(random_state=RANDOM_STATE)
        case "MultinomialNB":
            clf = MultinomialNB()
        case "LogisticRegression":
            clf = LogisticRegression()
        case "GaussianNB":
            clf = GaussianNB()
        case _:
            raise ValueError(f"Model {model_name} not supported")

    # Hyperperameter Tuning vis GridSearch
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid[model_name], cv=logo.split(X_embed, y, groups), scoring='accuracy')
    # Fit the model
    grid_search.fit(X_embed, y)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    # Best model predictions
    # Double check and predict
    #params = {'C': 0.001, 'gamma': 0.1, 'kernel': 'sigmoid'}
    clf.set_params(**grid_search.best_params_)

    # Store predictions
    predictions = np.zeros_like(y)
    pred_prob = np.zeros_like(y, dtype=float)

    # Perform Leave-One-Group-Out cross-validation
    for train_idx, test_idx in logo.split(X_embed, y, groups):
        X_train, X_test = X_embed[train_idx], X_embed[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        #print(df_norm.iloc[test_idx]['ID'])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:,1]
        #print(f"prediction: {y_pred}, real: {y_test}")

        # Store the predictions
        predictions[test_idx] = y_pred
        pred_prob[test_idx] = y_prob

    # Print the predicted states
    df_norm["Predicted_State"] = np.where(predictions == 1, "ON", "OFF")
    df_norm["probabilities"] = pred_prob
    print(df_norm[["ID", "State", "Predicted_State"]])

    print(pred_prob)
    print(classification_report(y, predictions, digits=3))
    print(roc_auc_score(y, pred_prob))
    vector_model = vector_model.split('/')[-1]
    #df_norm[["ID", "State", "Predicted_State","probabilities"]].to_csv(f"./output/{model_name}_{vector_model}.csv", index=False)


    group_size = 10
    n_samples = df_norm.shape[0]
    n_groups = n_samples // group_size

    # Create groups for cross-validation
    groups = np.arange(n_samples) // group_size

    # Store evaluation metrics
    accuracies = []
    auc_scores = []
    match model_name:
        case "SVC":
            clf1 = svm.SVC(probability=True, random_state=RANDOM_STATE)
        case "RandomForestClassifier":
            clf1 = RandomForestClassifier(random_state=RANDOM_STATE)
        case "MultinomialNB":
            clf1 = MultinomialNB()
        case "LogisticRegression":
            clf1 = LogisticRegression(random_state=RANDOM_STATE)
        case "GaussianNB":
            clf1 = GaussianNB()
        case _:
            raise ValueError(f"Model {model_name} not supported")

    clf1.set_params(**grid_search.best_params_)
    predictions = np.zeros_like(y)
    predictions = np.zeros_like(y)
    pred_prob = np.zeros_like(y, dtype=float)

    # Perform custom cross-validation
    for i in range(n_groups):
        # Split into training and test sets
        test_mask = (groups == i)
        train_mask = ~test_mask

        X_train, X_test = X_embed[train_mask], X_embed[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Train the model
        clf1.fit(X_train, y_train)

        # Evaluate the model
        y_pred = clf1.predict(X_test)
        y_prob = clf1.predict_proba(X_test)[:, 1]
        predictions[test_mask] = y_pred
        pred_prob[test_mask] = y_prob

        # Store metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_prob))

        print(f"Fold {i + 1}: Accuracy = {accuracies[-1]:.4f}, ROC AUC = {auc_scores[-1]:.4f}")

    # Print overall performance
    print("\nOverall Performance:")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Mean ROC AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    df_norm["Predicted_State"] = np.where(predictions == 1, "ON", "OFF")
    df_norm["probabilities"] = pred_prob
    print(df_norm[["ID", "State", "Predicted_State"]])

    print(pred_prob)
    print(classification_report(y, predictions, digits=3))
    print(roc_auc_score(y, pred_prob))
    vector_model = vector_model.split('/')[-1]
    #df_norm[["ID", "State", "Predicted_State","probabilities"]].to_csv(f"./output/{model_name}_{vector_model}.csv", index=False)

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
                        default='RandomForestClassifier',
                        type=str,
                        help='ML model: GaussianNB ,MultinomialNB, SVC, RandomForestClassifier,LogisticRegression')

    args = parser.parse_args()
    return args.transcriptions_path, args.output_path, args.vector_model, args.pca, args.model_name

if __name__ == "__main__":
    main()