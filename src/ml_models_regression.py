import utils_text_processing, utils_ml
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from scipy import stats
import argparse

RANDOM_STATE = 0
HYPERPARAMETERS_PATH = './input_files/hyperparamters_ml_regression.json'
pca_type = 'PCA'


def main():
    args = parse_arguments()
    df = pd.read_csv(args.transcriptions_path)

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
    embedder = utils_ml.TextEmbedder(args.vector_model)
    X_embed = embedder.compute_embeddings(X)

    print(f'\nembeddings model: {args.vector_model}\n')
    print(f"emb shape: {X_embed.shape}")
    print(f"ML model: {args.model_name}")

    y = df_norm["NFS_score"].values

    # ML MODEL TRAINING
    # output column
    trainer = utils_ml.ModelTrainer(args.model_name)
    predictions, probabilities = trainer.train(
        X_embed, y, df_norm["id"], HYPERPARAMETERS_PATH,
        'neg_root_mean_squared_error', args.pca, pca_type, probs=False
    ) # grouped based on id

    #Clip negative values to 0 and values >60 to 60
    predictions = np.clip(predictions, 0, 60)
    df_norm["prediction"] = predictions
    absolute_errors = np.abs(df_norm['NFS_score'] - df_norm['prediction'])
    std_absolute_errors = absolute_errors.std()
    quartiles = absolute_errors.quantile([0.25, 0.5, 0.75])
    # correlation
    p,p_value = stats.spearmanr(y, predictions)
    print(f"MAE: {mean_absolute_error(y, predictions)}, sd {std_absolute_errors:.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y, predictions))}")
    print(f"R^2: {r2_score(y, predictions)}")
    print(f"Spearman: {p}, p-value: {p_value}")
    print(f"Median,1-3 Quartile of Absolute Errors: {quartiles[0.50]:.4f},{quartiles[0.25]:.4f},{quartiles[0.75]:.4f}")
    vector_model = args.vector_model.split('/')[-1]
    df_norm[["ID", "State", "NFS_score","prediction"]].to_csv(f"../output/regression_{args.model_name}_{vector_model}.csv", index=False)

def parse_arguments():
    """
    Retrieve the arguments given as input to the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcriptions_path',
                        default='../data/Levodopa_NFS_en_medication_removed.csv',
                        type=Path,
                        help='Path of the csv file containing the transcriptions.')

    parser.add_argument('--hyperparameters_path',
                        default=HYPERPARAMETERS_PATH,
                        type=Path,
                        help='Path of the json files containing the hyperparameters ranges')

    parser.add_argument('--vector_model',
                        default='multi-qa-mpnet-base-dot-v1',
                        type=str,
                        help='Embedding model: multi-qa-mpnet-base-dot-v1, Alibaba-NLP/gte-Qwen2-1.5B-instruct, dunzhang/stella_en_1.5B_v5')

    parser.add_argument('--pca',
                        action='store_true',
                        help='Do dimensionality reduction')

    parser.add_argument('--model_name',
                        default='RandomForestRegressor',
                        type=str,
                        help='ML model: RandomForestRegressor, Ridge, SVR, XGBoost, LightGBM')

    return parser.parse_args()

if __name__ == "__main__":
    main()