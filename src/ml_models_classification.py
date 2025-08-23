import utils_text_processing, utils_ml
from pathlib import Path
import numpy as np
import argparse
from typing import Tuple, Union, List
import pandas as pd
import json
from sklearn.metrics import classification_report, roc_auc_score

RANDOM_STATE = 0
HYPERPARAMETERS_PATH = './input_files/hyperparamters_ml.json'
pca_type = 'PCA'

def main():
    args = parse_arguments()
    df = pd.read_csv(args.transcriptions_path)

    # extract only ids without MedOFF and MedON to group them
    df["id"] = df["ID"].apply(lambda x: str(x).split('_')[0])
    pipeline = utils_text_processing.Text_Processing('Transcription')

    ########## TEXT PROCESSING ##############
    # Version 1: Apply normalization steps and remove stopwords
    df_norm = (df.pipe(pipeline.create_processed_column)
               .pipe(pipeline.lowercasing)
               .pipe(pipeline.remove_punctuation)
               .pipe(pipeline.spacy_tokenizer_custom)
               .pipe(pipeline.stop_words_removal))
    print(df_norm.head())

    # Embeddings computation
    embedder = utils_ml.TextEmbedder(args.vector_model)
    X_embed = embedder.compute_embeddings(df_norm['processed_text'])

    print(f'\nembeddings model: {args.vector_model}\n')
    print(f"emb shape: {X_embed.shape}")
    print(f"ML model: {args.model_name}")

    mapping = {'ON': 1, 'OFF': 0}
    y = df_norm['State'].map(mapping) 

    # ML MODEL TRAINING
    # output column
    trainer = utils_ml.ModelTrainer(args.model_name)
    predictions, probabilities = trainer.train(
        X_embed, y, df_norm["id"], HYPERPARAMETERS_PATH,
        'accuracy', args.pca, pca_type, probs=True
    ) # grouped based on id

    print(classification_report(y, predictions, digits=3))
    print(roc_auc_score(y, probabilities))

    df_norm["Predicted_State"] = predictions
    df_norm["probabilities"] = probabilities
    vector_model = args.vector_model.split('/')[-1]
    df_norm[["ID", "State", "Predicted_State"]].to_csv(f"../output/{args.model_name}_{vector_model}.csv", index=False)

def parse_arguments():
    """
    Retrieve the arguments given as input to the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcriptions_path',
                        default='../data/Levodopa_NFS_en_medication_removed.csv',
                        type=Path,
                        help='Path of the csv file containing the transcriptions.')

    parser.add_argument('--output_path',
                        default='../output',
                        type=Path,
                        help='Path of the folder containing output files')

    parser.add_argument('--vector_model',
                        default='multi-qa-mpnet-base-dot-v1',
                        type=str,
                        help='Embedding model: multi-qa-mpnet-base-dot-v1, Alibaba-NLP/gte-Qwen2-1.5B-instruct, dunzhang/stella_en_1.5B_v5')

    parser.add_argument('--pca',
                        action='store_true',
                        help='Do dimensionality reduction')

    parser.add_argument('--model_name',
                        default='RandomForestClassifier',
                        type=str,
                        help='ML model: GaussianNB ,MultinomialNB, SVC, RandomForestClassifier,LogisticRegression')

    return parser.parse_args()

if __name__ == "__main__":
    main()