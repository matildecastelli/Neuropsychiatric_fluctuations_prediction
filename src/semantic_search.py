import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer,util
import torch
import argparse
from pathlib import Path
from typing import Tuple
from collections import Counter
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import utils_text_processing
from torch import Tensor
from scipy import stats

def read_and_encode_nfs_items(model_name: str) -> Tuple[pd.DataFrame, Tensor]:
    #store nfs items
    nfs_items = pd.read_csv('./input_files/NFS_items.csv')
    #print(nfs_items)
    pipeline = utils_text_processing.Text_Processing('items')
    nfs_norm = (nfs_items.pipe(pipeline.create_processed_column)
            .pipe(pipeline.lowercasing)
            .pipe(pipeline.remove_punctuation))
    nfs = nfs_norm['processed_text'].tolist()
    # compute embeddings of nfs items
    model = SentenceTransformer(model_name)
    embeddings_nfs = model.encode(nfs,convert_to_tensor=True)
    return nfs_items, embeddings_nfs

def top_k_retrival_nfs(df: pd.DataFrame, model_name: str, n: int, embeddings: Tensor) -> list:
    '''
    Retrieve the top-k most similar nfs items to each speech transcriptions.
    '''
    nfs_items, embeddings_nfs = read_and_encode_nfs_items(model_name)
    predictions = []
    #frequencies_on = np.zeros(20)
    #frequencies_off = np.zeros(20)
    top_k = min(n, embeddings.shape[0]-1)
    for i,query in enumerate(embeddings):
        top_5_labels = []
        labels_sum = 0
        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query, embeddings_nfs)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", df.iloc[i]['Transcription'])
        print(f"Label: {df.iloc[i]['State']}")
        print("\nTop 5 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            j = int(idx.cpu().numpy())
            print(nfs_items.iloc[j]['items'], "(Score: {:.4f})".format(score))
            print(f"Label: {nfs_items.iloc[j]['condition']}")
            top_5_labels.append(nfs_items.iloc[j]['condition'])

        counter = Counter(top_5_labels)
        prediction,_ = counter.most_common(1)[0]
        print(f"predicted label {prediction}")
        predictions.append(prediction)
    return predictions

def top_k_retrieval_patients(df: pd.DataFrame, n: int ,embeddings: Tensor, task: str) -> list:
    df["id"] = df["ID"].apply(lambda x: str(x).split('_')[0])
    if task == 'classification':
        label = 'State'
    else:
        label ='NFS_score'

    predictions = []
    top_k = min(n, embeddings.shape[0]-1)
    for i,query in enumerate(embeddings):
        current_patient_id = df.iloc[i]['id']
        speaker_indexes = df[df['id'] == current_patient_id].index.tolist()
        cosine_similarities = []
        top_5_labels = []
        #cosine-similarity and torch.topk to find the highest k scores
        cos_scores = util.cos_sim(query, embeddings)[0]
        cos_scores[speaker_indexes] = -float('inf') # Exclude the query itself and all speaker samples
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", df.iloc[i]['Transcription'])
        print("\nTop 5 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            j = int(idx.cpu().numpy())
            print(df.iloc[j]['Transcription'], "(Score: {:.4f})".format(score))
            print(f"Label: {df.iloc[j][label]}")
            top_5_labels.append(df.iloc[j][label])
            cosine_similarities.append(score.cpu().numpy())

        print(top_5_labels)

        if task == 'classification':
            counter = Counter(top_5_labels)
            prediction,_ = counter.most_common(1)[0] # most frequent label in the top-k texts
            print(f"predicted label {prediction}")
            predictions.append(prediction)

        else:
            # final predictions based on the weighted average (weight==cosine similarity)of the top k texts
            sum_cosine =  sum(cosine_similarities) # Remove the first similar in the list (cos_sim==1), i.e. the query
            labels_avg = sum(p*s for p, s in zip (top_5_labels,cosine_similarities)) / sum_cosine
            print(f"Predicted label: {labels_avg}")
            predictions.append((labels_avg)) 

    if task == 'classification':
        print(classification_report(df['State'],predictions, digits=3))

    else:
        print(f"\n\nRMSE {np.sqrt(mean_squared_error(df['NFS_score'],predictions))}")
        print(f"R2 {r2_score(df['NFS_score'], predictions)}")
        p,p_value = stats.spearmanr(df['NFS_score'], predictions)
        print(f"Spearman: {p}, p-value: {p_value}")
        absolute_errors = np.abs(df['NFS_score'] - predictions)
        std_absolute_errors = absolute_errors.std()
        quartiles = absolute_errors.quantile([0.25, 0.5, 0.75])
        print(f"MAE {mean_absolute_error(df['NFS_score'], predictions)}, {std_absolute_errors}")
        print(f"Median,1-3 Quartile of Absolute Errors: {quartiles.loc[0.50]:.4f},{quartiles.loc[0.25]:.4f}-{quartiles.loc[0.75]:.4f}")

    return predictions

def main():
    data_path, output_folder, model_name, corpus, n, task = parse_arguments()
    print(f"model_name: {model_name}")
    print(f"corpus: {corpus}")

    df = pd.read_csv(data_path) # dataset with transcriptions
    print(df.head())

    pipeline = utils_text_processing.Text_Processing('Transcription')

    # TEXT PROCESSING
    # Version 1: Apply normalization steps and remove stopwords
    df_norm = (df.pipe(pipeline.create_processed_column)
               .pipe(pipeline.lowercasing)
               .pipe(pipeline.remove_punctuation)
               .pipe(pipeline.spacy_tokenizer_custom)
               .pipe(pipeline.stop_words_removal,ss=True))
    print(df_norm.head())

    # compute embeddings of speeches' transcriptions
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df_norm['processed_text'].tolist(),convert_to_tensor=True)

    if corpus == 'nfs': # classification task using nfs as corpus
        predictions = top_k_retrival_nfs(df_norm,model_name,n,embeddings)
        print(classification_report(df_norm['State'],predictions,digits=3))

    else:
        predictions = top_k_retrieval_patients(df_norm,n,embeddings,task)

    df_predictions = df[['ID','State','NFS_ON','NFS_OFF','NFS_score']].copy()
    df_predictions['prediction'] = predictions

    model_name = model_name.split('/')[-1]
    df_predictions.to_csv(output_folder / f'ss_pred_n_{n}_corpus_{corpus}_task_{task}_{model_name}.csv',index=False)

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

    parser.add_argument('--model',
                        default='multi-qa-mpnet-base-dot-v1',
                        type=str,
                        help='Embedding model from sentence-transformers')

    parser.add_argument('--corpus',
                        default='patients',
                        type=str,
                        help='Select nfs or patients corpus (regression is possible only for patients corpus)')

    parser.add_argument('--top_k',
                        default=5,
                        type=int,
                        help='Number of most similar texts to consider')

    parser.add_argument('--task',
                        default='regression',
                        type=str,
                        help='classification or regression task, i.e. medication classification or NFS prediciton')

    args = parser.parse_args()
    return args.transcriptions_path, args.output_path, args.model, args.corpus, args.top_k, args.task

if __name__ == "__main__":
    main()