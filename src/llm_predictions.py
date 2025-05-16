import argparse
import re
from pathlib import Path
from textwrap import dedent
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import classification_report
import os
import utils_llm

# --------------------------------------------------------------------------- #
# GLOBAL VARIABLES
# --------------------------------------------------------------------------- #
INPUT_FOLDER   = Path("./input_files/")
DEFAULT_MODEL  = "google/gemma-2-9b"
EMB_MODEL      = "multi-qa-mpnet-base-dot-v1"
DATA_CSV       = Path("../data/Levodopa_NFS_en.csv")
OUT_DIR        = Path("./output")
FEW_SHOT_K     = 3
os.environ['HF_TOKEN'] = "" #insert the access token here

def parse_args():
    parser = argparse.ArgumentParser("Few-shot LLM classification/regression")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help=" Insert name of hugging face pre-trained model for causal inference (google/gemma-2-9b or meta-llama/Llama-3.1-8B)")
    parser.add_argument("--embedding_model_name", type=str, default=EMB_MODEL, help=" Insert name of sentence-transformer embedding model for similarity calculation")
    parser.add_argument("--task", type=str, default="classification", choices=("classification", "regression"), help=" Task to perform: classification or regression. Default is classification.")
    parser.add_argument('--data_path', default=DATA_CSV, type = Path,
                        help='Path of the csv data file containing the transcriptions')
    parser.add_argument('--output_path', default = OUT_DIR, type= str)
    parser.add_argument('--n_few_shot', default = FEW_SHOT_K, type= int)
    return parser.parse_args()

# Classification or regression task
def get_task_configuration(task_type: str) -> tuple[Path, str]:
    """
    Get the prompt file path and ground truth column name for a given task type.

    Args:
        task_type: Type of task ('classification' or 'regression')

    Returns:
        tuple: (prompt_file_path, ground_truth_column_name)

    Raises:
        ValueError: If task_type is not supported
    """
    task_configs = {
        "classification": (INPUT_FOLDER / "prompts_classification.txt", "State"),
        "regression": (INPUT_FOLDER / "prompts_regression.txt", "NFS_score")
    }

    task_type = task_type.lower()
    if task_type not in task_configs:
        raise ValueError("Invalid task. Choose from 'classification' or 'regression'.")

    return task_configs[task_type]


def extract_value_from_prediction(sentence: str, text: str = "State") -> str:
    '''
    Given a text, find the substring "State" or "NFS_score" and return a single word representing the single value predicted.

    Args:
        sentence: The prediction text to extract value from
        text: The type of value to extract ("State" or "NFS_score")

    Returns:
        str: Extracted prediction value
    '''
   # pattern = r"Score:\s*(\w+)"
    pattern = rf"{re.escape(text)}:\s*(\w+)"
    matches = re.findall(pattern, sentence)
    if matches:
        category_word = matches[-1]
    else:
        category_word = 'Not found'
    return category_word


def build_few_shot_block(texts: str, labels: List[Union[str, int]], text: str) -> str:
    """
    Turn two equal‑length iterables into a few‑shot block like
        Transcription: ...
        Class: ...

        Transcription: ...
        Class: ...

    Args:
        texts: List of transcription texts
        labels: List of corresponding labels
        text: Label type text ("State" or "NFS_score")

    Returns:
        str: Formatted few-shot block text
    """
    return '\n'.join(
        f"Transcription: {t}\n{text}:{c}"
        for t, c in zip(texts, labels)
    )

def main():
    args = parse_args()
    model_handler = utils_llm.LLMHandler(
        args.model_name,
        args.embedding_model_name,
    )
    # Read input data
    df = pd.read_csv(args.data_path)

    # Usage
    prompt_path, ground_truth_column = get_task_configuration(args.task)

    # Retrieve prompts based on task type
    with open(prompt_path, 'r', encoding="utf-8") as f:
        prompts = [line.rstrip("\n") for line in f]

    # Compute embeddings for all transcriptions in the dataframe
    model_sentence = SentenceTransformer(args.embedding_model_name)
    embeddings = model_sentence.encode(df['Transcription'].tolist(),convert_to_tensor=True)
    cos_sim_matrix = util.cos_sim(embeddings, embeddings)
    print(f"Similarity matrix: {cos_sim_matrix.shape}")
    # the diagonal corresponds to the sentence itself, so we set it to -inf tpo ignore in top-k
    cos_sim_matrix.fill_diagonal_(-float('inf'))

    predictions, probabilities = [], []
    for i,row in df.iterrows():

        query = row['Transcription']
        _, top_indices = torch.topk(cos_sim_matrix[i], k=args.n_few_shot)
        neighbours = df.iloc[top_indices.cpu().numpy()]
        few_shot_block  = build_few_shot_block(
            neighbours["Transcription"].values,
            neighbours[ground_truth_column].values,
            ground_truth_column
        )
        iter_prob, iter_pred = [], []
        for prompt in prompts:
            #On-off medication parkinson
            prompt_few_shot = dedent(f'''\n{prompt} 
            {few_shot_block}
            Transcriptions: {query}
            {ground_truth_column}:''')
            print(prompt_few_shot)

            prediction, probs = model_handler.generate_prediction(prompt_few_shot)
            iter_pred.append(extract_value_from_prediction(prediction, ground_truth_column))
            iter_prob.append(probs)

        probabilities.append(iter_prob)
        predictions.append(iter_pred)

    # Store the entire prediction
    df['prediction'] = predictions
    df['probabilities'] = probabilities
    output_file = f'{args.model_name.split("/")[-1]}_{args.task}_n_{args.n_few_shot}.csv'
    args.output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path / output_file, index = False)

if __name__ == "__main__":
    main()
