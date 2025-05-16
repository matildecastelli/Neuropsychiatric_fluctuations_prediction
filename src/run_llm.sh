#!/bin/bash

#SBATCH --job-name="llm_prediction"
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=24G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --error=%x_%j.err

module load CUDA/12.1
module load Anaconda3
eval "$(conda shell.bash hook)"

# Load environment
conda activate asr
pip install flash_attn>=2.5.6
echo "start"

srun python3 llm_predictions.py --task regression
