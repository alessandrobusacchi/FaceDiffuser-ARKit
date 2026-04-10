#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=slurm-%j.out    # Standard output file (%j is the Job ID)
#SBATCH --error=slurm-%j.err     # Standard error file
#SBATCH --time=10-00:00:00
#SBATCH --priority=TOP
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate venv_thesis


srun python evaluation/evaluation_facediff.py

conda deactivate
