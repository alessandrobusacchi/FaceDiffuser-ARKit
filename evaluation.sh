#!/bin/bash
#SBATCH --job-name=facediffuser_arkit   # Name of your job
#SBATCH --output=slurm-%j.out    # Standard output file (%j is the Job ID)
#SBATCH --error=slurm-%j.err     # Standard error file
#SBATCH --time=0-10:00:00        # Wall time (Day-HH:MM:SS)
#SBATCH --priority=TOP

# Execution Commands
# ------------------

# 1. Load any necessary modules (e.g., Python, CUDA)

# 2. Activate your environment (if applicable)
source ~/.bashrc
conda activate venv_thesis

# 3. Execute your program (using all requested CPUs)
srun python main.py

# 4. Deactivate the environment (optional but good practice)
conda deactivate
