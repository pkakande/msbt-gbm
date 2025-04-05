#!/bin/bash
#SBATCH --job-name=createEnv
#SBATCH --output=createEnv.log
#SBATCH --error=createEnv.err
#SBATCH --nodes=1

unset XDG_RUNTIME_DIR
module purge
module load anaconda3-2021.05-gcc-8.5.0-s7wekg6
source activate

echo "Job started"

# Create a new conda environment and install the packages
conda create -n research python=3.9 -y
source activate research
conda install -c conda-forge scanpy seaborn gseapy numpy pandas ipykernel -y
pip install sanbomics
pip install pydeseq2

# Double-check installation of ipykernel
conda list | grep ipykernel

# Register the conda environment as a Jupyter kernel
python -m ipykernel install --user --name research --display-name "Python (research)"

echo "Job ended"
