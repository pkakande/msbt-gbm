#!/bin/bash
#SBATCH --job-name=getOrgData
#SBATCH --output=GBMOrgData.log
#SBATCH --error=GBMOrgData.err
#SBATCH --nodes=1

unset XDG_RUNTIME_DIR
module purge
module load anaconda3-2021.05-gcc-8.5.0-s7wekg6
source activate
conda activate /etc/ace-data/home/pkakande/.conda/envs/dev

echo "Job started"
python /etc/ace-data/home/pkakande/pkakande/msbt-gbm/src/getOrganisedGBMData.py

echo "Job ended"
