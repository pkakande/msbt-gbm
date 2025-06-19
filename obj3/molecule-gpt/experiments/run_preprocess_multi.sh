#!/usr/bin/env bash
#SBATCH --job-name=preprocess
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/preprocess_%j.log

# run_preprocess_multi.sh
#
# SLURM wrapper to spin up a Ray cluster across multiple nodes,
# run `src/preprocess.py`, then tear down the cluster.
#
# Requirements:
#  - A top-level VERSION file.
#  - Conda env "molgpt_py3922" with ray, rdkit, selfies, pandas, etc.
#  - data/raw/ and data/processed/ directories exist.

# ────────────────────────────────────────────────────────────
# Echo pipeline version
# ────────────────────────────────────────────────────────────
VERSION=$(<"$SLURM_SUBMIT_DIR/VERSION")
echo "[run_preprocess_multi.sh] version: v$VERSION"

# ────────────────────────────────────────────────────────────
# Configurable timing params
# ────────────────────────────────────────────────────────────
HEAD_START_WAIT=30    # how long head waits for workers
WORKER_RETRIES=60     # number of worker polls
WORKER_SLEEP=2        # delay between polls

# ────────────────────────────────────────────────────────────
# Environment setup & diagnostics
# ────────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"                                    # project root
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate molgpt_py3922

echo "=== ENVIRONMENT DIAGNOSTICS ==="
echo "Project root: $SLURM_SUBMIT_DIR"
echo "Python:       $(which python) ($(python --version 2>&1))"
echo "Ray CLI:      $(which ray  || echo not-installed)"
echo "Ray version:  $(ray --version 2>&1 || echo n/a)"
echo "================================"

# ────────────────────────────────────────────────────────────
# Compute CPU allocation per task
# ────────────────────────────────────────────────────────────
TOTAL_CORES=$SLURM_CPUS_ON_NODE
RESERVED=8
USABLE=$(( TOTAL_CORES - RESERVED ))
TASKS=$SLURM_NTASKS_PER_NODE
CPUS_PER_TASK=$(( USABLE / TASKS ))

export SLURM_CPUS_PER_TASK=$CPUS_PER_TASK
export OMP_NUM_THREADS=$CPUS_PER_TASK
export MKL_NUM_THREADS=$CPUS_PER_TASK

TMP_ADDR=/tmp/ray_head_addr_$SLURM_JOB_ID

# ────────────────────────────────────────────────────────────
# Head vs Worker logic
# ────────────────────────────────────────────────────────────
if [ "$SLURM_PROCID" -eq 0 ]; then
  # ------------------ HEAD ------------------
  echo "[HEAD] $(hostname): launching Ray head (CPUs=$CPUS_PER_TASK)"
  HEAD_IP=$(hostname -I | awk '{print $1}')

  ray start --head \
            --port=6379 \
            --num-cpus=$CPUS_PER_TASK \
            --node-ip-address="$HEAD_IP" \
            --include-dashboard=false \
            --block &

  echo "$HEAD_IP:6379" | tee $TMP_ADDR
  export RAY_HEAD_ADDRESS="$HEAD_IP:6379"

  echo "[HEAD] waiting $HEAD_START_WAIT s for workers"
  sleep $HEAD_START_WAIT

  echo "[HEAD] running preprocess.py"
  python src/preprocess.py \
    --input data/raw/250k_rndm_zinc_drugs_clean_3.csv \
    --smiles-col smiles \
    --output data/processed/molecules_processed.csv \
    --vocab data/processed/vocab \
    2>&1 | tee -a logs/preprocess_py_${SLURM_JOB_ID}.log

  echo "[HEAD] stopping Ray cluster"
  ray stop --force
  rm -f $TMP_ADDR

else
  # ------------------ WORKERS ------------------
  echo "[WORKER $SLURM_PROCID] polling for head address"
  for i in $(seq 1 $WORKER_RETRIES); do
    if [ -f $TMP_ADDR ]; then break; fi
    echo "[WORKER $SLURM_PROCID] try $i/$WORKER_RETRIES – sleeping $WORKER_SLEEP s"
    sleep $WORKER_SLEEP
  done

  if [ ! -f $TMP_ADDR ]; then
    echo "[WORKER $SLURM_PROCID] ERROR: head never registered"
    exit 1
  fi

  HEAD_ADDR=$(cat $TMP_ADDR)
  export RAY_HEAD_ADDRESS=$HEAD_ADDR
  echo "[WORKER $SLURM_PROCID] connecting to $HEAD_ADDR (CPUs=$CPUS_PER_TASK)"

  WORKER_IP=$(hostname -I | awk '{print $1}')
  ray start --address="$HEAD_ADDR" \
            --num-cpus=$CPUS_PER_TASK \
            --node-ip-address="$WORKER_IP" \
            --block
fi