## `README.md`

# MolGPT Preprocessing & Generation Pipeline

A modular, multi-phase system for SMILES generation and evaluation using GPT-style transformers and Ray on HPC.

**Current version:** vv1.2.0

---

## Table of Contents

- [Vision](#vision)  
- [Project Structure](#project-structure)  
- [Environment Setup](#environment-setup)  
- [Usage](#usage)  
- [Pipeline Roadmap](#pipeline-roadmap)  
- [Logging & Outputs](#logging--outputs)  
- [Versioning & Changelog](#versioning--changelog)  

---

## Vision

End-to-end workflow:

1. Preprocess SMILES data at scale.  
2. Train CPU-optimized GPT model.  
3. Generate candidate molecules via Ray.  
4. Evaluate with RDKit metrics and Autodock Vina.  
5. Apply Pareto multi-objective optimization.

---

## Project Structure

project_root/
├── VERSION                      ← Project version (e.g., “1.2.0”)
├── CHANGELOG.md                 ← Detailed version history
├── environment.yml              ← Conda environment spec
├── README.md                    ← This file
├── config/
│   ├── defaults.yaml
│   └── experiments.yaml
├── src/
│   ├── preprocess.py            ← Phase 1: SMILES preprocessing
│   ├── model.py                 ← Phase 2: GPT model
│   ├── train.py                 ← Phase 3: training loop
│   ├── generate.py              ← Phase 4: distributed generation
│   ├── docking.py               ← Phase 5: Vina docking integration
│   ├── multicriteria.py         ← Pareto evaluation
│   └── utils.py                 ← Helpers & benchmarks
├── experiments/
│   ├── run_preprocess_multi.sh  ← SLURM wrapper Phase 1
│   ├── run_train.sh             ← SLURM wrapper Phase 3
│   ├── run_generate.sh          ← SLURM wrapper Phase 4
│   └── run_docking.sh           ← SLURM wrapper Phase 5
├── data/
│   ├── raw/                     ← Raw SMILES & proteins
│   ├── processed/               ← Cleaned SMILES & vocab
│   └── protein/                 ← PDBQT files for docking
├── notebooks/                   ← Exploratory notebooks
├── outputs/                     ← Checkpoints & generated CSVs
└── logs/                        ← SLURM & driver logs

---

## Environment Setup

```bash
# 1) Create Conda env
conda env create -f environment.yml

# 2) Activate it
conda activate molgpt_py3922
```

Important deps in `environment.yml`:

```yaml
name: molgpt_py3922
channels:
  - conda-forge
dependencies:
  - python=3.9
  - ray
  - pytorch
  - rdkit
  - selfies
  - pandas
  - numpy
  - vina
  - pip
```

Add new deps via:

```bash
conda env update -f environment.yml
```

---

## Usage

From project root, submit SLURM jobs:

```bash
sbatch experiments/run_preprocess_multi.sh   # Phase 1
sbatch experiments/run_train.sh              # Phase 3
sbatch experiments/run_generate.sh           # Phase 4
sbatch experiments/run_docking.sh            # Phase 5
```

Override defaults with CLI flags: `--input`, `--output`, `--vocab`, etc.

---

## Pipeline Roadmap

0. Repo bootstrap & versioning (v1.2.0)  
1. Preprocessing & vocab (v1.2.0)  
2. CPU GPT model (pending v1.3.0)  
3. CPU training with Ray (pending v1.4.0)  
4. Ray generation (pending v1.5.0)  
5. Vina docking (pending v1.6.0)  
6. Pareto optimization & packaging (pending v1.7.0)  

---

## Logging & Outputs

- **SLURM logs:** `logs/preprocess_<JOBID>.log`, `logs/train_<JOBID>.log`, …  
- **Driver logs:** `logs/preprocess_py_<JOBID>.log`, `logs/train_py_<JOBID>.log`, …  
- **Processed data:** `data/processed/*.csv` & `vocab_*.json`  
- **Model ckpts:** `outputs/checkpoints/`  
- **Generated molecules:** `outputs/generated_<JOBID>.csv`  

---

## Versioning & Changelog

We follow [Keep a Changelog](https://keepachangelog.com/).  
- Single source of truth: the `VERSION` file.  
- Full history in [`CHANGELOG.md`](CHANGELOG.md).  