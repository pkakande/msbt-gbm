# Changelog

_All notable changes to this project are documented here.  
This file follows [Keep a Changelog](https://keepachangelog.com/) conventions._

---

## [v1.2.0] – 2025-06-19

### Added
- Comprehensive docstrings and inline comments in `src/preprocess.py` and `experiments/run_preprocess_multi.sh`.  
- Structured Python `logging` (with timestamps and log levels) replacing ad-hoc `print()` statements.  
- Cluster-topology logging via `ray.nodes()`, showing per-node alive/dead status and CPU counts.  
- Resource summaries using `ray.cluster_resources()` and `ray.available_resources()`.  
- Parameterizable head/worker startup delays:
  - `HEAD_START_WAIT`  
  - `WORKER_RETRIES`  
  - `WORKER_SLEEP`  
- “Tee’d” driver log per SLURM job: `logs/preprocess_py_<SLURM_JOB_ID>.log`.  

### Changed
- Removed legacy `--tokenizer` flag; script now always runs all three tokenizers (char, regex, SELFIES).  
- Back-ported Python 3.10 union types to 3.7+-compatible `typing.Optional`, `Tuple`, `List`.  
- Switched to standalone Ray CLI (`ray start`/`ray stop`) bound to each node’s IP via `--node-ip-address`.  
- Disabled the Ray dashboard with `--include-dashboard=false` (compatible with Ray 2.x).  

### Fixed
- Dropped passing of `_system_config` when connecting to an existing cluster (avoids `ValueError`).  
- Ensured `selfies`, `ray`, and `rdkit` are installed and importable in the Conda environment.  
- Enforced `cd "$SLURM_SUBMIT_DIR"` in SLURM wrapper so all relative paths resolve correctly.  

---

## [v1.1.0] – 2025-06-18

### Added
- Multi-node SLURM wrapper `experiments/run_preprocess_multi.sh` with clear head vs. worker logic.  
- Dynamic CPU allocation per Ray process:
  - Reserve 8 cores per node for OS/Ray overhead.  
  - Evenly divide remaining cores across 4 tasks.  
- Support for CSV input in `preprocess.py` with `--input` and `--smiles-col`.  
- Automatic fallback to TSV or one-SMILES-per-line input format.  
- Initial use of `python -m ray` fallback when the standalone `ray` CLI wasn’t on `$PATH`.  

### Changed
- Refactored type hints from modern Python union syntax (`str | None`) to `List[str]` and `Tuple[Optional[str], …]` for 3.7+ compatibility.  
- Extracted canonicalization and tokenization logic into Ray remote tasks.  

### Fixed
- Resolved `ModuleNotFoundError: No module named ray` by enforcing a Conda-based installation and correct CLI usage.  
- Added instructions to install missing `selfies` dependency.  

---

## [v1.0.0] – 2025-06-17

### Added
- Initial project skeleton and folder layout:
  ```
  project_root/
  ├─ VERSION
  ├─ environment.yml
  ├─ src/preprocess.py
  ├─ experiments/run_preprocess.sh
  └─ data/raw/…
  ```
- `environment.yml` to pin Python (3.9) and key dependencies (RDKit, Ray, SELFIES, pandas, etc.).  
- Basic `preprocess.py`:
  1. Read raw SMILES (TSV or one-per-line).  
  2. Canonicalize via RDKit (single node).  
  3. Tokenize with character- and regex-based methods.  
  4. Export processed CSV and two vocab JSONs.  
- Single-node SLURM submission script: `experiments/run_preprocess.sh`.  

### Changed
- Hard-coded file paths and no resource splitting (single-task).  

### Fixed
- None (initial release).  

---