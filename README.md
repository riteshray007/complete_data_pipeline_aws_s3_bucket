# End-to-End Reproducible MLOps Pipeline with DVC & AWS S3

## Project Overview
This repository implements an end-to-end Machine Learning pipeline for **Email Spam Detection**, built with MLOps best practices. The project transitions from exploratory data analysis to a fully orchestrated, reproducible data engineering and model training pipeline.

By leveraging **Data Version Control (DVC)** backed by **Publicly Accessible AWS S3 Remote Storage**, code remains strictly separated from large binary data artifacts. Collaborators can clone this repository and download all tracked data and model checkpoints **without needing AWS credentials or access keys**.

---

## Architecture & Project Structure

```text
├── .dvc/                   # DVC remote configuration (Public AWS S3 storage settings)
├── dvclive/                # DVCLive tracked evaluation metrics, parameters, & TSV plots
│   ├── metrics.json        # Final evaluation metrics summary
│   ├── params.yaml         # Training parameters
│   └── plots/metrics/      # Accuracy, Precision, and Recall visualization logs
├── experiments/            # Exploratory Data Analysis (EDA) notebooks & sample data
│   ├── mynotebook.ipynb
│   └── spam.csv
├── src/                    # Modular pipeline source code
│   ├── data_ingestion.py   # Raw data fetching and train-test splitting
│   ├── data_preprocessing.py # Text cleaning, tokenization, and normalization
│   ├── feature_engineering.py # Vectorization (TF-IDF / feature extraction)
│   ├── model_training.py   # Classifier training and model artifact serialization
│   └── model_evaluation.py # Model evaluation and DVCLive telemetry logging
├── dvc.yaml                # DVC Pipeline Directed Acyclic Graph (DAG) definition
├── dvc.lock                # Immutable hash tracking file for reproducible pipeline runs
├── params.yaml             # Centralized configuration parameters for all pipeline stages
└── requirements.txt        # Python dependencies
```

---

## Key MLOps Features

* **Decoupled Code & Large Artifacts:** Git tracks lightweight scripts, `params.yaml`, `dvc.yaml`, `dvc.lock`, and evaluation metrics. Raw datasets, feature matrices, and trained `.pkl` model checkpoints are stored off-repo in AWS S3.
* **Public Anonymous Access:** The remote S3 bucket (`s3://email-spam-detection-dvc-s3`) is configured with public read permissions. Anyone can run `dvc pull` directly without AWS CLI credentials.
* **Pipeline Determinism:** Executing `dvc repro` or `dvc exp run` parses `params.yaml` and executes only modified pipeline stages, eliminating redundant computation.
* **Remote Experiment Tracking:** Candidate runs are tracked as Git experiment references (`refs/exps/`) and synced with S3 data caches, allowing team members to review, list, and apply alternative model iterations.

---

## Quick Start: Setup & Execution

### 1. Clone the Repository & Setup Environment
```bash
# Clone the repository
git clone https://github.com/riteshray007/complete_data_pipeline_aws_s3_bucket.git
cd complete_data_pipeline_aws_s3_bucket

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data & Model Artifacts (No AWS Credentials Needed)
Because the AWS S3 bucket policy allows public read access, download all tracked pipeline artifacts directly:
```bash
dvc pull
```

### 3. Reproduce the Full Pipeline
Run the end-to-end execution DAG from ingestion to evaluation:
```bash
dvc repro
```

---

## Managing Remote Experiments

This project tracks model iterations as DVC experiments synced across Git (`refs/exps/`) and AWS S3.

### 1. Download All Remote Experiments
To pull all experiment references and their associated dataset/model binaries from GitHub and S3:
```bash
# Fetch experiment references from Git
git fetch origin 'refs/exps/*:refs/exps/*'

# Pull associated experiment artifacts from S3
dvc exp pull origin
```

### 2. List Local & Remote Experiments
View all tracked experiment runs, parameter variations, and metrics:
```bash
# List all experiment names synced on origin
dvc exp list origin --all-commits

# Show detailed metrics comparison table in terminal
dvc exp show -A
```

### 3. Apply an Experiment to Your Workspace
To restore your local workspace (code, parameters, and model checkpoints) to any specific experiment run (e.g., `rocky-inks` or `washy-sins`):
```bash
dvc exp apply <experiment-name>
```

*Example:*
```bash
dvc exp apply rocky-inks
```

### 4. Promote Winning Experiment to `main`
Once an applied experiment is validated, commit and push it to the main repository branch:
```bash
git add .
git commit -m "Promote experiment <experiment-name> to main"
git push origin main
dvc push
```

---

## Metric & Parameter Diffing

To compare parameters and performance metrics between your current workspace and the `main` branch or specific commits:
```bash
# Compare evaluation metrics
dvc metrics diff main

# Compare pipeline parameters
dvc params diff main
```