# Project 2: Containerized MRPC Training

This project converts the hyperparameter tuning notebook from Project 1 into a containerized application for training a DistilBERT model on the MRPC (Microsoft Research Paraphrase Corpus) task.

## Project Structure

```
.
├── main.py              # Main training script
├── model.py             # PyTorch Lightning model module
├── data_module.py       # PyTorch Lightning data module
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration (to be added)
└── README.md           # This file
```

## Installation

### Option 1: Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   docker build -t mrpc-training .
   ```

2. **Run training:**
   ```bash
   docker run --rm mrpc-training
   ```

3. **Run with custom hyperparameters:**
   ```bash
   docker run --rm mrpc-training python main.py \
     --checkpoint_dir /app/models \
     --lr 4e-5 \
     --weight_decay 0.005 \
     --no_wandb
   ```

4. **Run with W&B logging:**
   ```bash
   docker run --rm -e WANDB_API_KEY=your_key_here mrpc-training \
     python main.py --checkpoint_dir /app/models --lr 5.5e-5
   ```

### Option 2: Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Login to Weights & Biases (optional, for experiment tracking):
```bash
wandb login
```

## Usage

### Basic Training Run

Run training with default hyperparameters:

```bash
python main.py --checkpoint_dir ./models
```

### Training with Best Hyperparameters (from Project 1)

Based on Project 1 results, the best configuration was:
- Learning rate: 5.5e-5
- Weight decay: 0.001
- Warmup steps: 0

Run with these settings:

```bash
python main.py \
  --checkpoint_dir ./models \
  --lr 5.5e-5 \
  --weight_decay 0.001 \
  --warmup_steps 0
```

### Command-Line Arguments

#### Required Arguments
- `--checkpoint_dir`: Directory to save model checkpoints

#### Hyperparameters
- `--lr`: Learning rate (default: 5e-5)
- `--weight_decay`: Weight decay for regularization (default: 0.001)
- `--warmup_steps`: Number of warmup steps for learning rate scheduler (default: 0)
- `--optimizer_eps`: Epsilon for AdamW optimizer (default: 1e-8)

#### Training Configuration
- `--epochs`: Number of training epochs (default: 3)
- `--train_batch_size`: Training batch size (default: 32)
- `--eval_batch_size`: Evaluation batch size (default: 32)
- `--accumulate_grad_batches`: Gradient accumulation steps (default: 1)
- `--gradient_clip_val`: Gradient clipping value (default: 0.0, no clipping)

#### Model and Data
- `--model_name`: Pretrained model (default: distilbert-base-uncased)
- `--task_name`: GLUE task (default: mrpc)
- `--max_seq_length`: Maximum sequence length (default: 128)

#### Experiment Tracking
- `--wandb_project`: Weights & Biases project name (default: MLOPS_Project2)
- `--experiment_name`: Custom experiment name (default: auto-generated)
- `--no_wandb`: Disable W&B logging

#### Other
- `--seed`: Random seed for reproducibility (default: 42)
- `--accelerator`: Device accelerator (default: auto)
- `--devices`: Number of devices to use (default: 1)

### Example Commands

**Training with custom hyperparameters:**
```bash
python main.py \
  --checkpoint_dir ./models \
  --lr 4e-5 \
  --weight_decay 0.005 \
  --warmup_steps 50 \
  --epochs 3
```

**Training without W&B logging:**
```bash
python main.py \
  --checkpoint_dir ./models \
  --lr 5.5e-5 \
  --no_wandb
```

**Training with gradient accumulation:**
```bash
python main.py \
  --checkpoint_dir ./models \
  --lr 5.5e-5 \
  --accumulate_grad_batches 4 \
  --train_batch_size 8
```

## Expected Output

The training script will:
1. Print the configuration summary
2. Download/prepare the MRPC dataset
3. Train for the specified number of epochs
4. Log metrics to Weights & Biases (if enabled)
5. Save checkpoints to the specified directory
6. Print final validation metrics (F1, Accuracy, Loss)

Example output:
```
================================================================================
TRAINING CONFIGURATION
================================================================================
Experiment name: lr_5.50e-05_wd_0.0010_warmup_0_20241111_120000
Model: distilbert-base-uncased
Task: mrpc
...
================================================================================
TRAINING COMPLETED
================================================================================

Final Results:
  F1 Score: 0.9030
  Accuracy: 0.8620
  Val Loss: 0.4500
```

## Project 1 Results Summary

Based on the hyperparameter tuning experiments:

**Week 1 Best Configuration:**
- Learning rate: 5e-5
- Weight decay: 0.001
- Warmup steps: 200
- F1 Score: 0.90052
- Accuracy: 0.86029

**Week 2 Best Configuration:**
- Learning rate: 5.5e-5
- Weight decay: 0.001
- Warmup steps: 0
- F1 Score: 0.903
- Accuracy: 0.862

**Week 3 (Automatic/Random Search):**
- Learning rate: ~4.8e-5
- Weight decay: 0.005
- Warmup steps: 50
- F1 Score: 0.895
- Accuracy: 0.854

The manual tuning in Weeks 1 and 2 achieved the best results, demonstrating that learning rate, weight decay, and warmup steps are the most influential hyperparameters for this task.

## Notes

- **Docker:** The image uses CPU-only PyTorch for compatibility. Training takes ~15-20 minutes on most machines.
- **GPU Training:** For GPU support locally, install the CUDA version of PyTorch instead.
- Training without GPU will be significantly slower than on Colab (~10-15 minutes per epoch on CPU vs. ~1-2 minutes on GPU)
- The default configuration uses the best hyperparameters found in Project 1
- All training runs are reproducible using the `--seed` parameter
- Checkpoints and logs are saved to the specified `--checkpoint_dir`

## Docker Image Details

- **Base image:** python:3.10-slim
- **Size:** ~2.3 GB (CPU-only PyTorch)
- **Build time:** ~5-10 minutes (depending on internet speed)
- **Runtime:** ~18-22 minutes for 3 epochs on typical CPU

---

## Project Results

### Task 1: Script Conversion
Successfully converted Jupyter notebooks into modular Python scripts with CLI support and W&B experiment tracking.

### Task 2: Local Docker Results

**Build Information:**
- Build command: `docker build -t mrpc-training .`
- Image size: 2.34 GB
- Platform: Local machine (CPU)

**Execution Results:**
- Run command: `docker run --rm mrpc-training`
- Training time: ~19 minutes (3 epochs)
- Final metrics:
  - **F1 Score: 0.899**
  - **Accuracy: 0.858**
  - **Validation Loss: 0.453**

The container executed successfully without modifications to the host environment, demonstrating proper dependency isolation.

### Task 3: Cloud Deployment Results

**Platform:** GitHub Codespaces (4 cores, 8GB RAM)

**Repository:** https://github.com/arnehbegi/MLOPS_Project2_ArnehB

**Deployment Process:**
1. Created public GitHub repository
2. Opened in Codespaces via browser
3. Built and ran Docker container with identical commands

**Adaptations Required:** None ✅  
The Dockerfile worked without any code changes, validating reproducibility.

**Execution Results:**
- Training time: ~22 minutes (3 epochs)
- Final metrics:
  - **F1 Score: 0.893**
  - **Accuracy: 0.854**
  - **Validation Loss: 0.458**

### Performance Comparison

| Environment | F1 Score | Accuracy | Val Loss | Training Time |
|-------------|----------|----------|----------|---------------|
| Project 1 (GPU, Colab) | 0.903 | 0.862 | 0.450 | ~2 min |
| Task 2 (Local CPU) | 0.899 | 0.858 | 0.453 | ~19 min |
| Task 3 (Codespaces CPU) | 0.893 | 0.854 | 0.458 | ~22 min |

**Analysis:**  
Results are consistent across environments with expected variance (F1 difference of 0.006 between local and cloud). Minor differences are due to CPU architecture variations and floating-point precision, which validates Docker's reproducibility while demonstrating natural computational variance across hardware platforms.

## Running on Cloud Platforms

### GitHub Codespaces

1. Fork/clone this repository on GitHub
2. Open in Codespaces (Code → Codespaces → Create codespace)
3. In the terminal:
   ```bash
   docker build -t mrpc-training .
   docker run --rm mrpc-training
   ```

### Docker Playground

⚠️ **Note:** Docker Playground has limited memory (4GB). Build may fail or be very slow.

1. Go to https://labs.play-with-docker.com/
2. Add new instance
3. Clone this repo: `git clone <your-repo-url>`
4. Build and run:
   ```bash
   docker build --memory=3g -t mrpc-training .
   docker run --rm --memory=3g mrpc-training
   ```