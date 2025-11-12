"""
Main training script for MRPC fine-tuning with hyperparameter support.
Usage: python main.py --checkpoint_dir models --lr 5e-5 --weight_decay 0.001 --warmup_steps 0
"""

import argparse
import os
from datetime import datetime

import lightning as L
import wandb
from lightning.pytorch.loggers import WandbLogger

from data_module import GLUEDataModule
from model import GLUETransformer
import os
os.environ["HF_HOME"] = "/app/hf_cache"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MRPC classification model")

    # Required arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints"
    )

    # Hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay (default: 0.001)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps (default: 0)"
    )
    parser.add_argument(
        "--optimizer_eps",
        type=float,
        default=1e-8,
        help="Optimizer epsilon (default: 1e-8)"
    )

    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Evaluation batch size (default: 32)"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Gradient accumulation batches (default: 1)"
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=0.0,
        help="Gradient clipping value (default: 0.0, no clipping)"
    )

    # Model and data
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained model name (default: distilbert-base-uncased)"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="mrpc",
        help="GLUE task name (default: mrpc)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)"
    )

    # Experiment tracking
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="MLOPS_Project2",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator type (default: auto)"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices (default: 1)"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set seed for reproducibility
    L.seed_everything(args.seed)

    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = (
            f"lr_{args.lr:.2e}_wd_{args.weight_decay:.4f}_"
            f"warmup_{args.warmup_steps}_{timestamp}"
        )

    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Experiment name: {args.experiment_name}")
    print(f"Model: {args.model_name}")
    print(f"Task: {args.task_name}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print("\nHyperparameters:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Optimizer epsilon: {args.optimizer_eps}")
    print("\nTraining settings:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Train batch size: {args.train_batch_size}")
    print(f"  Eval batch size: {args.eval_batch_size}")
    print(f"  Accumulate grad batches: {args.accumulate_grad_batches}")
    print(f"  Gradient clip value: {args.gradient_clip_val}")
    print(f"  Seed: {args.seed}")
    print("=" * 80)

    # Initialize logger
    if not args.no_wandb:
        try:
            wandb.login(relogin=True)
            logger = WandbLogger(
                project=args.wandb_project,
                name=args.experiment_name,
                config=vars(args)
            )
            print(f"\n✓ Weights & Biases logging enabled (project: {args.wandb_project})")
        except Exception as e:
            print(f"\n⚠ Warning: Could not initialize W&B logger: {e}")
            print("Continuing without W&B logging...")
            logger = None
    else:
        logger = None
        print("\n✓ W&B logging disabled")

    # Initialize data module
    print("\nInitializing data module...")
    dm = GLUEDataModule(
        model_name_or_path=args.model_name,
        task_name=args.task_name,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    dm.setup("fit")
    print(f"✓ Data module ready (num_labels: {dm.num_labels})")

    # Initialize model
    print("\nInitializing model...")
    model = GLUETransformer(
        model_name_or_path=args.model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=args.task_name,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        optimizer_eps=args.optimizer_eps,
    )
    print("✓ Model initialized")

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None,
        default_root_dir=args.checkpoint_dir,
        enable_progress_bar=True,
    )
    print("✓ Trainer ready")

    # Train model
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")

    trainer.fit(model, datamodule=dm)

    # Get final metrics
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    metrics = trainer.callback_metrics
    f1_score = metrics.get("f1", 0.0)
    accuracy = metrics.get("accuracy", 0.0)
    val_loss = metrics.get("val_loss", 0.0)

    # Convert tensors to floats if necessary
    if hasattr(f1_score, "item"):
        f1_score = f1_score.item()
    if hasattr(accuracy, "item"):
        accuracy = accuracy.item()
    if hasattr(val_loss, "item"):
        val_loss = val_loss.item()

    print(f"\nFinal Results:")
    print(f"  F1 Score: {f1_score:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"\nCheckpoint saved to: {args.checkpoint_dir}")

    # Finish W&B run
    if logger is not None:
        wandb.finish()
        print(f"\n✓ Results logged to Weights & Biases")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()