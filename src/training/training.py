"""
LSTM Training Script
====================

Trains the stock price prediction LSTM model.

Prerequisites:
- Run feature_engineering.py first (creates data/features/*.parquet)
- Run scaling_sequences.py first (creates data/sequences/*.npz)

Usage:
    python src/training.py                        # Fresh training
    python src/training.py --resume               # Resume from latest checkpoint
    python src/training.py --resume --checkpoint path/to/model.pt
"""

import torch
import numpy as np
import sys
import argparse
from pathlib import Path
from lstm_model import StockLSTM, LSTMTrainer, LSTMConfig, TrainingConfig


# ============================================================
# CONFIGURATION - Edit these settings
# ============================================================

# Data paths
TRAIN_PATH = 'data/sequences/train.npz'
VAL_PATH   = 'data/sequences/val.npz'
TEST_PATH  = 'data/sequences/test.npz'

# Checkpoint paths
OUTPUT_PATH     = 'models/best_model.pt'
CHECKPOINT_DIR  = 'models/checkpoints'
LATEST_CKPT     = f'{CHECKPOINT_DIR}/latest.pt'   # Saved every epoch for recovery

# Training parameters
EPOCHS        = 100
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model architecture
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
DROPOUT         = 0.2
USE_ATTENTION   = True
ATTENTION_HEADS = 4

# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description='Train LSTM stock prediction model')
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help=f'Path to checkpoint to resume from (default: {LATEST_CKPT})'
    )
    return parser.parse_args()


def find_resume_checkpoint(explicit_path: str | None) -> Path | None:
    """
    Determine which checkpoint to resume from.
    Priority: explicit --checkpoint arg > latest.pt > best_model.pt
    Returns None if no valid checkpoint is found.
    """
    candidates = [
        explicit_path,
        LATEST_CKPT,
        OUTPUT_PATH,
    ]
    for path in candidates:
        if path and Path(path).exists():
            return Path(path)
    return None


def build_fresh_trainer(n_features: int) -> LSTMTrainer:
    """Create a brand-new model and trainer from config constants"""
    model_config = LSTMConfig(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_attention=USE_ATTENTION,
        attention_heads=ATTENTION_HEADS
    )
    train_config = TrainingConfig(
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        optimizer='adamw',
        loss_fn='huber',
        use_scheduler=True,
        early_stopping=True,
        patience=15,
        gradient_clip_value=1.0,
        checkpoint_dir=CHECKPOINT_DIR,
        device=DEVICE
    )
    model = StockLSTM(model_config)
    return LSTMTrainer(model, train_config)


def build_resumed_trainer(checkpoint_path: Path, n_features: int) -> LSTMTrainer:
    """
    Reconstruct model and trainer from a checkpoint, then restore full state.
    The checkpoint is self-describing — model_config and train_config are embedded,
    so we don't need to rely on the constants above matching what was originally trained.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Reconstruct model from the config that was saved at training time
    model_config = LSTMConfig(**checkpoint['model_config'])
    train_config = TrainingConfig(**checkpoint['train_config'])
    model   = StockLSTM(model_config)
    trainer = LSTMTrainer(model, train_config)

    # Restore weights, optimizer moments, and training history
    trainer.load_checkpoint(str(checkpoint_path))

    print(f"  Resumed from epoch {trainer.current_epoch}")
    print(f"  Best val loss so far: {trainer.best_val_loss:.6f}")
    print(f"  Epochs without improvement: {trainer.epochs_without_improvement}")

    return trainer


def save_latest(trainer: LSTMTrainer):
    """
    Overwrite latest.pt after every epoch.
    This is the recovery checkpoint — always reflects the most recent epoch
    regardless of whether val loss improved. Separate from best_model.pt.
    """
    trainer.save_checkpoint('latest.pt')


def main():
    args = parse_args()

    print("=" * 60)
    print("LSTM TRAINING")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────
    print("\nChecking data files...")
    for path in [TRAIN_PATH, VAL_PATH]:
        if not Path(path).exists():
            print(f"ERROR: {path} not found. Run scaling_sequences.py first.")
            sys.exit(1)
        print(f"✓ {path}")

    print("\nLoading data...")
    train_data = np.load(TRAIN_PATH)
    val_data   = np.load(VAL_PATH)

    train_dict = {'X': train_data['X'], 'y': train_data['y']}
    val_dict   = {'X': val_data['X'],   'y': val_data['y']}

    n_features = train_data['X'].shape[2]
    print(f"Train: {train_data['X'].shape}  Val: {val_data['X'].shape}  Features: {n_features}")

    # ── Trainer: fresh or resumed ────────────────────────────
    if args.resume:
        ckpt_path = find_resume_checkpoint(args.checkpoint)
        if ckpt_path is None:
            print("\nWARNING: --resume specified but no checkpoint found. Starting fresh.")
            trainer = build_fresh_trainer(n_features)
        else:
            trainer = build_resumed_trainer(ckpt_path, n_features)
    else:
        print("\nStarting fresh training...")
        trainer = build_fresh_trainer(n_features)

    print(f"\nDevice: {DEVICE}  |  Parameters: {trainer.model.get_num_parameters():,}")
    print(f"Batch size: {BATCH_SIZE}  |  Max epochs: {EPOCHS}  |  LR: {LEARNING_RATE}")

    # ── Training loop (with per-epoch latest.pt save) ─────────
    print("\nStarting training...\n")

    from torch.utils.data import DataLoader
    from lstm_model import StockDataset

    train_loader = DataLoader(
        StockDataset(train_dict['X'], train_dict['y']),
        batch_size=trainer.config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE == 'cuda')
    )
    val_loader = DataLoader(
        StockDataset(val_dict['X'], val_dict['y']),
        batch_size=trainer.config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE == 'cuda')
    )

    # Resume from where we left off rather than epoch 0
    start_epoch = trainer.current_epoch if args.resume else 0

    for epoch in range(start_epoch, trainer.config.num_epochs):
        trainer.current_epoch = epoch

        train_loss = trainer.train_epoch(train_loader)
        val_loss   = trainer.validate(val_loader)

        trainer.train_losses.append(train_loss)
        trainer.val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | train={train_loss:.6f} val={val_loss:.6f}")

        # LR scheduler step
        if trainer.scheduler:
            import torch.optim as optim
            if isinstance(trainer.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                trainer.scheduler.step(val_loss)
            else:
                trainer.scheduler.step()

        # Save latest.pt every epoch — this is the crash-recovery checkpoint
        save_latest(trainer)

        # Save best_model.pt on genuine improvement
        if val_loss < (trainer.best_val_loss - trainer.config.min_delta):
            trainer.best_val_loss = val_loss
            trainer.epochs_without_improvement = 0
            trainer.save_checkpoint('best_model.pt')
            print(f"  ✓ New best model saved (val={val_loss:.6f})")
        else:
            trainer.epochs_without_improvement += 1

        # Early stopping
        if (trainer.config.early_stopping and
                trainer.epochs_without_improvement >= trainer.config.patience):
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {trainer.config.patience} epochs)")
            break

    # ── Final save ────────────────────────────────────────────
    print(f"\nSaving final model to {OUTPUT_PATH}...")
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Model saved to:       {OUTPUT_PATH}")

    # ── Optional test evaluation ──────────────────────────────
    if Path(TEST_PATH).exists():
        print("\nEvaluating on test set...")
        test_data = np.load(TEST_PATH)
        test_loader = DataLoader(
            StockDataset(test_data['X'], test_data['y']),
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        test_loss = trainer.validate(test_loader)
        print(f"Test loss: {test_loss:.6f}")

    print("\nNext steps:")
    print("  1. python src/kafka_inference_service.py")
    print("  2. python src/websocket_prediction_server.py")
    print("  3. Open realtime_dashboard.html")


if __name__ == "__main__":
    main()