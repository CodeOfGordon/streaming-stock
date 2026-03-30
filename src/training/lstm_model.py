"""
Production-Grade LSTM for Stock Price Prediction
================================================

Architecture Design:
- Multi-layer LSTM with dropout for regularization
- Optional attention mechanism for interpretability
- Batch normalization for training stability
- Multiple loss functions (MSE, MAE, Huber)

Training Features:
- Early stopping with patience
- Learning rate scheduling
- Gradient clipping (critical for RNNs)
- Mixed precision training (FP16)
- Model checkpointing

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dataclasses
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass
class LSTMConfig:
    """Configuration for LSTM model architecture"""
    # Architecture
    input_size: int                 # Number of input features
    hidden_size: int = 128          # LSTM hidden state dimension
    num_layers: int = 2             # Number of stacked LSTM layers
    dropout: float = 0.2            # Dropout between LSTM layers
    bidirectional: bool = False     # Bidirectional LSTM (doubles hidden size)

    # Attention
    use_attention: bool = True      # Attention over sequence
    attention_heads: int = 4        # Multi-head attention

    # Output
    output_size: int = 1            # Predict single value (return)

    # Normalization (mutually exclusive — pick one)
    batch_norm: bool = False         # Batch norm
    layer_norm: bool = True        # Layer norm (better for RNN/LSTM)


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    # Optimization
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 100
    optimizer: str = 'adamw'        # 'adam', 'adamw', 'sgd'
    weight_decay: float = 1e-5      # L2 regularization

    # Loss function
    loss_fn: str = 'huber'          # 'mse', 'mae', 'huber'
    huber_delta: float = 1.0        # Huber loss: robust to outliers (important for financial data)

    # Learning rate schedule
    use_scheduler: bool = True
    scheduler_type: str = 'reduce_on_plateau'   # 'reduce_on_plateau', 'cosine'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Gradient clipping — critical for LSTMs to prevent exploding gradients
    gradient_clip_value: float = 1.0

    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-6         # Minimum improvement to count as progress

    # Mixed precision training (faster on modern GPUs, requires CUDA)
    use_amp: bool = False

    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = 'models/checkpoints'

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttentionLayer(nn.Module):
    """
    Multi-head self-attention over LSTM output sequence.

    Lets the model focus on important timesteps (e.g. earnings, regime shifts)
    rather than relying solely on the final hidden state. Attention weights
    can also be visualized for interpretability.
    """

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context:      (batch, hidden_size) — attended summary of sequence
            attn_weights: (batch, seq_len)     — per-timestep importance scores
        """
        # Self-attention: query, key, value are all the LSTM output
        attn_output, attn_weights = self.attention(lstm_output, lstm_output, lstm_output)

        context = attn_output.mean(dim=1)        # Pool across sequence → single vector
        attn_weights = attn_weights.mean(dim=1)  # Average across heads for visualization

        return context, attn_weights


class StockLSTM(nn.Module):
    """
    Production LSTM for stock return prediction.

    Architecture:
        1. LSTM layers (with dropout between layers)
        2. Optional attention over LSTM outputs
        3. Layer normalization
        4. Fully connected output layer
        Input → Stacked LSTM Layers → Attention (optional) → LayerNorm → FC → Output

    Design decisions:
        - Stacked LSTM:  captures patterns at multiple levels of abstraction
        - Dropout:       prevents overfitting (0.2 is standard for financial data)
        - Attention:     focuses on regime changes, earnings dates, volatility spikes
        - Layer norm:    stable normalization at any batch size, including batch_size=1 at inference
        - Xavier init:   better gradient flow at the start of training
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,  # dropout requires >1 layer
            batch_first=True,
            bidirectional=config.bidirectional
        )

        # Bidirectional doubles the output size
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)

        # Attention or fallback to last hidden state
        self.attention = AttentionLayer(lstm_output_size, config.attention_heads) if config.use_attention else None

        # Normalization layer (batch norm struggles with batch_size=1 at inference — see eval() note)
        if config.batch_norm:
            self.norm = nn.BatchNorm1d(lstm_output_size)
        elif config.layer_norm:
            self.norm = nn.LayerNorm(lstm_output_size)
        else:
            self.norm = None

        # Two-layer FC head: compress → predict
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(lstm_output_size // 2, config.output_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for stable gradient flow at training start.
        Guards against 1D tensors — xavier_uniform_ requires >= 2 dimensions.
        LSTM bias vectors (bias_ih_l0, bias_hh_l0 etc.) are 1D, so they fall
        through to the zero-init branch rather than crashing.
        BatchNorm weight (gamma) is also 1D — it must be left at the PyTorch
        default of 1.0. Zeroing gamma collapses BatchNorm output to 0, which
        kills gradient flow through relu and prevents the entire model from
        learning (only fc2 bias receives gradients, converging to mean return ≈ 0).
        """
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Zero-init biases only — never touch BatchNorm/LayerNorm gamma here
                nn.init.constant_(param, 0)
            # 1D weight tensors (BatchNorm gamma, LayerNorm gamma) are intentionally
            # skipped — PyTorch already initialises them to 1.0, which is correct.

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            output:           (batch, output_size) — predicted return
            attention_weights: (batch, seq_len) or None
        """
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size * num_directions)

        # Summarize sequence via attention, or just take the final timestep
        if self.attention is not None:
            features, attention_weights = self.attention(lstm_out)
        else:
            features = lstm_out[:, -1, :]  # Last timestep only
            attention_weights = None

        if self.norm is not None:
            features = self.norm(features)

        out = self.fc1(features)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out, attention_weights

    def get_num_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StockDataset(Dataset):
    """Minimal PyTorch Dataset wrapper for pre-built sequence arrays"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: (n_samples, seq_len, n_features)
            y: (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMTrainer:
    """
    Training orchestrator with production best practices:
    early stopping, LR scheduling, gradient clipping,
    checkpointing, and optional mixed precision (AMP).
    """

    def __init__(self, model: StockLSTM, train_config: TrainingConfig):
        self.model = model
        self.config = train_config
        self.device = torch.device(train_config.device)
        self.model.to(self.device)

        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()
        self.scheduler = self._create_scheduler() if train_config.use_scheduler else None
        self.scaler = torch.cuda.amp.GradScaler() if train_config.use_amp else None

        # Training state (persisted into checkpoints)
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []

    def _create_optimizer(self) -> optim.Optimizer:
        # AdamW preferred: decoupled weight decay is more correct than Adam's
        opts = {
            'adam':  optim.Adam,
            'adamw': optim.AdamW,
        }
        if self.config.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.config.learning_rate,
                             momentum=0.9, weight_decay=self.config.weight_decay)
        cls = opts.get(self.config.optimizer)
        if not cls:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        return cls(self.model.parameters(), lr=self.config.learning_rate,
                   weight_decay=self.config.weight_decay)

    def _create_criterion(self) -> nn.Module:
        # Huber recommended for financial data: less sensitive to return outliers than MSE
        fns = {
            'mse':   nn.MSELoss(),
            'mae':   nn.L1Loss(),
            'huber': nn.HuberLoss(delta=self.config.huber_delta),
        }
        if self.config.loss_fn not in fns:
            raise ValueError(f"Unknown loss function: {self.config.loss_fn}")
        return fns[self.config.loss_fn]

    def _create_scheduler(self):
        if self.config.scheduler_type == 'reduce_on_plateau':
            # Halves LR when val loss stops improving — good default for LSTMs
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience
            )
        elif self.config.scheduler_type == 'cosine':
            # Smooth decay to eta_min; better for longer training runs
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs, eta_min=1e-6
            )
        raise ValueError(f"Unknown scheduler: {self.config.scheduler_type}")

    def _clip_and_step(self, loss):
        """
        Backward pass with gradient clipping.
        Clipping is critical for RNNs — gradients can explode through long sequences
        without it. Handles both AMP (FP16) and standard (FP32) paths.
        """
        if self.scaler:
            # AMP path: scale loss to prevent FP16 underflow, then unscale before clipping
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
            self.optimizer.step()

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()  # Enables dropout + norm training behaviour
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()

            if self.scaler:
                with torch.cuda.amp.autocast():  # FP16 forward pass
                    predictions, _ = self.model(X_batch)
                    loss = self.criterion(predictions.squeeze(), y_batch)
            else:
                predictions, _ = self.model(X_batch)
                loss = self.criterion(predictions.squeeze(), y_batch)

            self._clip_and_step(loss)
            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()       # Disables dropout; layer norm has no running stats to freeze
        val_loss = 0.0

        with torch.no_grad():   # No gradients needed — saves memory and speeds up inference
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions, _ = self.model(X_batch)
                val_loss += self.criterion(predictions.squeeze(), y_batch).item()

        return val_loss / len(val_loader)

    def fit(self, train_data: Dict[str, np.ndarray], val_data: Dict[str, np.ndarray]):
        train_loader = DataLoader(
            StockDataset(train_data['X'], train_data['y']),
            batch_size=self.config.batch_size,
            shuffle=True,           # Shuffle for better gradient estimates
            num_workers=4,
            pin_memory=(self.device.type == 'cuda')  # Faster CPU→GPU transfer
        )
        val_loader = DataLoader(
            StockDataset(val_data['X'], val_data['y']),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(self.device.type == 'cuda')
        )

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch:03d} | train={train_loss:.6f} val={val_loss:.6f}")

            # ReduceLROnPlateau needs the metric; other schedulers step blindly
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Checkpoint on genuine improvement (must beat best by at least min_delta)
            if val_loss < (self.best_val_loss - self.config.min_delta):
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                if self.config.save_best_only:
                    self.save_checkpoint('best_model.pt')
            else:
                self.epochs_without_improvement += 1

            if self.config.early_stopping and self.epochs_without_improvement >= self.config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    def save_checkpoint(self, filename: str):
        """Save full training state — enough to resume training or run inference"""
        path = Path(self.config.checkpoint_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': dataclasses.asdict(self.model.config),
            'train_config': dataclasses.asdict(self.config),
            'saved_at': datetime.now().isoformat()
        }, path)

    def load_checkpoint(self, filepath: str):
        """Restore model + optimizer state to resume training from a checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']