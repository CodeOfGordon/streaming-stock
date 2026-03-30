"""
Scaling and Sequence Generation for LSTM
=========================================

Simple script to:
1. Scale features with RobustScaler
2. Create sliding window sequences
3. Split into train/val/test

"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler
from pathlib import Path


def prepare_lstm_data(
    df: pd.DataFrame,
    lookback: int = 60,
    train_split: float = 0.7,
    val_split: float = 0.15
):
    """
    Prepare data for LSTM training.
    
    Args:
        df: DataFrame with features and 'close' column
        lookback: Number of timesteps to look back (60 hours)
        train_split: Fraction for training (0.7 = 70%)
        val_split: Fraction for validation (0.15 = 15%)
    
    Returns:
        train_data: {'X': (N, 60, features), 'y': (N,)}
        val_data: {'X': ..., 'y': ...}
        test_data: {'X': ..., 'y': ...}
        scalers: {'feature_scaler': scaler, 'target_scaler': scaler}
    """
    
    # 1. Create target (next-minute return)
    df['target'] = df['close'].pct_change().shift(-1)
    df = df.dropna()
    
    # 2. Select features (everything except raw OHLCV and target, i.e. the features from feature_eng file)
    exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [col for col in df.columns if col not in exclude]
    
    X = df[feature_cols].values
    y = df['target'].values
    
    print(f"Data shape: {X.shape[0]} samples, {X.shape[1]} features")
    
    # 3. Split temporally (maintain time order)
    n = len(X)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 4. Scale features and target
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()
    
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)
    
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # 5. Create sequences (sliding windows)
    X_train_seq, y_train_seq = make_sequences(X_train_scaled, y_train_scaled, lookback)
    X_val_seq, y_val_seq = make_sequences(X_val_scaled, y_val_scaled, lookback)
    X_test_seq, y_test_seq = make_sequences(X_test_scaled, y_test_scaled, lookback)
    
    print(f"\nSequences: Train={X_train_seq.shape}, Val={X_val_seq.shape}, Test={X_test_seq.shape}")
    
    # 6. Package results
    train_data = {'X': X_train_seq, 'y': y_train_seq}
    val_data = {'X': X_val_seq, 'y': y_val_seq}
    test_data = {'X': X_test_seq, 'y': y_test_seq}
    scalers = {'feature_scaler': feature_scaler, 'target_scaler': target_scaler, 'feature_names': feature_cols}
    
    return train_data, val_data, test_data, scalers


def make_sequences(X, y, lookback):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def save_scalers(scalers: dict, path: str = 'models/scalers.pkl'):
    """Save scalers for inference"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Saved scalers to {path}")


def load_scalers(path: str = 'models/scalers.pkl') -> dict:
    """Load scalers for inference"""
    with open(path, 'rb') as f:
        scalers = pickle.load(f)
    print(f"Loaded scalers from {path}")
    return scalers


# Example usage
if __name__ == "__main__":
    # Load featured data
    df = pd.read_parquet('data/features/AAPL_features.parquet')
    
    # Prepare sequences
    train, val, test, scalers = prepare_lstm_data(df, lookback=60)
    
    # Save
    np.savez('data/sequences/train.npz', **train)
    np.savez('data/sequences/val.npz', **val)
    np.savez('data/sequences/test.npz', **test)
    save_scalers(scalers)
    