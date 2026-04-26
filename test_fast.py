#!/usr/bin/env python3
"""
Quick test script to verify the fast model runs correctly.
Runs only 3 epochs for fast validation.
"""

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Patch the config in fast version to use fewer epochs
import attention_bilstm_dna_fast as fast

original_cfg = fast.cfg
fast.cfg.epochs = 3
fast.cfg.model_path = "test_model.keras"
fast.cfg.plot_dir = "plots_test"

print("Testing fast model with 3 epochs...")
print(f"Sequence length: {fast.cfg.seq_length}")
print(f"Batch size: {fast.cfg.batch_size}")
print(f"Epochs: {fast.cfg.epochs}")

# Run the main function
import tensorflow as tf
print("\nTensorFlow devices:", tf.config.list_physical_devices())

# Test data pipeline
print("\nTesting data pipeline...")
encoded = fast.encode_dna(fast.dna)[0]
print(f"Encoded shape: {encoded.shape}")

X_list, y_list = [], []
for i in range(len(encoded) - fast.cfg.seq_length):
    X_list.append(encoded[i:i+fast.cfg.seq_length])
    y_list.append(encoded[i+fast.cfg.seq_length])

X = fast.np.array(X_list, dtype=fast.np.float32)
y = fast.np.array(y_list, dtype=fast.np.float32)
valid_mask = y.sum(axis=1) > 0
X = X[valid_mask]
y = y[valid_mask]
print(f"Training samples: {len(X)}")

# Build and train
model = fast.build_model(fast.cfg)
print(f"\nModel built: {model.count_params()} parameters")

# Quick train on subset
n_sub = min(5000, len(X))
X_sub, y_sub = X[:n_sub], y[:n_sub]

history = model.fit(
    X_sub, y_sub,
    validation_split=0.2,
    epochs=3,
    batch_size=fast.cfg.batch_size,
    verbose=1
)

print("\nTest passed! Model trains successfully.")
print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
