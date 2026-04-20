#!/usr/bin/env python3
"""
==========================================================================
Reliable Attention-BiLSTM DNA Sequence Prediction Model (Production Grade)
==========================================================================

Key improvements over the original:
  1. Sequential data splitting — NO data leakage
  2. Handles unknown nucleotides gracefully
  3. MultiHeadAttention instead of redundant self-attention
  4. Residual connections between BiLSTM layers
  5. Layer normalization for training stability
  6. Proper train/val/test split with held-out test set
  7. Class-weighted loss for imbalanced nucleotide distributions
  8. Baseline comparison (Markov chain + random)
  9. Comprehensive evaluation (confusion matrix, per-class metrics)
 10. Autoregressive sequence generation
 11. Full reproducibility (seeds)
 12. Modern .keras save format with restore_best_weights

Author : Auto-generated
Date   : 2026-04-20
"""

# ============================================================
# 0. Imports & Reproducibility
# ============================================================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    MultiHeadAttention, GlobalAveragePooling1D,
    LayerNormalization, Add
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- Font setup (for plots that may contain CJK or special chars) ---
fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf")
fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# --- Reproducibility ---
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ============================================================
# 1. Configuration (all hyperparameters in one place)
# ============================================================
class Config:
    # Data
    dna_file        = "dna_sequences.txt"
    seq_length      = 100          # sliding-window length
    stride          = 1            # step between windows (1 = max data)

    # Model
    lstm_units_1    = 128          # first BiLSTM layer units (per direction)
    lstm_units_2    = 64           # second BiLSTM layer units (per direction)
    attention_heads = 8            # multi-head attention heads
    key_dim         = 32           # dimension of each attention head
    dense_units     = 64           # dense layer before output
    dropout_lstm    = 0.3          # dropout after BiLSTM layers
    dropout_attn    = 0.2          # dropout after attention
    dropout_dense   = 0.3          # dropout after dense

    # Training
    epochs          = 100
    batch_size      = 128
    learning_rate   = 1e-3
    patience_es     = 10           # early-stopping patience
    patience_lr     = 5            # reduce-LR patience
    lr_factor       = 0.5          # LR reduction factor

    # Splits (sequential, no leakage)
    train_ratio     = 0.70
    val_ratio       = 0.15         # remaining 0.15 = test

    # Output
    model_path      = "attention_bilstm_dna_model.keras"
    plot_dir        = "plots"

cfg = Config()

# ============================================================
# 2. Load & Preprocess DNA Sequence
# ============================================================
print("=" * 60)
print("STEP 1 : Loading & Preprocessing DNA Sequence")
print("=" * 60)

with open(cfg.dna_file) as f:
    raw_dna = f.read()

# Clean: remove whitespace, uppercase
dna = raw_dna.replace("\n", "").replace("\r", "").replace(" ", "").upper()
print(f"  Raw characters loaded  : {len(raw_dna)}")
print(f"  Cleaned sequence length: {len(dna)}")

# --- Nucleotide statistics before encoding ---
valid_nucs = {"A", "C", "G", "T"}
nuc_counts_raw = {n: dna.count(n) for n in valid_nucs}
unknown_count  = sum(1 for ch in dna if ch not in valid_nucs)
print(f"  Nucleotide counts      : {nuc_counts_raw}")
print(f"  Unknown characters     : {unknown_count}")

if unknown_count > 0:
    pct = unknown_count / len(dna) * 100
    print(f"  ⚠  {unknown_count} unknown chars ({pct:.2f}%) will be encoded as [0,0,0,0]")

# ============================================================
# 3. One-Hot Encoding (handles unknowns gracefully)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2 : One-Hot Encoding")
print("=" * 60)

MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}
INDEX_TO_NUC = {v: k for k, v in MAPPING.items()}

def one_hot_encode(seq):
    """Encode DNA string → (len(seq), 4) one-hot array.
    Unknown nucleotides become [0, 0, 0, 0]."""
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, ch in enumerate(seq):
        if ch in MAPPING:
            arr[i, MAPPING[ch]] = 1.0
    return arr

encoded = one_hot_encode(dna)
print(f"  Encoded shape: {encoded.shape}")

# ============================================================
# 4. Create Sliding-Window Sequences
# ============================================================
print("\n" + "=" * 60)
print("STEP 3 : Creating Sliding-Window Sequences")
print("=" * 60)

X_list, y_list = [], []
for i in range(0, len(encoded) - cfg.seq_length, cfg.stride):
    X_list.append(encoded[i : i + cfg.seq_length])
    y_list.append(encoded[i + cfg.seq_length])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

# Filter out samples where the target is all-zeros (unknown nucleotide)
valid_mask = y.sum(axis=1) > 0
X = X[valid_mask]
y = y[valid_mask]

print(f"  Total samples       : {len(X)}")
print(f"  X shape             : {X.shape}")
print(f"  y shape             : {y.shape}")
print(f"  Filtered (unknowns) : {(~valid_mask).sum()} samples removed")

# --- Class distribution ---
y_labels = np.argmax(y, axis=1)
unique, counts = np.unique(y_labels, return_counts=True)
dist = dict(zip([INDEX_TO_NUC[u] for u in unique], counts))
print(f"  Class distribution  : {dist}")
for nuc, cnt in dist.items():
    print(f"    {nuc}: {cnt:>8d}  ({cnt / len(y) * 100:.2f}%)")

# ============================================================
# 5. Sequential Train / Val / Test Split (NO leakage)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4 : Sequential Train/Val/Test Split")
print("=" * 60)

n = len(X)
split1 = int(n * cfg.train_ratio)
split2 = int(n * (cfg.train_ratio + cfg.val_ratio))

X_train, y_train = X[:split1], y[:split1]
X_val,   y_val   = X[split1:split2], y[split1:split2]
X_test,  y_test  = X[split2:], y[split2:]

print(f"  Train : {X_train.shape[0]:>8d} samples  (0 → {split1-1})")
print(f"  Val   : {X_val.shape[0]:>8d} samples  ({split1} → {split2-1})")
print(f"  Test  : {X_test.shape[0]:>8d} samples  ({split2} → {n-1})")

# ============================================================
# 6. Compute Class Weights (handles imbalance)
# ============================================================
train_labels = np.argmax(y_train, axis=1)
unique_tr, counts_tr = np.unique(train_labels, return_counts=True)
total_tr = counts_tr.sum()
class_weights_dict = {
    int(u): total_tr / (len(unique_tr) * c) for u, c in zip(unique_tr, counts_tr)
}
print(f"\n  Class weights: {class_weights_dict}")

# ============================================================
# 7. Baseline Models (for comparison)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5 : Baseline Comparisons")
print("=" * 60)

# --- Baseline A: Random prediction proportional to class freq ---
val_labels = np.argmax(y_val, axis=1)
probs = counts_tr / counts_tr.sum()
random_preds = np.random.choice(unique_tr, size=len(val_labels), p=probs)
random_acc = accuracy_score(val_labels, random_preds)
print(f"  Baseline (frequency-random) val accuracy : {random_acc:.4f}")

# --- Baseline B: Simple 1st-order Markov chain ---
print("  Building 1st-order Markov chain from training data...")
# Count transitions: last nucleotide of each X_train → y_train
last_nucs = np.argmax(X_train[:, -1, :], axis=1)  # last position in each window
next_nucs = np.argmax(y_train, axis=1)

trans_counts = np.zeros((4, 4), dtype=np.float32)
for ln, nn in zip(last_nucs, next_nucs):
    trans_counts[ln, nn] += 1

# Normalize rows
row_sums = trans_counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # avoid division by zero
trans_probs = trans_counts / row_sums

# Predict on val set
val_last = np.argmax(X_val[:, -1, :], axis=1)
markov_preds = np.argmax(trans_probs[val_last], axis=1)
markov_acc = accuracy_score(val_labels, markov_preds)
print(f"  Baseline (1st-order Markov) val accuracy  : {markov_acc:.4f}")

# ============================================================
# 8. Build the Model
# ============================================================
print("\n" + "=" * 60)
print("STEP 6 : Building Attention-BiLSTM Model")
print("=" * 60)

def build_model(cfg):
    """Build an Attention-BiLSTM model with residual connections
    and layer normalization."""

    inputs = Input(shape=(cfg.seq_length, 4), name="dna_input")

    # ---- BiLSTM Block 1 ----
    x = Bidirectional(
        LSTM(cfg.lstm_units_1, return_sequences=True, name="lstm1_fwd"),
        name="bilstm1"
    )(inputs)
    x = LayerNormalization(name="ln1")(x)
    x = Dropout(cfg.dropout_lstm, name="drop1")(x)

    # ---- BiLSTM Block 2 (with residual) ----
    # Project input to match x's dimensionality for residual connection
    residual = Dense(cfg.lstm_units_1 * 2, name="residual_proj")(inputs)

    x2 = Bidirectional(
        LSTM(cfg.lstm_units_2, return_sequences=True, name="lstm2_fwd"),
        name="bilstm2"
    )(x)
    x2 = LayerNormalization(name="ln2")(x2)

    # Project x to match x2's dimensionality for residual
    if cfg.lstm_units_1 * 2 != cfg.lstm_units_2 * 2:
        residual = Dense(cfg.lstm_units_2 * 2, name="residual_proj2")(residual)
        x = Dense(cfg.lstm_units_2 * 2, name="x_proj")(x)

    # Residual: add projected input to BiLSTM2 output
    x2 = Add(name="residual_add")([residual[:, -x2.shape[1]:, :], x2])
    x2 = Dropout(cfg.dropout_lstm, name="drop2")(x2)

    # ---- Multi-Head Self-Attention ----
    attn_output = MultiHeadAttention(
        num_heads=cfg.attention_heads,
        key_dim=cfg.key_dim,
        name="multihead_attention"
    )(x2, x2)
    attn_output = LayerNormalization(name="ln_attn")(attn_output)

    # Another residual: attention over BiLSTM output
    attn_output = Add(name="attn_residual")([x2, attn_output])
    attn_output = Dropout(cfg.dropout_attn, name="drop_attn")(attn_output)

    # ---- Pooling & Dense ----
    x = GlobalAveragePooling1D(name="gap")(attn_output)

    x = Dense(cfg.dense_units, activation="relu", name="dense1")(x)
    x = LayerNormalization(name="ln_dense")(x)
    x = Dropout(cfg.dropout_dense, name="drop_dense")(x)

    outputs = Dense(4, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name="AttentionBiLSTM_DNA")
    return model

model = build_model(cfg)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 9. Callbacks
# ============================================================
os.makedirs(cfg.plot_dir, exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=cfg.patience_es,
        restore_best_weights=True,   # ← critical for reliability
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=cfg.lr_factor,
        patience=cfg.patience_lr,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=cfg.model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# ============================================================
# 10. Train
# ============================================================
print("\n" + "=" * 60)
print("STEP 7 : Training")
print("=" * 60)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=cfg.epochs,
    batch_size=cfg.batch_size,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 11. Training Curves
# ============================================================
print("\n" + "=" * 60)
print("STEP 8 : Plotting Training Curves")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history["loss"], label="Train Loss", linewidth=1.5)
axes[0].plot(history.history["val_loss"], label="Val Loss", linewidth=1.5)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curves")
axes[0].legend(loc="best")
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history.history["accuracy"], label="Train Accuracy", linewidth=1.5)
axes[1].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=1.5)
axes[1].axhline(y=markov_acc, color="orange", linestyle="--",
                label=f"Markov Baseline ({markov_acc:.4f})")
axes[1].axhline(y=random_acc, color="red", linestyle="--",
                label=f"Random Baseline ({random_acc:.4f})")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy Curves")
axes[1].legend(loc="best")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
loss_curve_path = os.path.join(cfg.plot_dir, "training_curves.png")
plt.savefig(loss_curve_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved training curves → {loss_curve_path}")

# ============================================================
# 12. Final Evaluation on Test Set (never seen before)
# ============================================================
print("\n" + "=" * 60)
print("STEP 9 : Final Evaluation on Held-Out Test Set")
print("=" * 60)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n  Test Loss     : {test_loss:.4f}")
print(f"  Test Accuracy : {test_acc:.4f}")
print(f"  Markov Baseline: {markov_acc:.4f}")
print(f"  Random Baseline: {random_acc:.4f}")
improvement_vs_markov = (test_acc - markov_acc) * 100
improvement_vs_random = (test_acc - random_acc) * 100
print(f"  Improvement vs Markov : +{improvement_vs_markov:.2f}%")
print(f"  Improvement vs Random : +{improvement_vs_random:.2f}%")

# --- Per-class metrics ---
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

target_names = ["A", "C", "G", "T"]
print("\n  Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=target_names,
       yticklabels=target_names,
       title="Confusion Matrix (Test Set)",
       xlabel="Predicted Label",
       ylabel="True Label")

# Annotate cells
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12)

plt.tight_layout()
cm_path = os.path.join(cfg.plot_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved confusion matrix → {cm_path}")

# --- Per-nucleotide accuracy bar chart ---
per_class_acc = cm.diagonal() / cm.sum(axis=1)
fig, ax = plt.subplots(figsize=(6, 5))
colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
bars = ax.bar(target_names, per_class_acc, color=colors, edgecolor="black", linewidth=0.8)
for bar, acc in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{acc:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0, max(per_class_acc) + 0.1)
ax.set_xlabel("Nucleotide")
ax.set_ylabel("Accuracy")
ax.set_title("Per-Nucleotide Prediction Accuracy (Test Set)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
nuc_acc_path = os.path.join(cfg.plot_dir, "per_nucleotide_accuracy.png")
plt.savefig(nuc_acc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved per-nucleotide accuracy → {nuc_acc_path}")

# ============================================================
# 13. Autoregressive Sequence Generation
# ============================================================
print("\n" + "=" * 60)
print("STEP 10 : Autoregressive DNA Sequence Generation")
print("=" * 60)

def generate_dna(model, seed_seq, length=200, temperature=1.0):
    """Generate DNA sequence autoregressively.

    Parameters
    ----------
    model      : trained Keras model
    seed_seq   : one-hot encoded seed of shape (seq_length, 4)
    length     : number of nucleotides to generate
    temperature: sampling temperature (>1 = more random, <1 = more greedy)

    Returns
    -------
    generated : str, the generated DNA string (including seed)
    """
    current = seed_seq.copy().reshape(1, cfg.seq_length, 4)
    generated_indices = list(np.argmax(seed_seq, axis=1))

    for _ in range(length):
        preds = model.predict(current, verbose=0)[0]  # shape (4,)

        # Temperature scaling
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / exp_preds.sum()

        # Sample from distribution
        next_idx = np.random.choice(4, p=preds)
        generated_indices.append(next_idx)

        # Slide window: drop first, append new
        next_onehot = np.zeros((1, 1, 4), dtype=np.float32)
        next_onehot[0, 0, next_idx] = 1.0
        current = np.concatenate([current[:, 1:, :], next_onehot], axis=1)

    return "".join(INDEX_TO_NUC.get(i, "N") for i in generated_indices)

# Pick a random seed from test set
seed_idx = np.random.randint(0, len(X_test))
seed = X_test[seed_idx]

seed_str = "".join(INDEX_TO_NUC.get(int(np.argmax(seed[j])), "N") for j in range(cfg.seq_length))
print(f"  Seed (first 50 nt): {seed_str[:50]}...")

for temp in [0.5, 0.8, 1.0, 1.2]:
    generated = generate_dna(model, seed, length=200, temperature=temp)
    # Count nucleotide distribution in generated part
    gen_only = generated[cfg.seq_length:]
    gen_counts = {n: gen_only.count(n) for n in "ACGT"}
    gen_total = sum(gen_counts.values())
    gen_dist = {n: f"{c/gen_total*100:.1f}%" for n, c in gen_counts.items()}
    print(f"\n  Temperature={temp}:")
    print(f"    Generated (50 nt): {gen_only[:50]}...")
    print(f"    Distribution       : {gen_dist}")

# ============================================================
# 14. Save Model (modern format)
# ============================================================
print("\n" + "=" * 60)
print("STEP 11 : Saving Model")
print("=" * 60)

model.save(cfg.model_path)
print(f"  Model saved → {cfg.model_path}")

# ============================================================
# 15. Final Summary
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
  Model Architecture     : Attention-BiLSTM with Residual Connections
  Sequence Length         : {cfg.seq_length}
  Training Samples       : {len(X_train)}
  Validation Samples     : {len(X_val)}
  Test Samples           : {len(X_test)}

  Test Accuracy          : {test_acc:.4f}
  Markov Baseline        : {markov_acc:.4f}
  Random Baseline        : {random_acc:.4f}
  Improvement vs Markov  : +{improvement_vs_markov:.2f}%
  Improvement vs Random  : +{improvement_vs_random:.2f}%

  Model File             : {cfg.model_path}
  Plots Directory        : {cfg.plot_dir}/
    - training_curves.png
    - confusion_matrix.png
    - per_nucleotide_accuracy.png
""")
