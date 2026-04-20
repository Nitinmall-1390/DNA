#!/usr/bin/env python3
"""
==========================================================================
  Attention-BiLSTM DNA Sequence Prediction — FAST CPU VERSION
==========================================================================

Optimizations for CPU training:
  1. REDUCED seq_length 200 → 50  (5x fewer timesteps)
  2. SIMPLIFIED architecture — 2 BiLSTM layers instead of 3
  3. SMALLER LSTM units 256→64, 128→32  (8x fewer params)
  4. REDUCED attention heads 8→2
  5. REMOVED embedding — one-hot only (faster on CPU)
  6. INCREASED batch_size 64→256  (better CPU utilization)
  7. OPTIMIZED data pipeline with tf.data
  8. DISABLED mixed precision (CPU incompatible)
  9. FASTER callbacks — removed cosine schedule overhead
 10. REDUCED epochs 100→40  (early stopping still active)

Expected speedup: 10-20x on CPU vs original v2

Usage:
  python attention_bilstm_dna_fast.py

Requirements:
  pip install numpy tensorflow scikit-learn matplotlib
"""

import os, sys, time, math, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # CPU optimizations

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    MultiHeadAttention, GlobalAveragePooling1D,
    LayerNormalization, Add, Concatenate
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Font setup
for fp in ["/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# CPU optimizations
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

# GPU check
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"[INFO] GPU: {gpus[0].name}")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
else:
    print("[INFO] No GPU — using CPU-optimized configuration")

# ============================================================
# 1. CONFIG — optimized for FAST CPU training
# ============================================================
class Config:
    # Data
    dna_file            = "dna_sequences.txt"
    seq_length          = 50           # ← was 200, now 50 (4x faster)
    stride              = 1

    # Model — dramatically reduced for CPU
    lstm_units_1        = 64           # ← was 256
    lstm_units_2        = 32           # ← was 128
    attention_heads     = 2            # ← was 8
    key_dim             = 16           # ← was 32
    dense_units_1       = 64           # ← was 128
    dropout_lstm        = 0.2
    dropout_attn        = 0.15
    dropout_dense       = 0.2
    label_smoothing     = 0.1

    # Training — faster on CPU
    epochs              = 40           # ← was 100
    batch_size          = 256          # ← was 64 (larger batches for CPU)
    learning_rate       = 5e-4
    max_grad_norm       = 1.0
    patience_es         = 10
    patience_lr         = 3
    lr_factor           = 0.5

    # Splits
    train_ratio         = 0.70
    val_ratio           = 0.15

    # Output
    model_path          = "attention_bilstm_dna_fast.keras"
    plot_dir            = "plots_fast"

    # Imbalance threshold
    imbalance_threshold = 1.5

cfg = Config()

# ============================================================
# 2. Load DNA
# ============================================================
print("\n" + "=" * 65)
print("  STEP 1 : Loading DNA Sequence")
print("=" * 65)

if not os.path.exists(cfg.dna_file):
    print(f"  [ERROR] '{cfg.dna_file}' not found!")
    sys.exit(1)

with open(cfg.dna_file) as f:
    raw_dna = f.read()

dna = raw_dna.replace("\n","").replace("\r","").replace(" ","").upper()
print(f"  Sequence length: {len(dna)}")

MAPPING = {"A":0,"C":1,"G":2,"T":3}
INDEX_TO_NUC = {0:"A",1:"C",2:"G",3:"T"}
valid_nucs = set(MAPPING.keys())

nuc_counts = {n: dna.count(n) for n in valid_nucs}
unknown_count = sum(1 for ch in dna if ch not in valid_nucs)
print(f"  Nucleotide counts: {nuc_counts}")
if unknown_count > 0:
    print(f"  Unknown chars: {unknown_count} ({unknown_count/len(dna)*100:.2f}%)")

# Check imbalance
max_count = max(nuc_counts.values())
min_count = min(nuc_counts.values())
imbalance_ratio = max_count / max(min_count, 1)
print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}")
USE_CLASS_WEIGHT = imbalance_ratio > cfg.imbalance_threshold
print(f"  Using class_weight: {USE_CLASS_WEIGHT}")

# ============================================================
# 3. Encode — One-Hot only (removed embedding for speed)
# ============================================================
print("\n" + "=" * 65)
print("  STEP 2 : Encoding")
print("=" * 65)

def encode_dna(seq):
    """Returns one-hot array only."""
    onehot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, ch in enumerate(seq):
        if ch in MAPPING:
            onehot[i, MAPPING[ch]] = 1.0
    return onehot

encoded = encode_dna(dna)
print(f"  Encoded shape: {encoded.shape}")

# ============================================================
# 4. Create Sequences with tf.data later
# ============================================================
print("\n" + "=" * 65)
print("  STEP 3 : Creating Sequences")
print("=" * 65)

X_list, y_list = [], []
for i in range(len(encoded) - cfg.seq_length):
    X_list.append(encoded[i:i+cfg.seq_length])
    y_list.append(encoded[i+cfg.seq_length])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

# Filter unknowns
valid_mask = y.sum(axis=1) > 0
X = X[valid_mask]
y = y[valid_mask]

print(f"  Samples: {len(X)}")
print(f"  X shape  : {X.shape}")
print(f"  y shape  : {y.shape}")

# Class distribution
y_labels = np.argmax(y, axis=1)
unique, counts = np.unique(y_labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"    {INDEX_TO_NUC[u]}: {c:>8d} ({c/len(y)*100:.2f}%)")

# ============================================================
# 5. Sequential Split
# ============================================================
print("\n" + "=" * 65)
print("  STEP 4 : Sequential Split (NO leakage)")
print("=" * 65)

n = len(X)
s1 = int(n * cfg.train_ratio)
s2 = int(n * (cfg.train_ratio + cfg.val_ratio))

X_train, X_val, X_test = X[:s1], X[s1:s2], X[s2:]
y_train, y_val, y_test = y[:s1], y[s1:s2], y[s2:]

print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

# Class weights (only if imbalanced)
class_weights_dict = None
if USE_CLASS_WEIGHT:
    tr_labels = np.argmax(y_train, axis=1)
    u_tr, c_tr = np.unique(tr_labels, return_counts=True)
    total = c_tr.sum()
    class_weights_dict = {int(u): total/(len(c_tr)*c) for u,c in zip(u_tr, c_tr)}
    print(f"  Class weights: {class_weights_dict}")

# ============================================================
# 6. Baselines
# ============================================================
print("\n" + "=" * 65)
print("  STEP 5 : Baseline Comparisons")
print("=" * 65)

val_labels = np.argmax(y_val, axis=1)

# Random baseline
tr_labels = np.argmax(y_train, axis=1)
u_tr, c_tr = np.unique(tr_labels, return_counts=True)
probs = c_tr / c_tr.sum()
random_preds = np.random.choice(u_tr, size=len(val_labels), p=probs)
random_acc = accuracy_score(val_labels, random_preds)
print(f"  Random baseline: {random_acc:.4f}")

# 1st-order Markov
last_nucs = np.argmax(X_train[:,-1,:], axis=1)
next_nucs = np.argmax(y_train, axis=1)
trans = np.zeros((4,4), dtype=np.float32)
for ln, nn in zip(last_nucs, next_nucs):
    trans[ln, nn] += 1
rs = trans.sum(axis=1, keepdims=True)
rs[rs==0] = 1
trans_probs = trans / rs
val_last = np.argmax(X_val[:,-1,:], axis=1)
markov_preds = np.argmax(trans_probs[val_last], axis=1)
markov_acc = accuracy_score(val_labels, markov_preds)
print(f"  1st-order Markov: {markov_acc:.4f}")

# ============================================================
# 7. Build Model — CPU-OPTIMIZED
# ============================================================
print("\n" + "=" * 65)
print("  STEP 6 : Building Fast Model")
print("=" * 65)

def build_model(cfg):
    inp = Input(shape=(cfg.seq_length, 4), name="input")

    # Single BiLSTM block (fast)
    x = Bidirectional(LSTM(cfg.lstm_units_1, return_sequences=True), name="bilstm1")(inp)
    x = LayerNormalization(name="ln1")(x)
    x = Dropout(cfg.dropout_lstm, name="drop1")(x)

    # Second BiLSTM with residual
    residual = Dense(cfg.lstm_units_2 * 2, name="res_proj")(x)
    x2 = Bidirectional(LSTM(cfg.lstm_units_2, return_sequences=True), name="bilstm2")(x)
    x2 = LayerNormalization(name="ln2")(x2)
    x2 = Add(name="res_add")([residual, x2])
    x2 = Dropout(cfg.dropout_lstm, name="drop2")(x2)

    # Light attention
    attn = MultiHeadAttention(
        num_heads=cfg.attention_heads, key_dim=cfg.key_dim, name="mha"
    )(x2, x2)
    attn = LayerNormalization(name="ln_attn")(attn)
    attn = Add(name="attn_res")([x2, attn])

    x = GlobalAveragePooling1D(name="gap")(attn)
    x = Dense(cfg.dense_units_1, activation="relu", name="dense")(x)
    x = Dropout(cfg.dropout_dense, name="drop_dense")(x)

    outputs = Dense(4, activation="softmax", name="output")(x)

    model = Model(inp, outputs, name="AttentionBiLSTM_DNA_Fast")
    return model

model = build_model(cfg)

model.compile(
    optimizer=Adam(learning_rate=cfg.learning_rate, clipnorm=cfg.max_grad_norm),
    loss=CategoricalCrossentropy(label_smoothing=cfg.label_smoothing),
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 8. Callbacks (simplified)
# ============================================================
os.makedirs(cfg.plot_dir, exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor="val_loss", patience=cfg.patience_es,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=cfg.lr_factor,
        patience=cfg.patience_lr, min_lr=1e-6, verbose=1
    ),
    ModelCheckpoint(
        filepath=cfg.model_path, monitor="val_loss",
        save_best_only=True, verbose=1
    )
]

# ============================================================
# 9. Train with tf.data for CPU efficiency
# ============================================================
print("\n" + "=" * 65)
print("  STEP 7 : Training (CPU-Optimized)")
print("=" * 65)

# Create tf.data dataset for efficient CPU pipeline
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=SEED)
train_ds = train_ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

t0 = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=cfg.epochs,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)
train_time = time.time() - t0
print(f"\n  Training time: {train_time:.1f}s")
print(f"  Epochs run: {len(history.history['loss'])}")
print(f"  Avg time/epoch: {train_time/len(history.history['loss']):.2f}s")

# ============================================================
# 10. Training Curves
# ============================================================
print("\n" + "=" * 65)
print("  STEP 8 : Training Curves")
print("=" * 65)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["loss"], label="Train Loss", lw=1.5)
axes[0].plot(history.history["val_loss"], label="Val Loss", lw=1.5)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curves"); axes[0].legend(loc="best")
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["accuracy"], label="Train Acc", lw=1.5)
axes[1].plot(history.history["val_accuracy"], label="Val Acc", lw=1.5)
axes[1].axhline(y=markov_acc, color="orange", ls="--",
                label=f"Markov ({markov_acc:.4f})")
axes[1].axhline(y=random_acc, color="red", ls="--",
                label=f"Random ({random_acc:.4f})")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy Curves"); axes[1].legend(loc="best")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(cfg.plot_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 11. Test Evaluation
# ============================================================
print("\n" + "=" * 65)
print("  STEP 9 : Test Evaluation")
print("=" * 65)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n  Test Accuracy     : {test_acc:.4f}")
print(f"  Markov baseline   : {markov_acc:.4f}")
print(f"  Random baseline   : {random_acc:.4f}")
print(f"  vs Markov        : +{(test_acc - markov_acc)*100:.2f}%")
print(f"  vs Random        : +{(test_acc - random_acc)*100:.2f}%")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n  Classification Report:")
print(classification_report(y_true, y_pred, target_names=["A","C","G","T"], digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=range(4), yticks=range(4),
       xticklabels=["A","C","G","T"], yticklabels=["A","C","G","T"],
       title="Confusion Matrix", xlabel="Predicted", ylabel="True")
th = cm.max() / 2.0
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm[i,j], ha="center", va="center",
                color="white" if cm[i,j] > th else "black",
                fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(cfg.plot_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()

# Per-nucleotide accuracy
per_acc = cm.diagonal() / cm.sum(axis=1)
fig, ax = plt.subplots(figsize=(6, 5))
colors = ["#4CAF50","#2196F3","#FF9800","#F44336"]
bars = ax.bar(["A","C","G","T"], per_acc, color=colors, edgecolor="black", lw=0.8)
for b, a in zip(bars, per_acc):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{a:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0, max(per_acc)+0.12)
ax.set_xlabel("Nucleotide"); ax.set_ylabel("Accuracy")
ax.set_title("Per-Nucleotide Accuracy (Test)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.plot_dir, "per_nucleotide_accuracy.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 12. SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"""
  Architecture          : 2-layer BiLSTM + Light MHA (CPU-optimized)
  Sequence Length       : {cfg.seq_length}
  Training Samples      : {len(y_train)}
  Validation Samples    : {len(y_val)}
  Test Samples          : {len(y_test)}
  Training Time         : {train_time:.1f}s
  Epochs Run            : {len(history.history['loss'])}

  Test Accuracy         : {test_acc:.4f}
  Markov Baseline       : {markov_acc:.4f}
  Random Baseline       : {random_acc:.4f}

  vs Markov             : +{(test_acc-markov_acc)*100:.2f}%
  vs Random             : +{(test_acc-random_acc)*100:.2f}%

  CPU Optimizations:
    [REDUCED] seq_length 200→50 (4x seq len reduction)
    [REDUCED] LSTM units 256→64, 128→32 (8x fewer params)
    [REMOVED] Embedding layer (one-hot only)
    [REMOVED] 3rd BiLSTM layer (2 layers instead of 3)
    [REDUCED] Attention heads 8→2
    [INCREASED] batch_size 64→256 (better CPU utilization)
    [ADDED] tf.data pipeline with prefetch
    [SIMPLIFIED] Removed cosine LR schedule

  Output Files:
    {cfg.model_path}
    {cfg.plot_dir}/training_curves.png
    {cfg.plot_dir}/confusion_matrix.png
    {cfg.plot_dir}/per_nucleotide_accuracy.png
""")
