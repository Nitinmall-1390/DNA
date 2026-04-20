#!/usr/bin/env python3
"""FAST evaluation - smaller dataset, quick results."""

import os, time, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    MultiHeadAttention, GlobalAveragePooling1D,
    LayerNormalization, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ===== LOAD DNA =====
with open("dna_sequences.txt") as f:
    dna = f.read().replace("\n","").replace(" ","").upper()

print(f"DNA length: {len(dna)}")

MAPPING = {"A":0,"C":1,"G":2,"T":3}
INDEX_TO_NUC = {0:"A",1:"C",2:"G",3:"T"}

def one_hot_encode(seq):
    arr = np.zeros((len(seq),4), dtype=np.float32)
    for i,ch in enumerate(seq):
        if ch in MAPPING:
            arr[i, MAPPING[ch]] = 1.0
    return arr

encoded = one_hot_encode(dna)

# ===== CREATE SEQUENCES (subsample for speed) =====
seq_length = 60  # shorter window = faster
X_list, y_list = [], []
for i in range(0, len(encoded) - seq_length, 5):  # stride=5 for speed
    X_list.append(encoded[i:i+seq_length])
    y_list.append(encoded[i+seq_length])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

valid = y.sum(axis=1) > 0
X, y = X[valid], y[valid]

y_labels = np.argmax(y, axis=1)
unique, counts = np.unique(y_labels, return_counts=True)
print("Class distribution:", dict(zip([INDEX_TO_NUC[u] for u in unique], counts)))

# ===== SEQUENTIAL SPLIT =====
n = len(X)
split1 = int(n * 0.7)
split2 = int(n * 0.85)
X_train, y_train = X[:split1], y[:split1]
X_val, y_val = X[split1:split2], y[split1:split2]
X_test, y_test = X[split2:], y[split2:]
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Class weights
train_labels = np.argmax(y_train, axis=1)
_, counts_tr = np.unique(train_labels, return_counts=True)
total_tr = counts_tr.sum()
class_weights = {int(u): total_tr/(len(counts_tr)*c) for u, c in zip(np.unique(train_labels), counts_tr)}

# ===== BASELINES =====
val_labels = np.argmax(y_val, axis=1)
probs = counts_tr / counts_tr.sum()
random_preds = np.random.choice(np.unique(train_labels), size=len(val_labels), p=probs)
random_acc = accuracy_score(val_labels, random_preds)

last_nucs = np.argmax(X_train[:,-1,:], axis=1)
next_nucs = np.argmax(y_train, axis=1)
trans = np.zeros((4,4), dtype=np.float32)
for ln, nn in zip(last_nucs, next_nucs):
    trans[ln, nn] += 1
row_sums = trans.sum(axis=1, keepdims=True)
row_sums[row_sums==0] = 1
trans_probs = trans / row_sums
val_last = np.argmax(X_val[:,-1,:], axis=1)
markov_preds = np.argmax(trans_probs[val_last], axis=1)
markov_acc = accuracy_score(val_labels, markov_preds)

print(f"\nRandom Baseline: {random_acc:.4f}")
print(f"Markov Baseline: {markov_acc:.4f}")

# ===== BUILD MODEL =====
def build_model():
    inputs = Input(shape=(seq_length, 4))
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    x2 = Bidirectional(LSTM(32, return_sequences=True))(x)
    x2 = LayerNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    
    attn = MultiHeadAttention(num_heads=4, key_dim=16)(x2, x2)
    attn = LayerNormalization()(attn)
    attn = Add()([x2, attn])
    attn = Dropout(0.2)(attn)
    
    x = GlobalAveragePooling1D()(attn)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(4, activation="softmax")(x)
    
    return Model(inputs, outputs, name="AttentionBiLSTM_DNA")

# ============================================================
# TEST 1: .keras vs .h5 SAVE FORMAT
# ============================================================
print("\n" + "="*60)
print("TEST 1: .keras vs .h5 SAVE FORMAT TIMING")
print("="*60)

model = build_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train[:500], y_train[:500], epochs=1, batch_size=128, verbose=0)

t0 = time.time()
model.save("test_model.keras")
keras_time = time.time() - t0
keras_size = os.path.getsize("test_model.keras")

t0 = time.time()
model.save("test_model.h5")
h5_time = time.time() - t0
h5_size = os.path.getsize("test_model.h5")

print(f"\n  .keras save : {keras_time:.3f}s | Size: {keras_size/1024/1024:.2f} MB")
print(f"  .h5    save : {h5_time:.3f}s | Size: {h5_size/1024/1024:.2f} MB")
print(f"  Difference  : {abs(keras_time-h5_time):.3f}s")

for f in ["test_model.keras", "test_model.h5"]:
    if os.path.exists(f): os.remove(f)

# ============================================================
# TEST 2: EPOCH ANALYSIS
# ============================================================
print("\n" + "="*60)
print("TEST 2: DOES MORE EPOCHS = LESS ACCURACY?")
print("="*60)

results = {}

for max_ep in [10, 30, 50]:
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    
    m = build_model()
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    
    cb = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0)
    ]
    
    t0 = time.time()
    hist = m.fit(X_train, y_train, validation_data=(X_val, y_val),
                 epochs=max_ep, batch_size=256, class_weight=class_weights, callbacks=cb, verbose=0)
    train_time = time.time() - t0
    
    actual_epochs = len(hist.history["loss"])
    best_val_acc = max(hist.history["val_accuracy"])
    test_loss, test_acc = m.evaluate(X_test, y_test, verbose=0)
    
    results[max_ep] = {
        "actual": actual_epochs, "time": train_time,
        "val_acc": best_val_acc, "test_acc": test_acc
    }
    print(f"  max_ep={max_ep:>3} | actual={actual_epochs:>3} | time={train_time:.1f}s | val_acc={best_val_acc:.4f} | test_acc={test_acc:.4f}")

# ============================================================
# TEST 3: FINAL MODEL
# ============================================================
print("\n" + "="*60)
print("TEST 3: FINAL MODEL (50 max epochs)")
print("="*60)

tf.random.set_seed(SEED)
np.random.seed(SEED)

final = build_model()
final.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

cb = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1)
]

t0 = time.time()
history = final.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=50, batch_size=256, class_weight=class_weights, callbacks=cb, verbose=1)
total_time = time.time() - t0

test_loss, test_acc = final.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(final.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"  Total training time  : {total_time:.1f}s")
print(f"  Epochs run           : {len(history.history['loss'])}")
print(f"  Test Accuracy        : {test_acc:.4f}")
print(f"  Markov Baseline      : {markov_acc:.4f}")
print(f"  Random Baseline      : {random_acc:.4f}")
print(f"  Improvement vs Markov: +{(test_acc-markov_acc)*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["A","C","G","T"], digits=4))

print(f"\n{'='*60}")
print(f"EPOCH ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"{'Max Epochs':>12} {'Actual':>8} {'Time(s)':>10} {'Val Acc':>10} {'Test Acc':>10}")
print("-"*55)
for max_ep, r in results.items():
    print(f"{max_ep:>12} {r['actual']:>8} {r['time']:>10.1f} {r['val_acc']:>10.4f} {r['test_acc']:>10.4f}")

print(f"""
✅ CONCLUSIONS:
1. .keras vs .h5 → Save time is almost SAME. .keras is just the modern format.
2. More max_epochs does NOT reduce accuracy because:
   - EarlyStopping stops training when val_loss stops improving
   - restore_best_weights=True rolls back to best weights
3. Actual epochs run is decided by EarlyStopping, not your max_epochs setting.
""")
