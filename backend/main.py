"""
DNA LSTM Generator — FastAPI Backend
=====================================
Developed by Nitin Mall

Run locally:
    pip install fastapi uvicorn tensorflow pandas numpy scikit-learn python-multipart
    uvicorn main:app --reload --port 8000

The frontend talks to this server. It:
  1. Accepts a CSV upload
  2. Trains the LSTM on YOUR data (streams epoch-by-epoch progress)
  3. Generates sequences using the trained model
  4. Caches the model in memory so re-generation is instant
"""

import io
import os
import json
import random
import asyncio
import warnings
import threading
import numpy as np
import pandas as pd
from queue import Queue
from typing import Optional
import gc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    MultiHeadAttention, GlobalAveragePooling1D,
    LayerNormalization, Add, Embedding, Concatenate,
    Lambda
)
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
import math

app = FastAPI(title="DNA LSTM Generator — Nitin Mall")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory model cache ─────────────────────────────────────
class ModelStore:
    def __init__(self):
        self.model      = None
        self.char2idx   = None
        self.idx2char   = None
        self.config     = None
        self.sequences  = []
        self.trained    = False

store = ModelStore()

# ── Pydantic request models ───────────────────────────────────
class TrainConfig(BaseModel):
    sequence_col: str   = "sequence"
    min_seq_len:  int   = 100
    window_size:  int   = 100
    step:         int   = 3
    max_sequences:int   = 5000
    lstm_units_1: int   = 64
    lstm_units_2: int   = 32
    lstm_units_3: int   = 16
    attn_heads:   int   = 4
    dropout_rate: float = 0.2
    batch_size:   int   = 128
    epochs:       int   = 50
    learning_rate:float = 0.001
    val_split:    float = 0.15
    patience:     int   = 15
    label_smoothing: float = 0.1

class GenerateRequest(BaseModel):
    num_sequences: int   = 3
    gen_length:    int   = 100
    temperature:   float = 1.0
    seed:          Optional[str] = None

# ── Helpers ───────────────────────────────────────────────────
def clean_sequences(df: pd.DataFrame, col: str, min_len: int) -> list[str]:
    if col not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise ValueError(f'Column "{col}" not found. Available: {available}')
    seqs = (
        df[col].dropna().astype(str)
        .str.strip().str.upper()
        .apply(lambda s: "".join(c for c in s if c in "ACGT"))
        .pipe(lambda s: s[s.str.len() >= min_len])
        .tolist()
    )
    if not seqs:
        raise ValueError("No valid DNA sequences found after cleaning.")
    return seqs

MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}
INDEX_TO_NUC = {v: k for k, v in MAPPING.items()}

def encode_dna(seq):
    """Encode DNA string to both one-hot and integer-indexed arrays."""
    onehot = np.zeros((len(seq), 4), dtype=np.float32)
    integer = np.zeros(len(seq), dtype=np.int32)
    for i, ch in enumerate(seq):
        if ch in MAPPING:
            onehot[i, MAPPING[ch]] = 1.0
            integer[i] = MAPPING[ch] + 1  # 1-indexed for Embedding (0 is padding)
    return onehot, integer

def make_windows(sequences, window, step, max_samples):
    X_oh_list, X_int_list, y_list = [], [], []
    for seq in sequences:
        oh, integer = encode_dna(seq)
        for start in range(0, len(oh) - window, step):
            X_oh_list.append(oh[start: start + window])
            X_int_list.append(integer[start: start + window])
            y_list.append(oh[start + window])
        if len(X_oh_list) >= max_samples:
            break
    
    X_oh = np.array(X_oh_list, dtype=np.float32)
    X_int = np.array(X_int_list, dtype=np.int32)
    y = np.array(y_list, dtype=np.float32)

    # Filter unknowns
    valid_mask = y.sum(axis=1) > 0
    X_oh = X_oh[valid_mask]
    X_int = X_int[valid_mask]
    y = y[valid_mask]

    # Shuffle and trim
    indices = np.arange(len(X_oh))
    np.random.shuffle(indices)
    idx = indices[:max_samples]
    return X_oh[idx], X_int[idx], y[idx]

def positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding."""
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

def build_model(cfg: TrainConfig) -> tf.keras.Model:
    """Build v2 High Accuracy Model: 3-BiLSTM + MHA + Embedding + PosEnc."""
    inp_oh = Input(shape=(cfg.window_size, 4), name="onehot_input")
    inp_int = Input(shape=(cfg.window_size,), name="integer_input")

    # Embedding branch
    embed = Embedding(input_dim=5, output_dim=16, name="nuc_embedding")(inp_int)
    
    # Concat
    x = Concatenate(name="concat_encode")([inp_oh, embed]) # (batch, seq, 20)
    
    # Positional Encoding
    pe = positional_encoding(cfg.window_size, 20)
    pe_constant = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    x = Lambda(lambda t: t + pe_constant, name="add_pos_enc")(x)

    # BiLSTM 1
    x = Bidirectional(LSTM(32, return_sequences=True), name="bilstm1")(x)
    x = LayerNormalization(name="ln1")(x)
    x = Dropout(cfg.dropout_rate, name="drop1")(x)

    # BiLSTM 2 + Residual
    res1 = Dense(16 * 2, name="res_proj1")(x)
    x2 = Bidirectional(LSTM(16, return_sequences=True), name="bilstm2")(x)
    x2 = LayerNormalization(name="ln2")(x2)
    x2 = Add(name="res_add1")([res1, x2])
    x2 = Dropout(cfg.dropout_rate, name="drop2")(x2)

    # Multi-Head Attention (Now on x2)
    attn = MultiHeadAttention(num_heads=cfg.attn_heads, key_dim=32, name="mha")(x2, x2)
    attn = LayerNormalization(name="ln_attn")(attn)
    attn = Add(name="attn_res")([x2, attn])
    attn = Dropout(cfg.dropout_rate, name="drop_attn")(attn)

    # Output head
    x = GlobalAveragePooling1D(name="gap")(attn)
    x = Dense(32, activation="relu", name="dense1")(x)
    outputs = Dense(4, activation="softmax", name="output")(x)

    model = Model([inp_oh, inp_int], outputs, name="AttentionBiLSTM_DNA_v2")
    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate, clipnorm=1.0),
        loss=CategoricalCrossentropy(label_smoothing=cfg.label_smoothing),
        metrics=["accuracy"],
    )
    return model

def sample_temperature(probs, temperature):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.log(probs + 1e-8) / temperature
    exp_preds = np.exp(probs)
    probs = exp_preds / exp_preds.sum()
    return int(np.random.choice(4, p=probs))

def generate_one(model, seed_oh, seed_int, gen_length, temperature):
    """Seed_oh: (window, 4), Seed_int: (window,)"""
    cur_oh = seed_oh.copy().reshape(1, seed_oh.shape[0], 4)
    cur_int = seed_int.copy().reshape(1, seed_int.shape[0])
    out_indices = []
    
    for _ in range(gen_length):
        probs = model.predict([cur_oh, cur_int], verbose=0)[0]
        nxt = sample_temperature(probs, temperature)
        out_indices.append(nxt)
        
        # Slide window
        new_oh = np.zeros((1, 1, 4), dtype=np.float32)
        new_oh[0, 0, nxt] = 1.0
        new_int = np.array([[nxt + 1]], dtype=np.int32)
        
        cur_oh = np.concatenate([cur_oh[:, 1:, :], new_oh], axis=1)
        cur_int = np.concatenate([cur_int[:, 1:], new_int], axis=1)
        
    return "".join(INDEX_TO_NUC[i] for i in out_indices)

# ── SSE streaming callback ────────────────────────────────────
class StreamCallback(Callback):
    def __init__(self, q: Queue, total_epochs: int):
        super().__init__()
        self.q = q
        self.total = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Robust metric lookup
        loss = logs.get("loss") or logs.get("loss") or 0
        v_loss = logs.get("val_loss") or logs.get("val_loss") or 0
        acc = logs.get("accuracy") or logs.get("acc") or 0
        v_acc = logs.get("val_accuracy") or logs.get("val_acc") or 0

        self.q.put({
            "type":     "epoch",
            "epoch":    epoch + 1,
            "total":    self.total,
            "loss":     round(float(loss), 4),
            "val_loss": round(float(v_loss), 4),
            "acc":      round(float(acc) * 100, 2),
            "val_acc":  round(float(v_acc) * 100, 2),
        })
        print(f"DEBUG: Epoch {epoch + 1}/{self.total} complete. loss: {logs.get('loss', 0):.4f}")

    def on_train_end(self, logs=None):
        self.q.put({"type": "train_end"})

# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "DNA LSTM Backend running", "developer": "Nitin Mall"}

@app.get("/status")
def status():
    return {
        "trained":    store.trained,
        "sequences":  len(store.sequences),
        "config":     store.config,
    }

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), sequence_col: str = "sequence", min_seq_len: int = 50):
    """Parse and preview a CSV file — no training yet."""
    content = await file.read()
    try:
        raw = content.decode("utf-8-sig")          # handles UTF-8 BOM
        df = pd.read_csv(io.StringIO(raw))
        seqs = clean_sequences(df, sequence_col, min_seq_len)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    lengths = [len(s) for s in seqs]
    return {
        "rows":     len(seqs),
        "columns":  df.columns.tolist(),
        "avg_len":  round(sum(lengths) / len(lengths)),
        "min_len":  min(lengths),
        "max_len":  max(lengths),
        "sample":   seqs[:3],
    }

@app.post("/train")
async def train(file: UploadFile = File(...), config: str = "{}"):
    q: Queue = Queue()
    q.put({"type": "log", "msg": "[V5] Connection established. Waiting for server resources..."})

    # Read content once in main thread (required by FastAPI)
    content = await file.read()

    def run_training_v5(content_bytes, config_str):
        try:
            q.put({"type": "log", "msg": "[V5] Decoding DNA dataset..."})
            
            # Config
            cfg_dict = json.loads(config_str)
            cfg = TrainConfig(**cfg_dict)
            
            q.put({"type": "log", "msg": "🧬 Starting DNA Pipeline..."})
            raw = content_bytes.decode("utf-8-sig")
            df = pd.read_csv(io.StringIO(raw))
            q.put({"type": "log", "msg": "🧹 Cleaning DNA sequences..."})
            sequences = clean_sequences(df, cfg.sequence_col, cfg.min_seq_len)
            
            # Clear memory immediately
            del raw
            del df
            gc.collect()
            
            q.put({"type": "log", "msg": f"[V5] Data cleaned. Building sliding-window dataset..."})
            
            # LIMIT for Free Tier safety
            MAX_SAMPLES = 500 
            X_oh, X_int, y = make_windows(sequences, cfg.window_size, cfg.step, MAX_SAMPLES)
            
            q.put({"type": "log", "msg": f"🤖 Modeling Building... ({len(X_oh)} samples identified)"})

            model = build_model(cfg) 

            q.put({"type": "log", "msg": "[V5] Model live. Training started..."})
            model.fit(
                [X_oh, X_int], y,
                validation_split=0.1, # Small validation set for stats
                batch_size=32,
                epochs=cfg.epochs,
                callbacks=[StreamCallback(q, cfg.epochs)],
                verbose=0,
            )

            store.model     = model
            store.config    = cfg.dict()
            store.sequences = sequences
            store.trained   = True
            q.put({"type": "log", "msg": "[V5] SUCCESS: Model cached."})
            q.put({"type": "done"})

        except Exception as e:
            q.put({"type": "error", "msg": f"V5 Engine Crash: {str(e)}"})

    # Start the thread
    thread = threading.Thread(target=run_training_v5, args=(content, config), daemon=True)
    thread.start()

    def event_stream():
        # First push to flush connection
        yield "data: {\"type\": \"log\", \"msg\": \"🚀 Connection Established. Backend Warming Up...\"}\n\n"
        
        while True:
            try:
                # Use a timeout to allow heartbeat if queue is empty
                item = q.get(timeout=15) 
                yield f"data: {json.dumps(item)}\n\n"
                if item["type"] in ("done", "error"):
                    break
            except Exception:
                # Heartbeat to keep Railway connection alive during slow TF init
                yield ": heartbeat\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/generate")
def generate(req: GenerateRequest):
    """Generate sequences from the cached trained model."""
    if not store.trained or store.model is None:
        raise HTTPException(status_code=400, detail="No trained model found. Please train first.")

    window = store.config["window_size"]
    char2idx = store.char2idx
    idx2char = store.idx2char

    # Choose seed
    if req.seed and all(c in "ACGT" for c in req.seed.upper()):
        seed_str = req.seed.upper()
    elif store.sequences:
        src = random.choice(store.sequences)
        start = random.randint(0, max(0, len(src) - window))
        seed_str = src[start: start + window]
    else:
        seed_str = "ATGCCCCAACTAAATACT"

    seed_oh, seed_int = encode_dna(seed_str)
    if seed_oh.shape[0] < 5:
        raise HTTPException(status_code=400, detail="Seed too short or invalid characters.")

    results = []
    for i in range(req.num_sequences):
        seq = generate_one(store.model, seed_oh, seed_int, req.gen_length, req.temperature)
        gc = round((seq.count("G") + seq.count("C")) / max(len(seq), 1) * 100, 1)
        results.append({"id": i + 1, "sequence": seq, "length": len(seq), "gc_percent": gc})

    return {"sequences": results, "seed_used": seed_str, "model_config": store.config}


@app.delete("/model")
def clear_model():
    """Clear the cached model to free memory."""
    store.model = None
    store.trained = False
    store.sequences = []
    return {"cleared": True}
