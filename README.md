<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-CPU-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">
  <br/>
  <img width="80" src="https://emojicdn.elk.sh/🧬" alt="logo"/>
  <br/>
  <strong>DNA Sequence Generator</strong>
  <br/>
  <sub>BiLSTM + Multi-Head Attention</sub>
</h1>

<p align="center">
  A full-stack deep learning application that empowers researchers to upload DNA datasets, train a custom 3-Layer Bidirectional LSTM neural network with Multi-Head Attention, and generate novel DNA sequences from learned biological patterns.
</p>

<p align="center">
  <a href="https://dna-psi-smoky.vercel.app/" target="_blank">
    <img src="https://img.shields.io/badge/Live_Demo-Frontend-000?style=flat-square&logo=vercel&logoColor=white" alt="Frontend Demo"/>
  </a>
  <a href="https://dna-tezh.onrender.com" target="_blank">
    <img src="https://img.shields.io/badge/Live_Demo-Backend-000?style=flat-square&logo=render&logoColor=white" alt="Backend Demo"/>
  </a>
  <a href="https://github.com/Nitinmall-1390/DNA" target="_blank">
    <img src="https://img.shields.io/badge/Source_Code-GitHub-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-v2-blueviolet?style=flat-square" alt="Architecture v2"/>
  <img src="https://img.shields.io/badge/Model-BiLSTM%20%2B%20Attention-critical?style=flat-square" alt="Model"/>
  <img src="https://img.shields.io/badge/Real_Time_Training-SSE%20Streaming-ff69b4?style=flat-square" alt="SSE Streaming"/>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Model Architecture](#model-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Locally](#running-locally)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Performance & Optimization](#performance--optimization)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

---

## Overview

DNA Sequence Generator is an end-to-end deep learning platform designed for computational biology research. It combines a sophisticated neural network architecture with an intuitive web interface, enabling researchers to explore de novo DNA sequence synthesis without writing a single line of machine learning code.

The application accepts CSV-formatted DNA datasets, preprocesses them into fixed-length tokenized sequences, and trains a deep Bidirectional LSTM model augmented with Multi-Head Attention. Once trained, the model can generate entirely novel DNA sequences whose statistical properties mirror those of the training data, providing a powerful tool for sequence design, augmentation, and exploration.

The entire training process is streamed in real time to the browser via Server-Sent Events, giving researchers full visibility into loss curves, accuracy trends, and validation metrics as they develop.

---

## Key Features

| Feature | Description |
|:--------|:------------|
| **Real-Time Training Stream** | Epoch-by-epoch progress (Loss, Accuracy, Validation stats) streamed directly to the browser via SSE, with live loss charts rendered on Canvas. |
| **Advanced Neural Architecture** | 3-Layer Bidirectional LSTM with Multi-Head Attention, Sinusoidal Positional Encoding, and Integer Embedding for deep sequence understanding. |
| **Dynamic Sequence Generation** | Adjustable temperature parameter controls the trade-off between conservative (low temperature) and diverse (high temperature) sequence outputs. |
| **Futuristic Dark-Mode UI** | High-end responsive interface with DNA helix Canvas animations, glassmorphism effects, and vibrant neon aesthetics. |
| **Memory-Optimized Backend** | Engineered to run on constrained environments (Render Free Tier / 512 MB RAM) using `tensorflow-cpu` and proactive garbage collection. |
| **CSV Upload & Parsing** | Automatic detection and cleaning of DNA sequence columns from uploaded CSV files with minimal user configuration. |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLIENT (React)                          │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌────────────┐  │
│  │  Upload   │  │  Training    │  │  Generate │  │  Live      │  │
│  │  CSV Panel│  │  Config Panel│  │  Panel    │  │  Charts    │  │
│  └─────┬─────┘  └──────┬───────┘  └─────┬─────┘  └─────┬──────┘  │
│        │               │                │               │         │
│        ▼               ▼                ▼               ▼         │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              REST API + SSE Client Layer                 │    │
│  └─────────────────────────┬────────────────────────────────┘    │
└────────────────────────────┼─────────────────────────────────────┘
                             │  HTTP / SSE
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      SERVER (FastAPI)                            │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌────────────┐  │
│  │  Data     │  │  Model       │  │  Training │  │  Sequence  │  │
│  │  Pipeline │  │  Builder     │  │  Engine   │  │  Generator │  │
│  └─────┬─────┘  └──────┬───────┘  └─────┬─────┘  └─────┬──────┘  │
│        │               │                │               │         │
│        ▼               ▼                ▼               ▼         │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │         TensorFlow / Keras BiLSTM + Attention            │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Model Architecture

The sequence generation model is built on a carefully designed pipeline that captures both local motifs and long-range dependencies in DNA sequences:

```
Input (integer-encoded nucleotides)
        │
        ▼
┌─────────────────────┐
│ Integer Embedding   │   Projects discrete tokens into dense vectors
│ (d_model = 128)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Sinusoidal Positional│  Encodes position information without learned
│ Encoding            │  parameters, allowing generalization to
└─────────┬───────────┘  variable-length contexts
          │
          ▼
┌─────────────────────┐
│ BiLSTM Layer 1      │   128 units, bidirectional
│ (+ Dropout 0.2)     │   Captures forward and backward context
├─────────────────────┤
│ BiLSTM Layer 2      │   128 units, bidirectional
│ (+ Dropout 0.2)     │   Learns higher-order sequential patterns
├─────────────────────┤
│ BiLSTM Layer 3      │   64 units, bidirectional
│ (+ Dropout 0.2)     │   Refines temporal representations
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Multi-Head Attention│   4 heads, key_dim = 32
│                     │   Captures long-range dependencies across
│                     │   the entire sequence simultaneously
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Dense (Softmax)     │   Outputs probability distribution over
│ (vocab_size = 5)    │   {A, C, G, T, <pad>} for next-token
└─────────────────────┘   prediction
```

**Key Design Decisions:**

- **Bidirectional processing** enables the model to learn context from both 5'→3' and 3'→5' directions, which is essential for understanding DNA regulatory motifs that function in either orientation.
- **Three stacked LSTM layers** provide a hierarchical representation, where lower layers capture local dinucleotide/trinucleotide patterns and upper layers model larger structural motifs.
- **Multi-Head Attention** allows the model to attend to multiple distant positions simultaneously, overcoming the limited receptive field of recurrent layers for very long sequences.

---

## Technology Stack

### Backend

| Technology | Purpose |
|:-----------|:--------|
| ![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white) | Core runtime language |
| ![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat&logo=fastapi&logoColor=white) | High-performance async web framework |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-CPU-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Deep learning engine (CPU-optimized) |
| ![Keras](https://img.shields.io/badge/Keras-API-D00000?style=flat&logo=keras&logoColor=white) | High-level model building API |
| ![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat&logo=pandas&logoColor=white) | CSV parsing and data cleaning |
| ![NumPy](https://img.shields.io/badge/NumPy-Compute-013243?style=flat&logo=numpy&logoColor=white) | Numerical operations and array processing |
| ![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-336791?style=flat) | Production ASGI server |
| **SSE (Server-Sent Events)** | Real-time training progress streaming |

### Frontend

| Technology | Purpose |
|:-----------|:--------|
| ![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black) | Component-based UI framework |
| **Canvas API** | Custom DNA helix animations and live loss/accuracy charts |
| **CSS3** | Glassmorphism, neon gradients, and responsive dark-mode design |

---

## Project Structure

```
DNA/
├── backend/
│   ├── main.py              # FastAPI application, model architecture,
│   │                         # training loop, and sequence generation
│   ├── requirements.txt     # Python dependency specifications
│   └── runtime.txt          # Python version pin for Render deployment
├── src/
│   ├── App.js               # Main React component: upload, train, generate UI
│   └── index.js             # React DOM mounting point
├── public/                   # Static assets (favicon, icons, etc.)
├── package.json              # Node.js dependency manifest
├── vercel.json               # Vercel SPA routing and deployment config
└── README.md                 # Project documentation
```

---

## Getting Started

### Prerequisites

Ensure you have the following installed on your system:

| Requirement | Minimum Version | Recommended |
|:------------|:----------------|:------------|
| Python | 3.10+ | 3.11 |
| Node.js | 16+ | 18 LTS |
| npm | 8+ | 9+ |
| pip | 22+ | Latest |

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Nitinmall-1390/DNA.git
cd DNA
```

**2. Set up the backend**

```bash
cd backend
python -m venv venv

# Activate the virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

**3. Set up the frontend**

```bash
# Return to the project root
cd ..
npm install
```

### Running Locally

Start both the backend and frontend servers. They must run concurrently for the application to function.

**Terminal 1 — Backend:**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Terminal 2 — Frontend:**

```bash
# In the project root (DNA/)
npm start
```

The application will open at `http://localhost:3000`.

> **Note:** If your backend runs on a different port, update the API base URL in `src/App.js` accordingly.

---

## Usage Guide

### 1. Upload a Dataset

Prepare a CSV file containing DNA sequences. The system will automatically detect and parse nucleotide columns. Example format:

| sequence |
|:---------|
| ATCGATCGATCG |
| GCTAGCTAGCTA |
| TTAGCTTAGCTT |

### 2. Configure Training Parameters

| Parameter | Description | Default |
|:----------|:------------|:--------|
| Epochs | Number of training passes | 50 |
| Batch Size | Samples per gradient update | 32 |
| Learning Rate | Optimizer step size | 0.001 |
| Sequence Length | Token window size | 100 |

### 3. Monitor Training in Real Time

During training, the dashboard displays live metrics:

- **Training Loss** — Cross-entropy loss on training batches (rendered as a live chart)
- **Training Accuracy** — Token-level prediction accuracy
- **Validation Loss & Accuracy** — Evaluated on a held-out split at each epoch

### 4. Generate Novel Sequences

After training, adjust the **Temperature** slider:

| Temperature | Behavior |
|:------------|:---------|
| 0.1 – 0.5 | Conservative; outputs closely resemble training data |
| 0.5 – 1.0 | Balanced; introduces moderate diversity |
| 1.0 – 2.0 | Creative; generates highly diverse, exploratory sequences |

Click **Generate** to synthesize novel DNA sequences based on the learned model.

---

## API Reference

The backend exposes the following REST endpoints:

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/upload` | Upload and parse a CSV dataset |
| `POST` | `/train` | Start model training (streams progress via SSE) |
| `POST` | `/generate` | Generate novel DNA sequences |
| `GET` | `/health` | Service health check |

<details>
<summary><strong>Expand: Streaming Endpoint Details</strong></summary>

The `/train` endpoint uses Server-Sent Events to stream training progress. Each event payload includes:

```json
{
  "epoch": 1,
  "loss": 1.2456,
  "accuracy": 0.4823,
  "val_loss": 1.3102,
  "val_accuracy": 0.4491
}
```

Clients should open an `EventSource` connection to receive real-time updates.
</details>

---

## Deployment

### Backend — Render

| Setting | Value |
|:--------|:------|
| Runtime | Python 3.11 (specified in `runtime.txt`) |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Instance Type | Free (512 MB RAM) |

> The backend uses `tensorflow-cpu` and explicit garbage collection (`gc.collect()`) to remain within the free tier's memory constraints.

### Frontend — Vercel

| Setting | Value |
|:--------|:------|
| Framework | React (Create React App) |
| Build Command | `npm run build` |
| Output Directory | `build` |
| Configuration | `vercel.json` (SPA routing with rewrites) |

---

## Performance & Optimization

Running a TensorFlow model on a constrained environment (512 MB RAM) requires careful optimization. The following strategies are employed:

1. **CPU-only TensorFlow** — Eliminates GPU overhead and significantly reduces the package footprint.
2. **Proactive Garbage Collection** — `gc.collect()` is called between major training phases to reclaim unused memory.
3. **Efficient Data Pipeline** — Datasets are tokenized and batched using `tf.data.Dataset` for memory-efficient input streaming.
4. **Layer-wise Dropout** — Regularization via dropout (rate 0.2) after each BiLSTM layer prevents overfitting and reduces model complexity.
5. **Compact Model Dimensions** — Embedding size (128), LSTM units (128/128/64), and attention heads (4) are tuned to balance capacity with resource constraints.

---

## Contributing

Contributions are welcome! To contribute to this project:

1. **Fork** the repository.
2. Create a **feature branch** (`git checkout -b feature/your-feature-name`).
3. **Commit** your changes with descriptive messages.
4. **Push** to your branch (`git push origin feature/your-feature-name`).
5. Open a **Pull Request** with a clear description of your changes.

For substantial architectural changes or new features, please open an issue first to discuss the proposed approach.

---

## Author

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Nitinmall-1390">
        <img src="https://github.com/Nitinmall-1390.png" width="100" style="border-radius: 50%;" alt="Nitin Mall"/>
        <br/>
        <strong>Nitin Mall</strong>
      </a>
    </td>
  </tr>
</table>

<p align="center">
  <a href="https://github.com/Nitinmall-1390">
    <img src="https://img.shields.io/badge/GitHub-@Nitinmall--1390-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
  </a>
  <a href="https://dna-psi-smoky.vercel.app/">
    <img src="https://img.shields.io/badge/Live_App-DNA_Lab_Interface-4CAF50?style=for-the-badge&logo=vercel&logoColor=white" alt="Live App"/>
  </a>
</p>

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with TensorFlow, FastAPI, and React. Designed for computational biology research.</sub>
</p>
