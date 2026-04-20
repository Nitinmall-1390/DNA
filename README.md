# 🧬 DNA Sequence Generator (BiLSTM + Attention)

Developed by **Nitin Mall**

A full-stack deep learning application that allows researchers and developers to upload DNA datasets (CSV), train a custom **3-Layer Bidirectional LSTM** neural network with **Multi-Head Attention**, and generate novel DNA sequences based on the learned patterns.

---

## 🚀 Live Demo
- **Frontend (Vercel):** [https://dna-psi-smoky.vercel.app/](https://dna-psi-smoky.vercel.app/)
- **Backend (Render):** [https://dna-tezh.onrender.com](https://dna-tezh.onrender.com)

---

## ✨ Features
- **Real-time Training:** Stream epoch-by-epoch progress (Loss, Accuracy, Validation stats) directly to the browser.
- **Advanced Architecture (v2):** 
  - 3-Layer Bidirectional LSTM for deep sequence understanding.
  - Multi-Head Attention mechanism to capture long-range dependencies.
  - Sinusoidal Positional Encoding & Integer Embedding.
- **Dynamic Generation:** Adjust "Temperature" to control creativity (conservative vs. diverse sequences).
- **Responsive Web UI:** A high-end, futuristic dark-mode interface with DNA helix animations and real-time training logs.
- **Memory Optimized:** Specifically tuned to run on limited resources (Render Free Tier) using `tensorflow-cpu` and garbage collection.

---

## 🛠️ Technology Stack
### Backend
- **FastAPI:** High-performance Python web framework.
- **TensorFlow/Keras:** Deep learning engine for training and sequence generation.
- **Pandas/NumPy:** Data cleaning and numerical processing.
- **SSE (Server-Sent Events):** For real-time progress streaming.

### Frontend
- **React.js:** Modern UI framework.
- **Canvas API:** Custom helix animations and live loss charts.
- **CSS3:** Glassmorphism and vibrant neon aesthetics.

---

## 📦 Project Structure
```text
dna-lstm/
├── backend/
│   ├── main.py            # FastAPI Application & Model Logic
│   ├── requirements.txt   # Python dependencies
│   └── runtime.txt        # Specified Python version for Render
├── src/
│   ├── App.js             # Main React UI component
│   └── index.js           # React mounting point
├── public/                # Static assets
└── vercel.json            # Deployment config for Vercel
```

---

## 🚦 Getting Started (Local)

### 1. Clone the repository
```bash
git clone https://github.com/Nitinmall-1390/DNA.git
cd DNA
```

### 2. Run the Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Run the Frontend
```bash
# In the root DNA directory
npm install
npm start
```
The app will be available at `http://localhost:3000`.

---

## 📜 Deployment Details
- **Backend:** Hosted on **Render** (Python 3.11). Configured with `runtime.txt` and `tensorflow-cpu` to fit within 512MB RAM.
- **Frontend:** Hosted on **Vercel**. Configured with `vercel.json` for SPA routing.

---

## 🤝 Contributing
Feel free to fork this project and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

## 👤 Author
**Nitin Mall**
- GitHub: [@Nitinmall-1390](https://github.com/Nitinmall-1390)
- DNA Lab Interface: [Live App](https://dna-psi-smoky.vercel.app/)
