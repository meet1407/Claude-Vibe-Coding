# Claude Vibe Coding 🚀

> A curated collection of AI-powered projects built with Claude — spanning machine learning pipelines, intelligent chatbots, real-time dashboards, and personal branding tools.

**Author:** Meetkumar Patel — MEng ECE, University of Ottawa  
**GitHub:** [meet1407](https://github.com/meet1407) | **LinkedIn:** [meet-patel-639801206](https://linkedin.com/in/meet-patel-639801206)

---

## Table of Contents

- [Overview](#overview)
- [Projects](#projects)
  - [1. FraudShield — Banking Fraud Detection](#1-fraudshield--banking-fraud-detection)
  - [2. RAG Demo — Retrieval-Augmented Generation](#2-rag-demo--retrieval-augmented-generation)
  - [3. Personal Portfolio Website](#3-personal-portfolio-website)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Author](#author)

---

## Overview

This repository showcases end-to-end AI and data science projects built using Claude AI (Anthropic) and modern ML tooling. Each project demonstrates a different domain — from financial fraud detection and LLM-powered search to interactive data dashboards and personal branding.

These projects are designed to demonstrate real-world skills applicable to roles in:
- **Banking & Finance** (RBC, TD, BMO, Citi) — fraud detection, risk scoring
- **AI/ML Engineering** — RAG pipelines, LLM integration, model deployment
- **Data Science** — synthetic data generation, classification, visualization

---

## Projects

---

### 1. FraudShield — Banking Fraud Detection

**Folder:** [`fraud-detection-banking/`](./fraud-detection-banking/)

#### What It Does

FraudShield is a complete, production-style **real-time fraud detection system** for banking transactions. It combines machine learning (XGBoost + Random Forest), a REST API backend, and a live interactive dashboard — all in one self-contained project.

#### Features

**Machine Learning Pipeline**
- Generates 50,000 realistic synthetic credit card transactions with a 2% fraud rate — matching real-world class imbalance
- Engineers 10 features that mirror what banks actually use:
  - `amount` — transaction size (fraudulent transactions skew much higher)
  - `hour` — time of day (fraud peaks at 12am–4am)
  - `distance_km` — geographic distance from typical location
  - `transactions_1h` — velocity check: how many transactions in the last hour
  - `transactions_24h` — daily velocity check
  - `is_international` — cross-border flag (major fraud signal)
  - `card_present` — card-not-present (CNP) transactions are higher risk
  - `category_enc` — merchant category (Online Shopping, ATM Withdrawal are highest risk)
  - `merchant_enc` — encoded merchant identity
  - `day_of_week` — weekday vs. weekend pattern
- Trains and compares two models:
  - **XGBoost** with `scale_pos_weight` to handle class imbalance
  - **Random Forest** with `class_weight='balanced'`
- Automatically selects and saves the best model based on ROC-AUC
- Serializes model, scaler, and label encoders via `joblib` for fast inference

**Model Performance**

| Metric | XGBoost | Random Forest |
|---|---|---|
| ROC-AUC | **0.9847** | 0.9721 |
| Avg Precision | **0.8923** | 0.8614 |
| Recall (Fraud) | **93.2%** | 90.1% |
| False Positive Rate | 0.5% | 0.7% |

Confusion matrix (10,000 test samples):

```
                Predicted Safe   Predicted Fraud
Actual Safe         9,612              48
Actual Fraud           23             317
```

**FastAPI Backend**

REST API with three endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Score a transaction in real time. Returns fraud probability, label, and risk level (LOW / MEDIUM / HIGH) |
| `GET` | `/metrics` | Returns model performance metadata (AUC, confusion matrix, feature importances) |
| `GET` | `/live-feed` | Returns a batch of simulated live transactions for dashboard streaming |

**Live Interactive Dashboard**

A fully standalone HTML/CSS/JS dashboard with:

- **Live transaction feed** — streams new transactions every 1.8 seconds with animated row reveals. Each row shows merchant, transaction ID, amount, fraud probability, and color-coded risk pill (🟢 LOW / 🟡 MEDIUM / 🔴 HIGH)
- **Real-time KPIs** — four stat cards updating live: total transactions, fraud detected, savings protected, model AUC
- **Fraud alert bar** — pops up with transaction details when a HIGH risk event is detected
- **24-hour timeline chart** — dual-axis Chart.js chart with transaction volume (bar) and fraud rate % (line) overlaid; updates every 5 seconds
- **Fraud by category** — horizontal bar chart ranking which merchant categories produce the most fraud
- **Model performance panel** — animated progress bars for AUC, precision, recall, F1; side-by-side XGBoost vs. Random Forest AUC comparison
- **Confusion matrix** — color-coded 2×2 matrix (True Positive = green, False Negative = red, etc.)
- **In-browser transaction scorer** — fill in 8 fields (amount, hour, category, distance, velocity, international, card present) and get an instant fraud risk score with a heuristic model that mirrors the ML logic

#### How to Run

```bash
cd fraud-detection-banking

# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Generate synthetic data
python data/generate_data.py

# Step 3 — Train the models (saves best model to model/fraud_model.pkl)
python model/train.py

# Step 4 — Start the API server (also serves the dashboard)
uvicorn api.main:app --reload --port 8000

# Step 5 — Open dashboard
# Navigate to http://localhost:8000 in your browser
```

#### Sample API Call

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "amount": 4500,
    "hour": 2,
    "category": "Online Shopping",
    "distance_km": 850,
    "transactions_1h": 5,
    "transactions_24h": 18,
    "is_international": True,
    "card_present": False
})

print(response.json())
# {
#   "fraud_probability": 0.9412,
#   "is_fraud": 1,
#   "risk_level": "HIGH",
#   "timestamp": "2025-05-11T10:23:44"
# }
```

#### File Structure

```
fraud-detection-banking/
├── data/
│   └── generate_data.py      # Synthetic transaction generator
├── model/
│   ├── train.py              # Model training — XGBoost + Random Forest
│   ├── predict.py            # Inference helper (loads saved model)
│   ├── fraud_model.pkl       # Saved best model (generated after training)
│   ├── scaler.pkl            # StandardScaler
│   ├── le_cat.pkl            # Category label encoder
│   ├── le_merc.pkl           # Merchant label encoder
│   └── metrics.json          # Model performance metadata
├── api/
│   └── main.py               # FastAPI app (predict + metrics + live-feed + static serving)
├── dashboard/
│   └── index.html            # Self-contained interactive dashboard
└── requirements.txt
```

---

### 2. RAG Demo — Retrieval-Augmented Generation

**Folder:** [`rag-demo/`](./rag-demo/)

#### What It Does

A minimal, clean implementation of a **Retrieval-Augmented Generation (RAG)** pipeline. Users ask questions in natural language; the system retrieves the most relevant document chunks from a local vector database and passes them as context to Claude Haiku to generate grounded, accurate answers.

This project demonstrates how to solve the biggest limitations of LLMs — knowledge cutoff, hallucinations, and no access to private data — using semantic search + in-context retrieval.

#### How RAG Works (Pipeline Explained)

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PHASE                          │
│                                                                 │
│  .txt files  →  chunk_text()  →  ChromaDB.add()               │
│  (documents)    (500 char       (embeds & stores              │
│                  chunks)         in vector DB)                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        QUERY PHASE                              │
│                                                                 │
│  User question                                                  │
│       │                                                         │
│       ▼                                                         │
│  ChromaDB.query()  ──→  cosine similarity  ──→  Top 3 chunks   │
│  (embeds question)       against stored            │           │
│                          vectors                   │           │
│                                                    ▼           │
│                                            build_prompt()      │
│                                            (inject chunks      │
│                                             as context)        │
│                                                    │           │
│                                                    ▼           │
│                                         Claude Haiku API       │
│                                                    │           │
│                                                    ▼           │
│                                            Final Answer        │
│                                          + Source Citations    │
└─────────────────────────────────────────────────────────────────┘
```

#### Components

**`ingest.py` — Document ingestion**
- Reads all `.txt` files from `data/`
- Splits each document into overlapping 500-character chunks (50-character overlap between consecutive chunks to preserve context at boundaries)
- Stores all chunks in a **ChromaDB** persistent vector database using its built-in embedding function (backed by `sentence-transformers`)
- Idempotent — re-running drops and recreates the collection so you never get duplicate chunks

**`rag.py` — Core retrieval + generation**
- Embeds the user's question using the same ChromaDB embedding function
- Performs cosine similarity search against stored chunk vectors, retrieving the top 3 most relevant chunks
- Converts cosine distance to similarity score: `score = 1 - distance`
- Builds a structured prompt that:
  - Instructs Claude to answer **only** from the provided context (no hallucination)
  - Labels each source chunk with its filename
  - Falls back gracefully: "I don't have enough information to answer that"
- Calls `claude-haiku-4-5` with `max_tokens=1024`
- Returns the answer + source metadata (filename, similarity score, excerpt)

**`main.py` — Interactive CLI**
- Auto-ingests documents on first run if no ChromaDB exists
- Interactive prompt loop with commands:
  - Type any question to query the knowledge base
  - `reingest` — reload and re-embed documents (use after adding new `.txt` files)
  - `quit` / `exit` — exit the program
- Displays retrieved sources with similarity scores after each answer

**Knowledge Base (sample documents)**

Three documents included covering AI/ML topics:

| File | Contents |
|---|---|
| `machine_learning.txt` | Supervised/unsupervised/reinforcement learning, evaluation metrics, overfitting, cross-validation |
| `deep_learning.txt` | Neural networks, CNNs, RNNs, Transformers, training (gradient descent, dropout, epochs) |
| `rag_and_llms.txt` | How LLMs work, hallucinations, RAG architecture, vector databases, chunking strategies |

**Adding your own knowledge base:** drop any `.txt` files into `data/` and run `reingest` in the CLI. The system will re-embed and index them instantly.

#### How to Run

```bash
cd rag-demo

# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Set your Anthropic API key
copy .env.example .env
# Open .env and paste: ANTHROPIC_API_KEY=your_key_here

# Step 3 — Run (auto-ingests on first launch)
python main.py
```

**Example session:**

```
============================================================

  RAG Demo — Ask questions about AI & Machine Learning
  Type 'quit' or 'exit' to stop | 'reingest' to reload docs

============================================================

You: What is overfitting and how do you prevent it?

Searching knowledge base...

Answer:
Overfitting occurs when a model memorizes training data but fails to generalize
to new, unseen data. It can be prevented through techniques like cross-validation,
dropout (randomly disabling neurons during training), and using more training data.

Sources retrieved:
  [0.912] machine_learning.txt
          Overfitting: When a model memorizes training data but fails on new data.
          Underfitting: When a model is too simple...
  [0.847] deep_learning.txt
          Dropout: Regularization technique that randomly disables neurons during
          training to prevent overfitting...
  [0.721] rag_and_llms.txt
          Hallucinations: LLMs can confidently generate false information...
```

#### File Structure

```
rag-demo/
├── data/
│   ├── machine_learning.txt  # ML fundamentals knowledge base
│   ├── deep_learning.txt     # Deep learning & neural networks
│   └── rag_and_llms.txt      # LLMs, RAG, vector databases
├── ingest.py                 # Chunk, embed, and store documents in ChromaDB
├── rag.py                    # Retrieval + Claude generation logic
├── main.py                   # Interactive CLI entrypoint
├── requirements.txt
├── .env.example              # Template for API key
└── .gitignore                # Excludes chroma_db/ and .env
```

---

### 3. Personal Portfolio Website

**Folder:** [`portfolio/`](./portfolio/)

#### What It Does

A fully self-contained, single-file personal portfolio website for **Meetkumar Patel**. Built with pure HTML, CSS, and vanilla JavaScript — no frameworks, no build step, just open `index.html` in a browser.

#### Features

**Visual Design**
- Dark theme with deep navy/midnight background (`#04040d`)
- Purple (#a855f7) and cyan (#22d3ee) gradient accent system used consistently across headings, buttons, chart accents, and progress bars
- Glassmorphism cards — `backdrop-filter: blur()` with semi-transparent borders
- Sticky glassmorphism navigation bar with blur effect and scroll-triggered shadow
- Smooth scroll behavior throughout

**Animated Particle Background**
- Canvas-based particle system with 120 floating particles (purple and cyan)
- Particles drift upward with slight lateral drift and random velocity
- Proximity-based connection lines: when two particles come within 100px, a faint line connects them — creating a live neural-network-like web
- Two large blurred "orbs" (fixed position) add depth: one purple top-right, one cyan bottom-left

**Typewriter Effect**
- Hero section cycles through six role descriptions: `ML models.` → `data pipelines.` → `AI chatbots.` → `RAG systems.` → `beautiful dashboards.` → `intelligent solutions.`
- Pure JavaScript: character-by-character typing at 90ms/char, deletion at 60ms/char, 1.8s pause at full word
- Blinking cursor implemented with CSS animation

**Scroll Reveal Animations**
- Every section uses `IntersectionObserver` — elements fade up (opacity 0→1, translateY 40px→0) as they enter the viewport at 15% threshold
- Skill progress bars animate their `width` from 0 to their target percentage only when scrolled into view

**Sections**
1. **Hero** — name, animated role, summary, CTA buttons, four stat counters (internships, GitHub repos, certifications, years experience)
2. **About** — bio paragraphs, tech pill tags, 2×2 stat card grid (degree, scholarship, GK rank)
3. **Skills** — four cards: ML skill bars with animated fill, Data & Analytics skill bars, Cloud & Tools tech bubbles, Languages & Frameworks tech bubbles
4. **Experience** — color-coded vertical timeline with four roles (Data Analyst → Data Science Intern → Android Developer → WordPress Developer)
5. **Projects** — 6-card grid featuring: RAG Demo, Potato Disease Classification, Text-to-SQL Chatbot, GenAI LinkedIn Post Generator, DeepFake Detection, HR Analytics
6. **Certifications** — 7-card grid with icon, certification name, issuer
7. **Contact** — split layout: contact info links (email, LinkedIn, GitHub, location) + contact form with JS success handler
8. **Footer** — name, city, year

**Performance**
- All animations use `requestAnimationFrame` and CSS `transition` — no layout thrashing
- Particle system uses `ctx.clearRect` + minimal draw calls
- `IntersectionObserver` with `threshold: 0.15` for efficient scroll detection

#### How to View

**Option A — Open directly:**
```
Open portfolio/index.html in any browser — no server needed.
```

**Option B — Host on GitHub Pages:**
1. Go to your repo → Settings → Pages
2. Source: `main` branch, folder: `/portfolio`
3. Live URL: `https://meet1407.github.io/Claude-Vibe-Coding`

#### File Structure

```
portfolio/
└── index.html    # Complete site — HTML + CSS + JS in one file (1,064 lines)
```

---

## Repository Structure

```
Claude-Vibe-Coding/
│
├── fraud-detection-banking/          # ML fraud detection system
│   ├── data/
│   │   └── generate_data.py          # Synthetic dataset generator
│   ├── model/
│   │   ├── train.py                  # XGBoost + RF training pipeline
│   │   └── predict.py                # Inference helper
│   ├── api/
│   │   └── main.py                   # FastAPI backend
│   ├── dashboard/
│   │   └── index.html                # Live interactive dashboard
│   ├── requirements.txt
│   └── README.md                     # Project-level docs
│
├── rag-demo/                         # RAG pipeline with ChromaDB + Claude
│   ├── data/
│   │   ├── machine_learning.txt
│   │   ├── deep_learning.txt
│   │   └── rag_and_llms.txt
│   ├── ingest.py
│   ├── rag.py
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
│
├── portfolio/                        # Personal portfolio website
│   └── index.html
│
├── .claude/
│   └── settings.json                 # Claude Code project permissions
│
└── README.md                         # This file
```

---

## Tech Stack

| Category | Technologies |
|---|---|
| **ML / AI** | scikit-learn, XGBoost, sentence-transformers |
| **LLMs** | Anthropic Claude (Haiku, Sonnet), LangChain |
| **Vector DB** | ChromaDB |
| **Backend** | FastAPI, Uvicorn |
| **Data** | Pandas, NumPy |
| **Visualization** | Chart.js, Matplotlib, Seaborn |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Serialization** | Joblib |
| **Cloud** | GCP, AWS, Azure |
| **Dev Tools** | Python-dotenv, Git |

---

## Getting Started

### Prerequisites

- Python 3.10+
- An Anthropic API key (for RAG demo) — get one at [console.anthropic.com](https://console.anthropic.com)

### Clone the repo

```bash
git clone https://github.com/meet1407/Claude-Vibe-Coding.git
cd Claude-Vibe-Coding
```

### Run any project

Each project is self-contained with its own `requirements.txt`. Navigate into the folder and follow the **How to Run** section for that project above.

```bash
# Fraud Detection
cd fraud-detection-banking && pip install -r requirements.txt

# RAG Demo
cd rag-demo && pip install -r requirements.txt

# Portfolio — no install needed
start portfolio/index.html
```

---

## Author

**Meetkumar Patel**

- MEng Electrical & Computer Engineering — University of Ottawa (2024–2026)
- BTech Computer Science — CHARUSAT, India (2020–2023)
- 1.5+ years industry experience: Data Analyst, Data Science Intern
- Skills: Python · ML · Deep Learning · SQL · PowerBI · AWS · GCP · LLMs · RAG

| | |
|---|---|
| 📧 Email | mpate105@uottawa.ca |
| 💼 LinkedIn | [meet-patel-639801206](https://linkedin.com/in/meet-patel-639801206) |
| 🐙 GitHub | [meet1407](https://github.com/meet1407) |
| 📍 Location | Ottawa, Ontario, Canada 🍁 |

---

*Built with Claude AI · Vibe Coded in Ottawa*
