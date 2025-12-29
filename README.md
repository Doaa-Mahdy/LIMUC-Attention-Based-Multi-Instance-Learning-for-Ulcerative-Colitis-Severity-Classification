# 🏥 LIMUC-AI-System: Advanced Ulcerative Colitis Assessment
### Multi-Instance Learning & Explainable AI for Endoscopic Severity Scoring

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/Streamlit-Frontend-red?logo=streamlit)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

## 📋 Overview
**LIMUC-AI-System** is a sophisticated deep learning system designed to assist endoscopists in the objective assessment of **Ulcerative Colitis (UC)** using endoscopic imagery. 

Unlike traditional black-box models, this system provides transparent, patient-level diagnosis using **Multiple Instance Learning (MIL)** and offers granular severity tracking through a novel "Gray Zone" regression module. The system transitions from image-level understanding to patient-level aggregation and finally to continuous severity regression.

### The System Addresses 3 Critical Tasks:
1.  **Single Frame Classification:** Instant Mayo Score (0-3) for individual frames.
2.  **Patient-Level Diagnosis (MIL):** Aggregating multiple images from one patient to form a holistic diagnosis, intelligently prioritizing informative frames (ulcers/bleeding) while ignoring noise.
3.  **Continuous Severity Regression:** A precision score (e.g., 1.7) to track subtle disease progression (Improving vs. Worsening) often missed by integer scoring.

---

## 🚀 Key Features
* **Full-Stack Architecture:** Decoupled **FastAPI** backend for high-performance inference and **Streamlit** frontend for an interactive clinical dashboard.
* **Explainable AI (XAI):** Integrated **Grad-CAM** heatmaps that visualize exactly *where* the model sees inflammation, building clinician trust.
* **Multi-Modal Analysis:** Handles both single images and "bags" of patient images dynamically.
* **Dockerized Deployment:** Cloud-native architecture ready for Hugging Face Spaces.

---

## 🧠 Technical Architecture & Methodology

This system is the result of a multi-phase deep learning study utilizing a **ConvNeXt-Tiny** backbone.

### 1. Dataset & Challenges
The dataset consists of **11,276 endoscopic images**. The system was designed to overcome:
* **Class Imbalance:** Significant variance in the number of images per Mayo class.
* **Inter-observer Variability:** Label noise where adjacent Mayo scores (e.g., 1 vs 2) look similar.
* **Variable Patient Data:** Patients have a variable number of images, requiring an adaptive aggregation strategy.

### 2. Model Pipeline
* **Backbone:** ConvNeXt-Tiny (Selected for efficiency and feature extraction).
* **MIL Module (Task II):** Uses a dual-branch **Attention Mechanism** ($V$ and $U$ branches) to assign "attention weights" to each image in a patient's bag. This allows the model to "focus" on severe frames.
* **Regression Head (Task III):** A custom head (Linear + ReLU + Sigmoid) that outputs a Continuous Precision Score (0.0–3.0) to handle the "Gray Zone" between classes.

### 3. Training Phases & Performance
The training was conducted in four distinct phases to ensure stability and high performance:

| Phase | Description | Status | Key Metrics (Final) |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Image-level Backbone Training | Unfrozen | Learning basic features |
| **Phase 2** | MIL Head Training | Frozen Backbone | **Acc:** 0.8590, **QWK:** 0.9376 |
| **Phase 3** | Full Fine-Tuning | All Unfrozen | **Acc:** 0.8902, **QWK:** 0.9547 |
| **Phase 4** | Severity Regression | Gray Zone Opt. | **MSE:** 0.1510 |

*> **Note:** A Quadratic Weighted Kappa (QWK) of **0.9547** indicates near-perfect agreement with expert annotators.*

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.10 or 3.11
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/Doaa-Mahdy/LIMUC-Attention-Based-Multi-Instance-Learning-for-Ulcerative-Colitis-Severity-Classification.git]
```
### 2. Install Dependencies
```Bash

pip install -r requirements.txt
```

### 3. Run the System
You need to run the backend and frontend simultaneously.

Option A: Using the Helper Script (Linux/Mac/Git Bash)

```Bash

chmod +x run.sh
./run.sh
```

Option B: Manual Start (Windows) Terminal 1 (Backend):

``` Bash

uvicorn main:app --host 127.0.0.1 --port 8000
```

Terminal 2 (Frontend):

Bash

streamlit run frontend.py
Access the dashboard at: http://localhost:8501

📂 Project Structure
Plaintext

📦 LIMUC-AI-System
 ┣ 📜 main.py           # FastAPI Backend (Model Inference & XAI)
 ┣ 📜 frontend.py       # Streamlit Dashboard (UI & Visuals)
 ┣ 📜 model_arch.py     # PyTorch Model Definitions (ConvNeXt, MIL, Regressor)
 ┣ 📜 utils.py          # Image processing & helper functions
 ┣ 📜 requirements.txt  # Python dependencies
 ┣ 📜 Dockerfile        # Container configuration
 ┗ 📜 run.sh            # Startup script for deployment
🛡️ License
This project is licensed under the MIT License - see the LICENSE file for details.