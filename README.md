# 📡 ChurnGuard AI: End-to-End Telecom Churn Predictor
> **Retain your customers with data-driven intelligence.**

---

## 💼 The Business Problem
In the highly competitive Telecom industry, the cost of **acquiring a new customer** is often 5x to 25x higher than **retaining an existing one**. Customer churn (loss) directly impacts the bottom line. 

**ChurnGuard AI** solves this by identifying high-risk customers before they leave, allowing marketing teams to intervene with personalized retention offers, effectively reducing churn rates and maximizing Lifetime Value (LTV).

## 🚀 Technical Architecture
This project implements a full-stack Machine Learning pipeline:

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) – A modern, fast web framework for building APIs with Python 3.7+.
- **Model**: [XGBoost](https://xgboost.readthedocs.io/) – Gradient Boosted Decision Trees optimized for speed and performance.
- **Preprocessing**: 
    - **SMOTE** (Synthetic Minority Over-sampling Technique) was used during training to handle class imbalance.
    - **StandardScaler** for feature normalization.
    - **One-Hot Encoding** for categorical variables.
- **Frontend**: A modern SaaS-style dashboard built with **Tailwind CSS**, **Chart.js**, and asynchronous JavaScript.

## 📊 Performance Metrics
The model was evaluated on a held-out test set with the following results:
- **ROC-AUC Score**: `0.84`
- **Precision/Recall**: Balanced via SMOTE to ensure high sensitivity to churners.

## 🛠️ Project Structure
```text
.
├── models/             # Serialized ML models (.pkl)
├── static/             # Frontend assets (HTML, CSS, JS)
├── app.py              # FastAPI server & prediction logic
├── requirements.txt    # Dependency list
└── README.md           # Project documentation
```

## ⚙️ How to Run
Follow these steps to set up the project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/churn-predictor.git
   cd churn-predictor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**:
   ```bash
   uvicorn app:app --reload --port 8080
   ```

4. **Open the Dashboard**:
   Navigate to `http://127.0.0.1:8080` in your web browser.
   <img width="2559" height="1217" alt="Screenshot 2026-05-15 034353" src="https://github.com/user-attachments/assets/fe58c816-4b38-486f-8b74-31e168961524" />



## 🔮 Future Work
- [ ] **Dockerization**: Containerize the application for seamless deployment to AWS/GCP.
- [ ] **A/B Testing**: Integrate a module to track the effectiveness of retention offers.
- [ ] **Retraining Pipeline**: Implement automated model monitoring and retraining using MLflow.

---
Developed by Shada Khaled 
