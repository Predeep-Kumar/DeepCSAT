# 🤖 DeepCSAT — AI Powered Customer Satisfaction Prediction

A production-grade **Deep Learning powered customer satisfaction intelligence system** that predicts CSAT scores from customer support interaction data using feature engineering, NLP processing, neural networks, and an interactive Streamlit dashboard.

The system performs:

- Customer Satisfaction Score Prediction
- Deep Learning Model Training & Comparison
- Hyperparameter Optimization
- Automated Best Model Selection
- Real-time CSAT Prediction Dashboard
- Text Processing using NLP
- Feature Engineering from Support Interactions
- Model Monitoring & System Health Checks
- Production-ready Streamlit UI with Glassmorphism Design

---

## 🚀 Key Highlights

- End-to-end ML pipeline (data → preprocessing → model → deployment)
- Deep Neural Network architecture comparison
- Hyperparameter tuning using Keras Tuner
- Class imbalance handling using class weights
- NLP preprocessing pipeline for customer remarks
- Automated best-model selection using JSON configuration
- Real-time CSAT prediction interface
- Professional Streamlit UI with glassmorphism design
- Prediction probability visualization
- Modular production-ready architecture

---

## 📁 Project Structure
---

## 📥 Download Full Project

#### Google Drive Link:
https://drive.google.com/drive/folders/16bESa73oRMqD--F12sTshkPzgak7gEIL?usp=sharing


---

## ⚙️ Installation & Setup (Step by Step)

### 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/DeepCSAT.git

```

```
cd DeepCSAT
```

### 2️⃣ Create Virtual Environment

#### Windows
```
py -m venv venv
```

#### Mac / Linux
```
python -m venv venv
```
or
```
python3 -m venv venv
```

### Activate Virtual Environment
#### macOS / Linux
```
source venv/bin/activate
```
#### Windows
```
venv\Scripts\activate
```

### 3️⃣ Install Requirements
```
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```
streamlit run app.py
```
The Streamlit dashboard will open automatically in your browser.

---
## 🧠 System Architecture (High Level)
```
Raw Customer Support Dataset
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Text Preprocessing (NLP)
        ↓
Exploratory Data Analysis
        ↓
Hypothesis Testing
        ↓
Train/Test Split
        ↓
Class Imbalance Handling
        ↓
Feature Scaling
        ↓
Deep Learning Model Training
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Streamlit Deployment
```
---

## 📊 Core Functional Modules
### 📈 1. CSAT Prediction Engine

Predicts the customer satisfaction score (1–5 scale) using customer support interaction data.

#### Inputs

- Support Channel

- Issue Category

- Sub Category

- Customer City

- Agent Shift

- Response Time

- Survey Delay

- Customer Remarks

#### Pipeline

- Text preprocessing

- Feature engineering

- Feature scaling

- Model inference

- Probability distribution generation

#### Output

- Predicted CSAT score

- Probability distribution

- Model used

### 🧠 2. Deep Learning Models

Three neural network architectures were trained.

#### Baseline ANN
- Input Layer
- Dense 128
- Dense 64
- Dense 32
- Softmax Output

#### Dropout ANN
- Input Layer
- Dense 128
- Dropout
- Dense 64
- Dropout
- Dense 32
- Softmax Output
#### Deep ANN
- Input Layer
- Dense 256
- Dense 128
- Dense 64
- Dense 32
- Softmax Output

Each architecture is trained with class imbalance handling and evaluated using multiple metrics.

### 🔧 3. Hyperparameter Tuning

Model performance was optimized using Keras Tuner.

#### Parameters Tuned

- Number of neurons

- Dropout rate

- Learning rate

- Network depth

The best model is selected based on validation accuracy.

---
## 📊 Model Evaluation Metrics

Models are evaluated using:

- Accuracy

- Precision

- Recall

- F1 Score

- Confusion Matrix

- Validation Accuracy

These metrics ensure the model performs well across all CSAT classes.

---
## 🎨 Streamlit Dashboard Features
### 📊 Dashboard Modules

- CSAT Prediction
- Probability Distribution Visualization
- Dataset Comparison Insights
- Model Monitoring

### 🎛 Sidebar Controls

- Automatic Best Model Selection

- Manual Model Override

- System Health Monitoring

- Model Loading Status

- Scaler Status

- Dataset Status


### 📊 Prediction Visualization

- CSAT prediction gauge

- Probability distribution bar chart
- Confidence breakdown table
---
## 🛡 System Stability Features

- Safe model loading

- JSON-based best model configuration

- Feature alignment protection

- Class imbalance handling

- Scaler compatibility protection

- Prediction fallback handling

- Streamlit widget key protection
---
## 📌 Ideal Use Cases

- Customer support analytics platforms

- SaaS customer experience dashboards

- AI-powered support intelligence tools

- ML deployment portfolios

- End-to-end ML system demonstrations

---
## 🚀 Future Scope & Enhancements

- Sentiment analysis of customer remarks

- Transformer-based NLP models (BERT)

- Explainable AI using SHAP

- Automated model retraining pipelines

- Real-time API deployment

- Cloud-based deployment (AWS/GCP)

- Customer support analytics dashboards

- Real-time streaming data integration

- Data drift detection

- Model monitoring pipelines

---
## 🏁 Conclusion

DeepCSAT demonstrates the complete lifecycle of a production-ready machine learning system for customer satisfaction prediction.

The system integrates advanced data preprocessing, deep learning models, hyperparameter optimization, and interactive visualization into a unified application.

By combining structured feature engineering, NLP processing, and automated model selection, the platform showcases real-world AI deployment practices and provides actionable insights for customer support analytics.

This project highlights strong applied machine learning engineering practices, scalable architecture design, and modern data application deployment.

--- 
## 🤝 Author
### Predeep Kumar

Machine Learning Engineer | Applied AI Systems | Production ML Deployment

Built with ❤️ as a full-stack AI system demonstrating real-world customer satisfaction prediction and analytics.
