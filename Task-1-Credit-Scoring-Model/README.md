# 💳 Credit Scoring Model using Machine Learning

> **CodeAlpha Machine Learning Internship - Task 1**

A Machine Learning project that predicts an individual's **creditworthiness** using historical financial information. The model classifies whether a customer is likely to default on credit payments by analyzing financial indicators such as income, debt ratio, payment history, credit utilization, and more.

---

# 📌 Project Overview

Credit scoring is one of the most important applications of Machine Learning in the banking and finance industry. Financial institutions use credit scoring models to determine whether a customer is eligible for loans or credit cards.

This project implements multiple classification algorithms and compares their performance to select the best model for predicting credit default risk.

---

# 🎯 Objective

Develop a machine learning model that predicts whether a customer is likely to default on credit within the next two years based on previous financial history.

---

# 🚀 Features

- Data Cleaning & Preprocessing
- Missing Value Handling
- Duplicate Removal
- Feature Engineering
- Class Imbalance Handling using SMOTE
- Multiple Machine Learning Models
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Model Comparison
- Best Model Selection
- Model Evaluation
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Credit Risk Prediction
- Streamlit Web Application
- Model Serialization using Joblib

---

# 📂 Project Structure

```
Task-1-Credit-Scoring-Model
│
├── app/
│   ├── app.py
│   └── prediction.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   └── Credit_Scoring_EDA.ipynb
│
├── reports/
│   ├── figures/
│   ├── metrics.csv
│   └── model_report.txt
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   ├── logger.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Streamlit
- Joblib

---

# 📊 Dataset

This project uses the **Give Me Some Credit** dataset.

Dataset Link:

https://www.kaggle.com/competitions/GiveMeSomeCredit

Download the following files:

```
cs-training.csv

cs-test.csv
```

Place them inside:

```
data/raw/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/MohammedBawazir25/codealpha_tasks.git
```

Navigate to the project

```bash
cd codealpha_tasks/Task-1-Credit-Scoring-Model
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Train the model

```bash
python main.py
```

Run Streamlit App

```bash
streamlit run app/app.py
```

---

# 📈 Machine Learning Pipeline

```
Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
SMOTE Balancing
     │
     ▼
Train-Test Split
     │
     ▼
Model Training
     │
     ▼
Model Evaluation
     │
     ▼
Best Model Selection
     │
     ▼
Prediction
```

---

# 🤖 Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

# 📊 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

---

# 📷 Results

Example comparison:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------|---------:|----------:|-------:|---------:|--------:|
| Logistic Regression | 74.19% | 16.85% | 72.48% | 27.35% | 79.55% |
| Decision Tree | 86.15% | 15.10% | 23.08% | 18.25% | 57.36% |
| Random Forest | 92.08% | 30.21% | 13.84% | 18.98% | 75.98% |

**Best Model:** Logistic Regression

---

# 📸 Screenshots

## Dashboard

Add screenshot here

```
assets/dashboard.png
```

---

## Prediction

Add screenshot here

```
assets/prediction.png
```

---

## ROC Curve

Add screenshot here

```
assets/roc_curve.png
```

---

# 💡 Feature Engineering

Additional features created:

- Debt Per Income
- Total Late Payments
- Loans Per Dependent

These engineered features improve the predictive capability of the model.

---

# 🔍 Future Improvements

- XGBoost
- LightGBM
- CatBoost
- Hyperparameter Tuning using GridSearchCV
- Explainable AI (SHAP)
- Docker Deployment
- Cloud Deployment
- REST API using Flask

---

# 🎓 Learning Outcomes

Through this project, I learned:

- Data preprocessing techniques
- Feature engineering
- Handling imbalanced datasets using SMOTE
- Training multiple ML models
- Model comparison
- Performance evaluation
- Model deployment using Streamlit
- GitHub project management

---

# 👨‍💻 Author

**Mohammed Saleh Bawazir**

Artificial Intelligence & Machine Learning Engineering Student

GitHub:
https://github.com/MohammedBawazir25

LinkedIn:
https://www.linkedin.com/in/mohammed-saleh-bawazir/

---

# 📜 License

This project is created for educational and internship purposes under the CodeAlpha Machine Learning Internship Program.

---

# ⭐ If you found this project useful, consider giving it a Star on GitHub!