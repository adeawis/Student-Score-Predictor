# 🎓 Student Score Predictor

A machine learning web application that predicts student exam scores using a trained LightGBM Regressor model.  
The system combines data analysis, ensemble learning, and a Flask-based web interface to provide real-time score predictions.

---

## 📌 Project Overview

This project focuses on predicting student academic performance using historical data and machine learning techniques.

A LightGBM Regressor model was selected due to its high efficiency and strong performance on tabular datasets. After training and evaluation, the model achieved a low Mean Absolute Error (MAE), indicating accurate prediction capability.

To demonstrate real-world usage, the trained model was integrated into a Flask web application where users can input student-related data and instantly receive predicted exam scores.

---

## 🧠 Key Features

- 📊 Data preprocessing and analysis  
- ⚡ LightGBM ensemble regression model  
- 📈 Feature importance analysis  
- 🌐 Flask web application for real-time predictions  
- 📦 Pre-trained model and scaler included  

---

## 📁 Repository Structure

```bash
Student-Score-Predictor/
│
├── project/
│ ├── static/ # CSS and static files
│ ├── templates/ # HTML templates for Flask
│ ├── app.py # Flask application
│ ├── requirements.txt # Required Python libraries
│ ├── model_columns.pkl # Feature columns used in training
│ ├── scaler.pkl # Data scaler
│ ├── student_score_model.pkl# Trained LightGBM model
│ └── student_score_model.txt# Model details
│
├── train.csv # Training dataset
├── test.csv # Testing dataset
├── sample_submission.csv # Sample prediction format
└── .hintrc
```

---

## 🛠️ Technologies Used

- Python  
- LightGBM  
- pandas & NumPy  
- scikit-learn  
- Flask  
- HTML/CSS  

---

## 🚀 How to Run the Web App

### 1. Clone the repository

```bash
git clone https://github.com/adeawis/Student-Score-Predictor.git
cd Student-Score-Predictor/project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Flask app

```bash
python app.py
```
