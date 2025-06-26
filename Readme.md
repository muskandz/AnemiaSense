# AnemiaSense: Leveraging Machine Learning for Precise Anemia Detection

AnemiaSense is a web-based machine learning application that predicts the likelihood of anemia using hematological data. It enables early detection, personalized intervention, and remote health monitoring.

---

## Table of Contents
- [About the Project](#about-the-project)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Features](#features)
- [How It Works](#how-it-works)
- [Run Locally](#run-locally)

---

## About the Project

Anemia affects nearly 25% of the global population, yet remains underdiagnosed due to limited access to testing facilities. AnemiaSense automates anemia detection by analyzing basic blood parameters using a trained ML model, and makes this predictive tool available through a web interface built with Flask.

---

## Tech Stack

- Python 3.10+
- Scikit-learn, Pandas, Seaborn, Matplotlib
- Flask
- Joblib (for saving/loading ML models)
- HTML5, CSS3, Bootstrap

---

## Dataset

- **File:** `anemia.csv`  
- **Rows:** 1421  
- **Features:**
  - Gender (0 = Female, 1 = Male)
  - Hemoglobin
  - MCH
  - MCHC
  - MCV
  - Result (Target: 0 = Normal, 1 = Anemia)

---

## ✨ Features

- Machine Learning-based prediction (Random Forest, Logistic Regression, etc.)
- Visualizations (distribution, heatmap, correlation)
- Exploratory Data Analysis
- Web-based interface for easy access
- Binary output: "Anemia Detected" or "Normal"

---


---

## 🔄 How It Works

### 1. Data Preprocessing & Model Training (`model_training.py`)
- Load and clean dataset
- Train/test split
- Apply ML algorithms: Logistic Regression, Random Forest, SVM, etc.
- Evaluate models
- Save the best one (`model.pkl`)

### 2. Web Interface with Flask (`app.py`)
- Renders an input form on the homepage
- Accepts user inputs (gender, hemoglobin, etc.)
- Loads saved model and predicts
- Displays output: "Anemia Detected" or "Normal"

---

## ▶️ Run Locally

> 🛠 Prerequisites: Python 3.10+, pip installed

```bash
# Step 1: Clone the repo
git clone https://github.com/muskandz/anemiasense.git
cd anemiasense/Flask

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Train model (or use existing model.pkl)
python model_training.py

# Step 4: Start Flask app
python app.py
