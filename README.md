# 🍲 Food Recipe Rating Prediction

This project predicts user ratings for food recipes using a mix of structured metadata and free-text reviews. It’s inspired by a Kaggle-style competition and simulates a real-world recommendation scenario.

---

## 🔍 Problem Statement

Given recipe-related data — including review text, thumbs-up/down counts, and user reputation — predict the recipe rating from 0 to 5 stars.

---

## 📊 Dataset Overview

- Text: `RecipeReview` (free-text user reviews)
- Metadata: `Thumbs Up Count`, `Thumbs Down Count`, `User Reputation`, `Best Score`, etc.
- Target: `Rating` (0 to 5, where 0 means "no rating")

---

## 🧠 Approach

### 1. **Exploratory Data Analysis**
- Inspected null values, class imbalance, outliers, and feature correlations.
- Visualized:
  - Rating distribution (class imbalance)
  - Thumbs up vs. down
  - Scatter matrix of numerical features

### 2. **Preprocessing**
- Imputed missing values (`SimpleImputer`)
- Scaled numeric features (`StandardScaler`)
- Vectorized text reviews using `TF-IDF`
- Combined text and numeric features using sparse matrix stacking

### 3. **Modeling**
Tested and compared the following models:

| Model            | Training Acc | Validation Acc |
|------------------|--------------|----------------|
| Logistic Regression | 80%         | 76%            |
| Linear SVC         | 76%         | 75%            |
| XGBoost (best)     | 84%         | 75%            |

- Used `RandomizedSearchCV` for XGBoost tuning, but default performed best
- Plotted confusion matrix and accuracy bar chart for model comparison

### 4. **Class Imbalance Handling**
- Tried `class_weight='balanced'` for Logistic Regression
- Chose not to use it finally, since accuracy (used for evaluation) dropped due to overcompensation
- Focused on improving accuracy for the majority class (5-star ratings)

---

## 🛠️ Technologies Used

- Python (Pandas, NumPy, Scikit-learn)
- NLP: TF-IDF Vectorizer
- Models: Logistic Regression, SVC, XGBoost
- Visualization: Matplotlib, Seaborn

---

## 📁 Project Structure

FOOD-RECIPE-RATING-PREDICTION/
├── datasets/ # Contains train.csv and test.csv
├── notebooks/ # Jupyter notebook(s) with analysis
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## ✅ Key Learnings

- Handling mixed data (text + numeric) in a single ML pipeline
- Managing class imbalance when evaluation metric favors accuracy
- Practical model tuning and decision making under real constraints

---

## 🚀 Future Enhancements

- Use SHAP for model explainability
- Explore deep learning (LSTM, BERT) for review text
- Deploy as a Streamlit app for public demo

---

## 📎 Credits

- Dataset source: Kaggle-style competition (link omitted per license)