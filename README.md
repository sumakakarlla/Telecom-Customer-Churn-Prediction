# 📊 Telecom Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Customer churn is one of the biggest challenges faced by telecom companies. Churn occurs when customers discontinue using a company's services, leading to revenue loss and increased customer acquisition costs. Identifying customers who are likely to churn enables businesses to take proactive measures and improve customer retention.

This project uses Machine Learning techniques to analyze customer demographics, service subscriptions, billing information, and behavioral patterns to predict whether a customer is likely to churn.

The project covers the complete Machine Learning lifecycle, including data preprocessing, feature engineering, model training, evaluation, model selection, serialization, and deployment using Flask.

---

## 🎯 Business Problem

Telecom companies lose significant revenue when customers leave their services. The goal of this project is to build a predictive model that identifies customers at risk of churning, allowing businesses to implement targeted retention strategies and reduce customer attrition.

---

## 📂 Dataset Description

The dataset contains customer information including:

* Customer Demographics
* Service Subscriptions
* Internet Services
* Billing Information
* Contract Details
* Payment Methods
* Churn Status

---

# 🔍 Data Preprocessing

## 1. Missing Value Handling

The dataset contained missing values in the **TotalCharges** feature.

### Missing Values Before Imputation

| Dataset       | Missing Values |
| ------------- | -------------- |
| Training Data | 10             |
| Testing Data  | 1              |

### Technique Used

Mode Imputation was selected because it was the most suitable method for preserving the existing data distribution.

### Missing Values After Imputation

| Dataset       | Missing Values |
| ------------- | -------------- |
| Training Data | 0              |
| Testing Data  | 0              |

---

## 2. Train-Test Split

The dataset was divided into:

* Training Dataset
* Testing Dataset

This ensures that model performance is evaluated on unseen data.

---

## 3. Feature Separation

### Numerical Features

* SeniorCitizen
* tenure
* MonthlyCharges
* TotalCharges

### Categorical Features

* gender
* Partner
* Dependents
* PhoneService
* MultipleLines
* InternetService
* OnlineSecurity
* OnlineBackup
* DeviceProtection
* TechSupport
* StreamingTV
* StreamingMovies
* Contract
* PaperlessBilling
* PaymentMethod
* sims

---

## 4. Variable Transformation

Numerical features were analyzed for skewness.

A Yeo-Johnson transformation was applied to reduce skewness in the TotalCharges feature.

### Before Transformation

* SeniorCitizen
* tenure
* MonthlyCharges
* TotalCharges

### After Transformation

* SeniorCitizen
* tenure
* MonthlyCharges
* TotalCharges_yeo

### Benefits

* Reduced skewness
* Improved feature distribution
* Enhanced model learning

---

## 5. Outlier Analysis

Outlier detection was performed on all numerical features.
<img width="567" height="352" alt="image" src="https://github.com/user-attachments/assets/c91e74fc-adfe-4b60-aa7a-d7943d02a3b8" />


### Result

✅ No significant outliers were detected.

Therefore, no outlier treatment was required.

---

# 🎯 Feature Selection

Statistical Hypothesis Testing was performed to identify important features.

Features that were not statistically significant were removed from the dataset.

### Benefits

* Reduced noise
* Improved interpretability
* Reduced overfitting
* Better model performance

---

# 🔄 Feature Encoding

Categorical variables were converted into numerical format.

## One-Hot Encoding

Applied to:

* Gender
* Paperless Billing
* Payment Method
* SIM Provider

## Ordinal Encoding

Applied to:

* Partner
* Dependents
* PhoneService
* MultipleLines
* InternetService
* OnlineSecurity
* OnlineBackup
* DeviceProtection
* TechSupport
* StreamingTV
* StreamingMovies
* Contract

After encoding, numerical and categorical datasets were merged into a final modeling dataset.

---

# ⚖️ Data Balancing

The training dataset contained class imbalance.

### Technique Used

Oversampling

### Benefits

* Balanced class distribution
* Improved minority class prediction
* Better churn detection performance

---

# 📏 Feature Scaling

Feature scaling was performed using:

### StandardScaler (Z-Score Normalization)

Formula:

z = (x − μ) / σ

### Benefits

* Standardized feature ranges
* Improved model convergence
* Better performance for distance-based algorithms

---

# 🤖 Machine Learning Models Trained

The following machine learning algorithms were trained and evaluated:

1. K-Nearest Neighbors (KNN)
2. Naive Bayes
3. Logistic Regression
4. Decision Tree
5. Random Forest
6. AdaBoost
7. Gradient Boosting
8. XGBoost
9. Support Vector Machine (SVM)

---

# 📊 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| KNN                 | 68.28%   |
| Naive Bayes         | 76.79%   |
| Logistic Regression | 74.52%   |
| Decision Tree       | 72.53%   |
| Random Forest       | 77.79%   |
| AdaBoost            | 75.66%   |
| Gradient Boosting   | 71.04%   |
| XGBoost             | 79.56%   |
| SVM                 | 78.42%   |

---

# 📈 Model Evaluation

Each model was evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report
* ROC Curve
* ROC-AUC Score

The ROC Curve was used as the primary metric for model selection because customer churn prediction is an imbalanced classification problem.

---

# 🏆 Final Model Selection

Although XGBoost achieved the highest accuracy score, the final model was selected based on ROC-AUC performance.

### Why ROC-AUC?

ROC-AUC measures a model's ability to distinguish between churn and non-churn customers across different classification thresholds.

This metric is more reliable than accuracy for imbalanced classification problems.

### Selected Model

✅ Logistic Regression

### Reasons for Selection

* Highest ROC-AUC Score
* Strong class separation capability
* Better generalization performance
* Interpretable model coefficients
* Lower complexity
* Easy deployment and maintenance

---

# 🔮 Churn Prediction

The final Logistic Regression model was used to predict customer churn on unseen test data.

The model successfully identified customers at risk of churning and generated reliable predictions.

---

# 💾 Model Serialization

The trained artifacts were saved using Pickle.

### Saved Files

* Logistic Regression Model (.pkl)
* StandardScaler (.pkl)

Benefits:

* Faster deployment
* Reusable models
* Consistent predictions

---

# 🌐 Web Application Development

A complete web application was developed for real-time churn prediction.

## Backend

* Flask

## Frontend

* HTML
* CSS

## Features

* User-friendly interface
* Real-time predictions
* Customer churn risk assessment
* Responsive design

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy
* matplot.pyplot

### Machine Learning

* Scikit-Learn
* XGBoost

### Web Development

* Flask
* HTML
* CSS

### Version Control

* Git
* GitHub

---

# 📸 Project Screenshots

## Data Preprocessing

(Add Screenshot Here)

## Feature Engineering

(Add Screenshot Here)

## Model Comparison

(Add Screenshot Here)

## ROC Curve

(Add Screenshot Here)

## Web Application

(Add Screenshot Here)

---

# 🚀 Live Demo

### Render Deployment Link

🔗 **Add Your Render URL Here**

---



---

# 👨‍💻 Author

K Suma Bindu

### Skills

* Python
* SQL
* Machine Learning
* Power BI
* Tableau
* Excel

Passionate about transforming raw data into actionable insights and building end-to-end Machine Learning solutions for real-world business problems.


