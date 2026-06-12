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

 TotalCharges was transformed using the Yeo-Johnson method to reduce positive skewness and improve distributional symmetry.
 <img width="800" height="450" alt="Screenshot 2026-06-12 161914" src="https://github.com/user-attachments/assets/df64166e-3441-4fe4-8102-1c6d937adf94" />

 tenure and MonthlyCharges did not exhibit significant skewness, and the transformation did not provide any meaningful improvement in their distributions. Therefore, these variables were retained in their original form.
 <img width="800" height="450" alt="image" src="https://github.com/user-attachments/assets/7a7f50fc-4a08-415b-bdcb-3d46f00890d0" />
 <img width="800" height="450" alt="image" src="https://github.com/user-attachments/assets/feaeb7f4-0496-408f-aac4-655ffdbf1296" />

 Since SeniorCitizen is a binary feature (0 = No, 1 = Yes), most numerical transformation techniques are either inappropriate or provide no benefit:

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

<img width="450" height="300" alt="image" src="https://github.com/user-attachments/assets/c4037442-f3d8-4d62-b9b6-866204d20b5f" />
<img width="450" height="300" alt="image" src="https://github.com/user-attachments/assets/46b84bcb-c804-406a-80f4-3d7442592680" />
<img width="450" height="300" alt="image" src="https://github.com/user-attachments/assets/baf2dd33-3096-4e45-86c8-2b78d77bb40f" />



### Result

✅ No significant outliers were detected.

Therefore, no outlier treatment was required.

---

# 🎯 Feature Selection

Statistical Hypothesis Testing was performed to identify important features.

Features that were not statistically significant were removed from the dataset.
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/48cc4214-70c3-4dfd-b23a-c6724a340e2b" />


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

* <img width="750" height="287" alt="image" src="https://github.com/user-attachments/assets/e35e461e-85d6-4fcd-81e9-2e3301af96e4" />


After encoding, numerical and categorical datasets were merged into a final modeling dataset.

---

# ⚖️ Data Balancing

The training dataset exhibited a significant class imbalance, which could lead to biased predictions toward the majority class.

### Class Distribution Before Balancing

| Customer Type              | Count |
| -------------------------- | ----- |
| Good Customers (Churn = 1) | 1,496 |
| Bad Customers (Churn = 0)  | 4,138 |

The dataset was heavily skewed toward non-churn customers, making it difficult for machine learning models to learn churn patterns effectively.

### Technique Used

**Random Oversampling** was applied to the minority class in the training dataset.

### Class Distribution After Balancing

| Customer Type              | Count |
| -------------------------- | ----- |
| Good Customers (Churn = 1) | 4,138 |
| Bad Customers (Churn = 0)  | 4,138 |

After oversampling, both classes contained an equal number of observations, resulting in a balanced training dataset.

### Benefits of Data Balancing

* Improved minority class representation
* Reduced model bias toward the majority class
* Enhanced churn detection capability
* Improved Recall and ROC-AUC performance
* Better generalization on unseen customer data

Balancing the dataset helped the machine learning models learn customer churn patterns more effectively and improved overall predictive performance.


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
<img width="650" height="400" alt="image" src="https://github.com/user-attachments/assets/f95c1abf-4b71-4603-bb9f-343634f6e9f4" />


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

### before Churn prediction 
<img width="706" height="482" alt="image" src="https://github.com/user-attachments/assets/f7d964c5-6efa-4d59-8fd9-3e97d5340a11" />

### After churn prediction 

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/82f33d6e-f58e-494a-9e46-94b6a152ea7b" />
The Logistic Regression model correctly identified 312 churning customers and 738 non-churning customers. The model missed only 61 actual churners, indicating a strong recall for churn prediction. Although 298 non-churning customers were incorrectly classified as churners, the model demonstrates good effectiveness in identifying customers at risk of churning


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
 <img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/ec0671dc-641f-484f-be94-3e6b1b59c103" />
 

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

# 🚀 Live Demo

### Render Deployment Link

🔗 **https://telecom-customer-churn-prediction-xt9p.onrender.com**

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


