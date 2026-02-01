# ML Assignment 2 – Classification Models
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Overview
This project implements multiple machine learning classification models and deploys them using a Streamlit web application.

## A- Problem Statement
The objective of this assignment is to design and implement an end-to-end machine learning classification pipeline using a real-world dataset. The task involves selecting an appropriate classification dataset, performing necessary data preprocessing, training multiple machine learning classification models, and evaluating their performance using standard evaluation metrics.

Six different classification algorithms are implemented, including both individual and ensemble learning methods. The performance of these models is compared using metrics such as Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). These metrics provide a comprehensive understanding of each model’s predictive capability, especially in the context of binary classification problems.

In addition to model development and evaluation, the assignment also emphasizes practical deployment. An interactive web application is developed using Streamlit to demonstrate model predictions in real time. The application is deployed on Streamlit Community Cloud, enabling public access through a clickable link. This assignment helps in understanding the complete machine learning workflow, from data analysis and modeling to user interface design and cloud deployment.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## B- Dataset Description:- 
The dataset used for this assignment is the Heart Failure Prediction Dataset, obtained from Kaggle. This dataset is derived from the UCI Machine Learning Repository and is specifically designed for predicting the presence of heart disease in patients based on various clinical attributes. The problem is formulated as a binary classification task, where the goal is to predict whether a patient has heart disease or not.

The dataset contains 918 instances and 12 features, satisfying the minimum requirements specified in the assignment. The target variable, HeartDisease, indicates the presence (1) or absence (0) of heart disease. The features include a combination of numerical and categorical attributes such as age, sex, chest pain type, resting blood pressure, cholesterol level, maximum heart rate, exercise-induced angina, and ECG-related measurements.

The dataset does not contain missing values, making it suitable for direct modeling after appropriate preprocessing. Categorical features are encoded using suitable encoding techniques, and numerical features are scaled to ensure optimal performance of distance-based and gradient-based models. Overall, this dataset provides a realistic and medically relevant use case for evaluating and comparing different machine learning classification algorithms.
## Dataset
- Dataset: Heart Disease Prediction
- Source: Kaggle (UCI Heart Disease)
- Instances: 918
- Features: 12
- Type: Binary Classification
  
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## C - Model Used:- 

## Models Implemented
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors
- Naive Bayes
- Random Forest
- XGBoost

## Evaluation Metrics
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

# Comparison Table below with the evaluation metrics calculated:- 

| ML Model Name             | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC      |
| ------------------------- | -------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression       | 0.885870 | 0.929699 | 0.871560  | 0.931373 | 0.900474 | 0.769383 |
| Decision Tree             | 0.788043 | 0.781325 | 0.788991  | 0.843137 | 0.815166 | 0.569116 |
| K-Nearest Neighbors (kNN) | 0.885870 | 0.935976 | 0.885714  | 0.911765 | 0.898551 | 0.768600 |
| Naive Bayes               | 0.913043 | 0.945122 | 0.930000  | 0.911765 | 0.920792 | 0.824626 |
| Random Forest (Ensemble)  | 0.869565 | 0.931432 | 0.875000  | 0.892157 | 0.883495 | 0.735558 |
| XGBoost (Ensemble)        | 0.858696 | 0.921927 | 0.872549  | 0.872549 | 0.872549 | 0.714012 |

# Comparison Table with the evaluation metrics calculated 3 round of value:- 

| ML Model Name             | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
| ------------------------- | -------- | ----- | --------- | ------ | -------- | ----- |
| Logistic Regression       | 0.886    | 0.930 | 0.872     | 0.931  | 0.900    | 0.769 |
| Decision Tree             | 0.788    | 0.781 | 0.789     | 0.843  | 0.815    | 0.569 |
| K-Nearest Neighbors (kNN) | 0.886    | 0.936 | 0.886     | 0.912  | 0.899    | 0.769 |
| Naive Bayes               | 0.913    | 0.945 | 0.930     | 0.912  | 0.921    | 0.825 |
| Random Forest (Ensemble)  | 0.870    | 0.931 | 0.875     | 0.892  | 0.883    | 0.736 |
| XGBoost (Ensemble)        | 0.859    | 0.922 | 0.873     | 0.873  | 0.873    | 0.714 |

## Dataset observations on the performance of each model:- 

| ML Model Name                 | Observation about Model Performance |

| **Logistic Regression**       | Logistic Regression achieved high accuracy (0.886) and AUC (0.930), indicating strong baseline performance. Its high recall                                                             |  (0.931) shows effective identification of heart disease cases. However, as a linear model, it may not capture complex non-                                                             |   linear patterns present in the dataset.|

| **Decision Tree**             |  The Decision Tree model recorded the lowest performance among all models with an accuracy of 0.788 and AUC of 0.781.                                                                   |  Although recall was reasonably high (0.843), the low MCC (0.569) indicates poor overall predictive balance, suggesting                                                                 |  overfitting and weaker generalization.|

| **K-Nearest Neighbors (kNN)** | kNN demonstrated strong performance with high accuracy (0.886) and AUC (0.936). Balanced precision and recall values                                                                    | indicate consistent classification results. However, the model is sensitive to feature scaling and computationally                                                                      | expensive for larger datasets.|

| **Naive Bayes**               | Naive Bayes outperformed all other models, achieving the highest accuracy (0.913), AUC (0.945), F1 score (0.921), and MCC                                                               | (0.825). Despite its assumption of feature independence, it generalized exceptionally well and proved highly effective for                                                              | this dataset.|

| **Random Forest (Ensemble)**  | Random Forest showed stable and reliable performance with good accuracy (0.870) and AUC (0.931). By aggregating multiple                                                                |  decision trees, it reduced overfitting compared to a single tree, though its performance was slightly lower than Naive                                                                 |   Bayes.|

| **XGBoost (Ensemble)**        | XGBoost delivered balanced performance with precision, recall, and F1 score all around 0.873. While generally powerful, its                                                             | performance was marginally lower than Random Forest and Naive Bayes, possibly due to limited hyperparameter tuning or                                                                   |  dataset simplicity.|

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

GitHub Repository Link containing (Complete source code, equirements.txt and README.md)
GitHub repo link :
https://github.com/Brajesh0907/ML-Assignment-2-Classification
https://github.com/Brajesh0907/ML-Assignment-2-Classification.git
GIT Hub Project folder Link:-
https://github.com/Brajesh0907/ML-Assignment-2-Classification/tree/main/project-folder

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Streamlit Application
The Streamlit app allows users to input patient health parameters and select a
machine learning model to predict the presence of heart disease.

### Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

## Live Streamlit App Link:-
 
https://streamlit.io/cloud --> My apps --> ml-assignment-2-classification
https://ml-assignment-2-classification-8nfotrycph8vnkexhjqnme.streamlit.app/
ml-assignment-2-classification ∙ main ∙ app.py

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
