# 🧠 AI Diabetes Clinical Data Analysis

This repository contains an **educational Artificial Intelligence project** designed for undergraduate students at **Amirkabir University of Technology (Tehran Polytechnic)**.
The project explores **clinical diabetes data** through **exploratory data analysis, clustering, and classification**, highlighting the impact of unsupervised learning on predictive performance.


## 📌 Project Overview

This project introduces students to a full AI pipeline on real-world medical data. It combines data exploration, preprocessing, clustering, and supervised classification to analyze how data structure influences model performance in healthcare applications.


## 🎯 Learning Objectives

Students completing this project will learn to:

* Perform exploratory data analysis (EDA) on clinical datasets
* Preprocess mixed numerical and categorical features
* Handle imbalanced data using **SMOTE**
* Apply **clustering techniques** to structure data
* Train and evaluate **classification models**
* Analyze the effect of clustering on classification performance
  

## 🩺 Dataset: Comprehensive Diabetes Clinical Dataset

This project uses the **Comprehensive Diabetes Clinical Dataset**, which contains **clinical and demographic information** of patients related to diabetes.

* **Number of records:** ~100,000
* **Task:** Diabetes risk analysis and prediction
* **Source:** Kaggle
* **Link:** [https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset/data](https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset/data)

The primary objective of this dataset is to analyze patient health indicators and **predict the likelihood of diabetes** based on clinical and lifestyle factors such as glucose level, BMI, hypertension, and smoking history.

---

## 📊 Feature Description

| Feature                 | Description                                                                  |
| ----------------------- | ---------------------------------------------------------------------------- |
| **Gender**              | Patient’s gender (e.g., Male, Female).                                       |
| **Age**                 | Age of the patient in years.                                                 |
| **Location**            | Geographical location (city, state, or region).                              |
| **Race**                | Patient’s race or ethnicity (e.g., Caucasian, African American, Asian).      |
| **Hypertension**        | Indicates hypertension status (1 = Yes, 0 = No).                             |
| **Heart Disease**       | Indicates heart disease status (1 = Yes, 0 = No).                            |
| **Smoking History**     | Smoking status (e.g., never, former, current).                               |
| **BMI**                 | Body Mass Index calculated from height and weight.                           |
| **HbA1c Level**         | Average blood sugar level over the past 2–3 months.                          |
| **Blood Glucose Level** | Blood glucose level at the time of measurement.                              |
| **Diabetes**            | Target variable indicating diabetes status (1 = Diabetic, 0 = Non-diabetic). |

---

## 💡 Dataset Notes

* The dataset contains a **significant class imbalance**, motivating the use of **SMOTE** during preprocessing.
* Features include a mix of **numerical and categorical variables**, making it suitable for both EDA and machine learning experiments.
* Due to its size and realism, the dataset is well-suited for **educational AI projects** and healthcare-related analysis.

## 🛠️ Tools & Libraries

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Opendatasets


## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)

* Inspection of dataset structure, feature types, and summary statistics
* Visualization of feature distributions and relationships
* Analysis of class imbalance in diabetes outcomes


### 2. Data Preprocessing

* Encoding categorical variables using label encoding
* Handling class imbalance through **SMOTE (Synthetic Minority Over-sampling Technique)**
* Preparing feature matrices for clustering and classification models


### 3. Clustering

* **K-Means clustering** is applied to identify latent patient subgroups
* Cluster labels are used to incorporate structural information into downstream classification tasks


### 4. Classification Models

The following supervised learning models are implemented:

* XGBoost
* Random Forest
* Logistic Regression
* Support Vector Machine (SVM)
* Decision Tree
* K-Nearest Neighbors (KNN)
* Naive Bayes

Models are evaluated using **Accuracy, Precision, Recall, and F1-score** to account for class imbalance and clinical relevance.


### 5. Hyperparameter Tuning

To improve model performance and ensure fair comparison, **hyperparameter tuning** is performed for selected classification models.

* Key hyperparameters such as tree depth, number of estimators, regularization strength, kernel parameters, and neighbor counts are systematically adjusted
* Model configurations are optimized based on **F1-score**, prioritizing balanced performance between precision and recall
* Tuned models demonstrate improved stability and generalization compared to default parameter settings

This step highlights the importance of model optimization in real-world medical applications, where default parameters may not yield optimal results.


### 6. Evaluation Strategy

* Performance is evaluated both **with and without clustering** to assess the impact of unsupervised learning
* Comparative analysis focuses on how clustering and hyperparameter tuning influence classification effectiveness

## 📊 Results & Discussion

### 🔹 Classification Without Clustering
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/65e483c0-49ba-4b9e-949f-19453de69fe5" />


This figure compares classification models trained **directly on the dataset without clustering**.

**Observations:**

* **XGBoost** and **Random Forest** achieve the highest overall performance, with F1-scores around **0.60**
* High **recall** across most models indicates strong sensitivity to diabetic cases
* Lower **precision** reflects the inherent difficulty of reducing false positives in medical diagnosis
* **Naive Bayes** and **KNN** show weaker performance, particularly in precision and F1-score

---

### 🔹 Classification with K-Means Clustering

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/58f3e8d4-bb42-491a-aba0-f70ca028742f" />


In this experiment, **K-Means clustering** is applied before classification.

**Observations:**

* Tree-based models (**XGBoost, Random Forest**) benefit most from clustering
* **F1-score improvements** are observed in several models, especially XGBoost
* Clustering helps reveal latent structure, improving class separability
* **Naive Bayes** performs poorly after clustering, indicating sensitivity to feature distributions
* The effect of clustering varies by model, highlighting algorithm-dependent benefits

---

### 🔹 Comparative Discussion

| Aspect     | Without Clustering      | With K-Means                       |
| ---------- | ----------------------- | ---------------------------------- |
| Best Model | XGBoost / Random Forest | XGBoost                            |
| Recall     | Generally high          | Slightly reduced but more balanced |
| Precision  | Low–moderate            | Improved for tree-based models     |
| F1 Score   | Moderate                | Improved in several classifiers    |

Overall, the results demonstrate that **unsupervised learning can enhance supervised classification** by providing additional structural insight into complex clinical datasets.

---

## 🎓 Academic Context

* **Course:** Artificial Intelligence
* **University:** Amirkabir University of Technology
* **Instructor:** Dr. Mehdi Ghatee
* **Teaching Assistant:** Behnam Youssefi-Mehr

