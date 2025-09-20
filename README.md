# Predicting Late Delivery Risk

## Project Overview

This project aims to develop a predictive model that identifies orders at risk of late delivery using the DataCo Supply Chain Dataset. Timely delivery is crucial for customer satisfaction and business success, and this project aims to provide a tool to mitigate late deliveries proactively.

## Business Objective

This framework provides a comprehensive approach to predict customer returns, optimize logistics, and identify high-risk product categories using the DataCo Supply Chain dataset. This project will implement multiple machine learning techniques to address three core business objectives.

## Data Source

The dataset used for this project is the [**DataCo Supply Chain Dataset**](https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fcode%2Fadityabatsexemplary%2Fsupply-chain%2Finput%2F), available on Kaggle. It contains transaction records with various details about orders, shipping, customers, and products.


## Methodology

The project followed a standard machine learning workflow:

1.  **Data Loading and Cleaning**: Loaded the dataset, checked for and removed duplicate rows, and handled missing values by dropping columns with excessive missingness and dropping rows with a small number of missing values in key columns.
2.  **Exploratory Data Analysis (EDA)**: Explored data distributions, relationships between numerical features (using a correlation matrix), and relationships between categorical features and the target variable (using count plots). Identified potential outliers but decided not to handle them at this stage based on domain considerations. Explored in-depth relationships between numerical and categorical features and the target variable using box plots.
3.  **Feature Engineering**: Extracted temporal features (year, month, day, day of week) from order and shipping dates.
4.  **Data Splitting**: Split the data into training and testing sets.
5.  **Preprocessing**: Defined preprocessing pipelines using `ColumnTransformer` to handle numerical and categorical features. This included imputation for remaining missing values, scaling of numerical features, and one-hot encoding of categorical features. Irrelevant columns (identifiers, high cardinality, data leakage, original dates) were excluded. The preprocessor was applied to the training and testing data.
6.  **Model Selection and Training**: Evaluated several classification models (Logistic Regression, Random Forest, KNN, Dummy Classifier, Decision Tree, Gradient Boosting). Defined pipelines for each model incorporating the preprocessing steps. Trained the models on the preprocessed training data.
7.  **Hyperparameter Tuning**: Optimized the hyperparameters of the best-performing models (Random Forest and Gradient Boosting) using RandomizedSearchCV.
8.  **Model Evaluation**: Evaluated the tuned models on the test data using metrics such as Accuracy, ROC-AUC, Confusion Matrix, Precision, Recall, and F1-score. Plotted ROC curves for visual comparison.
9.  **Model Interpretation**: Analyzed feature importances from the tuned Random Forest and Gradient Boosting models to understand the key predictors of late delivery risk.
10. **Final Model Selection**: Selected Random Forest as the final model based on its performance and robustness.
11. **Model Saving**: Saved the trained Random Forest model pipeline.

## Key Findings

*   Features related to **shipping mode**, **order status** (particularly 'SUSPECTED_FRAUD' and 'CANCELED'), and **temporal features** (day of the week for order and shipping dates) were identified as the most important predictors of late delivery risk.
*   The chosen Random Forest model achieved strong performance metrics in predicting late delivery risk.

## Repository Contents:

  * **data:** Contains the dataset used to build predictive models. [click here for datasets](https://github.com/gethiten/SupplyChainOptimization-AI/blob/main/data/)
  * **Supply-Chain-Optimization.ipynb
:** Jupyter Notebook containing the code for data exploration, analysis, modelling, evaluation, and visualization. [Supply-Chain-Optimization.ipynb
](https://github.com/gethiten/SupplyChainOptimization-AI/blob/main/Supply-Chain-Optimization.ipynb)
  * **images:** Folder containing generated visualizations. [Images](https://github.com/gethiten/SupplyChainOptimization-AI/tree/main/images)
  * **README.md:** This file provides an overview of the project. [README.md](https://github.com/gethiten/SupplyChainOptimization-AI/tree/main/README.md)
  * **models:** Tuned Random Forest model ['random_forest_late_delivery_risk_model.joblib'](https://github.com/gethiten/SupplyChainOptimization-AI/blob/main/random_forest_late_delivery_risk_model.joblib) saved to models folder

## How to Run the Code

To run the code in this project:

1.  Ensure you have a Python environment with the necessary libraries installed (e.g., pandas, numpy, scikit-learn, matplotlib, seaborn, joblib). You can install them using pip:
