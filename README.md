# Predicting Late Delivery Risk

## Project Overview

This project focuses on building a predictive model to identify orders at risk of late delivery using the DataCo Supply Chain Dataset. Timely delivery is crucial for customer satisfaction and business success, and this project aims to provide a tool to proactively mitigate late deliveries.

## Data Source

The dataset used for this project is the **DataCo Supply Chain Dataset**, available on Kaggle. It contains transaction records with various details about orders, shipping, customers, and products.

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

## How to Run the Code

To run the code in this project:

1.  Ensure you have a Python environment with the necessary libraries installed (e.g., pandas, numpy, scikit-learn, matplotlib, seaborn, joblib). You can install them using pip:
