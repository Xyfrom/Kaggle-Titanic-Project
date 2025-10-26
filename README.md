# Titanic: Survival Analysis and Prediction

This repository contains a data analysis and machine learning project based on Kaggle's "Getting Started" competition: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic).

The project's goal is to analyze passenger data (e.g., class, sex, age) to discover factors that influenced survival rates and to build a predictive model based on these insights.

**Final Kaggle Public Score (Accuracy): 72.72%**

## Technologies Used
* **Python 3**
* **Pandas:** For data manipulation and analysis.
* **Scikit-learn (sklearn):** For building and training the machine learning model (DecisionTreeClassifier).
* **Jupyter Notebook (via VS Code):** For interactive development and analysis.

## Project Workflow

This project follows a complete, end-to-end data science lifecycle:

### 1. Exploratory Data Analysis (EDA)
The `train.csv` data was loaded and analyzed using `groupby` functions to test initial hypotheses:
* **Sex:** The survival rate for females (~74%) was significantly higher than for males (~18%).
* **Ticket Class (Pclass):** 1st Class passengers (~63%) had a much higher chance of survival than 3rd Class passengers (~24%).
* **Cross-Analysis:** A key insight was that a 3rd Class female (~50%) had a better chance of survival than a 1st Class male (~37%), indicating that "Sex" was a stronger predictor than "Class".

### 2. Data Cleaning and Feature Engineering
To prepare the data for the model, the following steps were taken:
* Missing `Age` values (`NaN`) were filled with the mean age of all passengers (`fillna`).
* The categorical `Sex` column ('female'/'male') was converted into numerical values (1/0) for the model to understand (`map`).

### 3. Model Training
* `Pclass`, `Age`, and `Sex_numeric` were selected as the features (`X`) for the model.
* The `Survived` column was set as the target (`y`).
* A `DecisionTreeClassifier` model from `scikit-learn` was trained on the `train.csv` data using the `.fit()` method.
* The model achieved an accuracy of **87.99%** on the training data.

### 4. Prediction and Submission
* The `test.csv` data was processed using the *exact same* cleaning and feature engineering steps.
* The trained model (`model.predict()`) was used to predict the survival outcomes for the passengers in `test.csv`.
* These predictions were saved to a `titanic_submission.csv` file in the format required by Kaggle.
* Upon submission, the model achieved a final public accuracy score of **72.72%**.
