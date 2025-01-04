# **Loan Prediction**

This repository contains a Jupyter Notebook that demonstrates how to predict loan approval status using machine learning techniques. The dataset used in this project provides information about loan applicants and their associated features.

---

## **Overview**

Predicting loan approval is a critical task for financial institutions to assess the eligibility of applicants. This project uses machine learning to classify whether a loan will be approved or not based on applicant details such as income, credit history, and property area. The model implemented in this notebook is based on **Support Vector Machines (SVM)**.

The dataset includes features such as applicant income, co-applicant income, loan amount, credit history, and more. The target variable (`Loan_Status`) indicates whether the loan was approved (`1`) or not (`0`).

---

## **Dataset**

- **Source**: The dataset appears to be related to publicly available loan prediction datasets.
- **Features**:
  - `Loan_ID`: Unique identifier for each loan application.
  - `Gender`: Gender of the applicant (1 = Male, 0 = Female).
  - `Married`: Marital status of the applicant (1 = Married, 0 = Not Married).
  - `Dependents`: Number of dependents.
  - `Education`: Education level (1 = Graduate, 0 = Not Graduate).
  - `Self_Employed`: Employment status (1 = Self-employed, 0 = Not Self-employed).
  - `ApplicantIncome`: Monthly income of the applicant.
  - `CoapplicantIncome`: Monthly income of the co-applicant.
  - `LoanAmount`: Loan amount requested.
  - `Loan_Amount_Term`: Loan repayment term in months.
  - `Credit_History`: Credit history (1.0 = Good credit history, 0.0 = Bad credit history).
  - `Property_Area`: Property location (e.g., Urban, Semiurban, Rural).
- **Target Variable**:
  - `Loan_Status`: Indicates whether the loan was approved (1) or not (0).

---

## **Project Workflow**

1. **Data Loading**:
   - The dataset (`LoanData.csv`) is loaded into a Pandas DataFrame.
2. **Exploratory Data Analysis (EDA)**:
   - Summary statistics and visualizations are used to understand the dataset.
3. **Data Preprocessing**:
   - Missing values are handled appropriately.
   - Categorical variables are encoded into numerical values for machine learning models.
4. **Model Training**:
   - Support Vector Machine (SVM) with a linear kernel is used as the classification model.
   - The dataset is split into training and testing sets using `train_test_split`.
5. **Model Evaluation**:
   - Accuracy score is calculated to evaluate the model's performance.

---

## **Dependencies**

To run this project, you need the following Python libraries:

- numpy
- pandas
- seaborn
- scikit-learn

You can install these dependencies using pip:

```bash
pip install numpy pandas seaborn scikit-learn
```

---

## **How to Run**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/LoanPrediction.git
   cd LoanPrediction
   ```

2. Ensure that the dataset file (`LoanData.csv`) is in the same directory as the notebook.

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Loan-Pred.ipynb
   ```

4. Run all cells in the notebook to execute the code.

---

## **Results**

The Support Vector Machine (SVM) model provides an accuracy score that indicates its performance in predicting loan approval status. Further improvements can be made by experimenting with other machine learning models or feature engineering techniques.

---

## **Acknowledgments**

- The dataset was sourced from publicly available loan prediction datasets or competitions.
- Special thanks to Scikit-learn for providing robust machine learning tools.

---
