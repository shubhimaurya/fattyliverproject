# Fatty Liver Disease Prediction
This project uses machine learning models to predict fatty liver disease (FLD) based on various health metrics such as age, gender, BMI, blood pressure, glucose levels, and more. The project explores Logistic Regression and Random Forest classifiers, and evaluates their performance using accuracy, confusion matrix, classification report, and ROC curve.
## Project Structure
Data Generation: A synthetic dataset is created to simulate real-world health data. Replace this with actual data loading for real-world use.
Data Preprocessing: Features are standardized using StandardScaler to improve model performance.
Model Training: Two models—Logistic Regression and Random Forest—are trained using the preprocessed data.
Model Evaluation: Models are evaluated based on:
Accuracy score
Confusion matrix
Classification report
ROC curve and AUC score
## Installation
To run this project, you need to have Python and the following packages installed:
``` pip install numpy pandas scikit-learn matplotlib ```
## Dataset
A synthetic dataset is generated with the following features:
- Age (years)
- Gender (binary: 0 for female, 1 for male)
- Height (in cm)
- Weight (in kg)
- BMI (Body Mass Index)
- Systolic_BP (Systolic Blood Pressure)
- Diastolic_BP (Diastolic Blood Pressure)
- Hypertension (binary: 0 for no, 1 for yes)
- Hyperlipidemia (binary: 0 for no, 1 for yes)
- Smoking_Status (binary: 0 for non-smoker, 1 for smoker)
- Diabetes_Mellitus (binary: 0 for no, 1 for yes)
- Metabolic_Syndrome (binary: 0 for no, 1 for yes)
- Leukocytes (white blood cell count in 10^9/L)
- Hemoglobin (in g/dL)
- Total_Cholesterol (in mg/dL)
- Glucose (in mg/dL)
- Insulin (in µIU/mL)
- FLD_label (binary label for fatty liver disease: 0 for no, 1 for yes)
## Models Used
### 1. Logistic Regression
Logistic Regression is a simple but effective linear model used for binary classification.

### 2. Random Forest Classifier
Random Forest is an ensemble method that combines multiple decision trees to improve prediction accuracy.
