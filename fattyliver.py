import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Sample dataset generation (replace with your actual data loading)
np.random.seed(42)
data = {
    'Age': np.random.randint(20, 70, 100),
    'Gender': np.random.randint(0, 2, 100),  # 0 for female, 1 for male
    'Height': np.random.uniform(150, 200, 100),  # Height in cm
    'Weight': np.random.uniform(50, 120, 100),   # Weight in kg
    'BMI': np.random.uniform(18.5, 35.0, 100),
    'Systolic_BP': np.random.randint(90, 180, 100),
    'Diastolic_BP': np.random.randint(60, 120, 100),
    'Hypertension': np.random.randint(0, 2, 100),
    'Hyperlipidemia': np.random.randint(0, 2, 100),
    'Smoking_Status': np.random.randint(0, 2, 100),
    'Diabetes_Mellitus': np.random.randint(0, 2, 100),  # Binary label for diabetes
    'Metabolic_Syndrome': np.random.randint(0, 2, 100), # Binary label for metabolic syndrome
    'Leukocytes': np.random.uniform(4.0, 11.0, 100),    # Leukocytes count in 10^9/L
    'Hemoglobin': np.random.uniform(12.0, 18.0, 100),   # Hemoglobin in g/dL
    'Total_Cholesterol': np.random.uniform(150, 250, 100), # Total cholesterol in mg/dL
    'Glucose': np.random.uniform(70, 200, 100),         # Glucose level in mg/dL
    'Insulin': np.random.uniform(2.0, 25.0, 100),       # Insulin level in µIU/mL
    'FLD_label': np.random.randint(0, 2, 100)           # Binary label for fatty liver disease
}
df = pd.DataFrame(data)

# Display first few rows and inspect columns
print(df.head())

# Split dataset into features (X) and target (y)
X = df.drop('FLD_label', axis=1)
y = df['FLD_label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to train
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {name}: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print(f'Confusion Matrix of {name}:')
    print(confusion_matrix(y_test, y_pred))
    
    # ROC Curve
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc='lower right')
    plt.show()