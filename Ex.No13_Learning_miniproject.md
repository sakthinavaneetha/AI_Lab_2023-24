# Ex.No: 10 Autism Detection: A Data-Driven Solution
### DATE: 24.10.2042                                                                  
### REGISTER NUMBER : 212222040138
### AIM: 

To write a program to train the classifier for Autism Spectrum Disorder (ASD) prediction.

###  Algorithm:
Data Preprocessing: Load and clean the data, handling any missing or inconsistent entries.
Feature Selection: Identify critical features for ASD prediction.
Model Training: Train a classifier using TensorFlow or PyTorch for deep learning or scikit-learn for traditional ML.
Model Evaluation: Calculate accuracy, precision, recall, and F1-score to evaluate model performance.
Model Explainability: Use SHAP or LIME to interpret predictions.
Save Model: Save the trained model with joblib.

### Program:

```
python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')

data_path = '/content/drive/MyDrive/combined_autism_dataset_with_genetic_data.csv'
data_fin = pd.read_csv(data_path)

data_fin.head()
print(data_fin.info())

missing_values = pd.DataFrame(data_fin.isnull().sum(), columns=["Missing Values"])
print(missing_values)

X = data_fin.drop("target_column", axis=1)
y = data_fin["target_column"]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.23, random_state=45)

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

model = LogisticRegression()
results = train_model(model, x_train, y_train, x_test, y_test)
print("Logistic Regression Results:\n", results)

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'n_estimators': [10, 50, 100, 200]
}
grid_search = GridSearchCV(estimator=rf, param_grid=params, cv=4, n_jobs=-1, verbose=1, scoring="accuracy")
grid_search.fit(x_train, y_train)

rf_best = grid_search.best_estimator_
print("Best Random Forest Model:", rf_best)
print("Best Score from Grid Search:", grid_search.best_score_)

y_pred = rf_best.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
nn_model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "KNN": KNeighborsClassifier()
}

results_list = []
for model_name, model in models.items():
    accuracies = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
    results_list.append({
        "Model": model_name,
        "Accuracy": accuracies.mean()
    })

results = pd.DataFrame(results_list)
print(results)
```




### Output:
![image](https://github.com/user-attachments/assets/575d3f27-6e21-4d97-9685-11c7d1453c64)

![image](https://github.com/user-attachments/assets/a0005982-7de3-4068-91b9-0fb12bb5613d)


Detection Accuracy: 96.7% 
### Result:
Thus the system was trained successfully and the prediction was carried out.
