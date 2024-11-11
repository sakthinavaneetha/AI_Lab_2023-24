# Ex.No: 10 Autism Detection: A Data-Driven Solution
### DATE: 11.11.2042                                                                  
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

Open In Colab

from google.colab import drive
drive.mount('/content/drive')
     
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).

Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings

Ignore warnings for a clean output
warnings.filterwarnings('ignore')

     

Load the dataset from Google Drive
data_path = '/content/drive/MyDrive/combined_autism_dataset_with_genetic_data.csv'
data_fin = pd.read_csv(data_path)

Display the first few rows of the dataset
data_fin.head()


Display information about dataset columns and data types
print(data_fin.info())

Check for missing values
missing_values = pd.DataFrame(data_fin.isnull().sum(), columns=["Missing Values"])
print(missing_values)


Display column names to verify their actual names in the dataset
print(data_fin.columns)

Define the list of expected categorical columns based on available column names
categorical_cols = [col for col in ['Ethnicity', 'Who_completed_the_test', 'Sex', 'Jaundice', 'Family_mem_with_ASD', 'ASD_traits'] if col in data_fin.columns]

Fill missing values for categorical columns with the mode, only if they exist in the dataset
for col in categorical_cols:
    data_fin[col].fillna(data_fin[col].mode()[0], inplace=True)

For numerical columns, fill missing values with the mean
numerical_cols = data_fin.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    data_fin[col].fillna(data_fin[col].mean(), inplace=True)

Verify that there are no missing values left
print(data_fin.isnull().sum())

Define replacements for categorical columns to standardize values
replacements = {
    'Sex': {'f': 'F', 'm': 'M'},
    'Jaundice': {'yes': 'Yes', 'no': 'No'},
    'Family_mem_with_ASD': {'yes': 'Yes', 'no': 'No'},
    'ASD_traits': {'YES': 'Yes', 'NO': 'No'},
    'Ethnicity': {
        'middle eastern': 'Middle Eastern', 'Middle Eastern ': 'Middle Eastern',
        'mixed': 'Mixed', 'asian': 'Asian', 'black': 'Black',
        'south asian': 'South Asian', 'PaciFica':'Pacifica', 'Pasifika':'Pacifica'
    },
    'Who_completed_the_test': {
        'Health care professional':'Health Care Professional',
        'family member':'Family Member', 'Family member':'Family Member'
    }
}

Apply replacements to each column, if the column exists in the dataset
for col, replace_dict in replacements.items():
    if col in data_fin.columns:
        data_fin[col] = data_fin[col].replace(replace_dict)

Initialize LabelEncoder and encode categorical columns
This part checks that the column exists before applying encoding
le = LabelEncoder()
for col in categorical_cols:
    if col in data_fin.columns:
        data_fin[col] = le.fit_transform(data_fin[col])

Display the first few rows to verify changes
data_fin.head()


 Redefine numerical columns by selecting only columns of type int or float
numerical_cols = data_fin.select_dtypes(include=['float64', 'int64']).columns

 Plot boxplots for numerical columns to inspect potential outliers
plt.figure(figsize=(15, 10))
data_fin[numerical_cols].boxplot()  # Use only the numerical columns
plt.xticks(rotation=90)
plt.show()

     


print(data_fin.columns)
     


print(data_fin.columns)
     

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.23, random_state=45)

     

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Change to 'weighted' or 'macro' as appropriate
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Create a DataFrame to display metrics
    score_df = pd.DataFrame([[accuracy, precision, recall, f1]],
                            columns=["Accuracy", "Precision", "Recall", "F1 Score"])
    return score_df

Example with Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
results = train_model(model, x_train, y_train, x_test, y_test)
print("Logistic Regression Results:\n", results)

     
Logistic Regression Results:
    Accuracy  Precision    Recall  F1 Score
0  0.997531   0.997591  0.997531  0.997541

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

Define Random Forest and parameter grid for tuning
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'n_estimators': [10, 50, 100, 200]
}

Grid search for best parameters
grid_search = GridSearchCV(estimator=rf, param_grid=params, cv=4, n_jobs=-1, verbose=1, scoring="accuracy")
grid_search.fit(x_train, y_train)

Retrieve the best model from grid search
rf_best = grid_search.best_estimator_
print("Best Random Forest Model:", rf_best)
print("Best Score from Grid Search:", grid_search.best_score_)

     
Fitting 4 folds for each of 100 candidates, totalling 400 fits
Best Random Forest Model: RandomForestClassifier(max_depth=5, min_samples_leaf=5, n_estimators=50,
                       n_jobs=-1, random_state=42)
Best Score from Grid Search: 1.0

from sklearn.metrics import classification_report, confusion_matrix

Predict and evaluate using the best model
y_pred = rf_best.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Assuming y_train and y_test are your target variables
# Convert labels to binary if they're in string format
y_train = np.where(y_train == "YES", 1, 0)
y_test = np.where(y_test == "YES", 1, 0)

Define a neural network model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Single output for binary classification
])

 Compile the model
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

 Train the neural network model
history = nn_model.fit(x_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping], batch_size=32)

 Evaluate neural network model
nn_eval = nn_model.evaluate(x_test, y_test)
print(f"Neural Network Accuracy: {nn_eval[1]:.2f}")




from sklearn.svm import SVC

Initialize the model
svm_model = SVC(kernel='linear', random_state=42)

Fit the model
svm_model.fit(x_train, y_train)

Make predictions
y_pred = svm_model.predict(x_test)

Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))


import xgboost as xgb

Initialize the model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

Fit the model
xgb_model.fit(x_train, y_train)

Make predictions
y_pred = xgb_model.predict(x_test)

 Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))


     


#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

     

from sklearn.model_selection import cross_val_score

Example for Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf_model, x_train, y_train, cv=5)
print(f"Random Forest CV Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

     
Random Forest CV Accuracy: 1.00 ± 0.00

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),  # Set probability=True for ROC AUC if needed
    "XGBoost": XGBClassifier(),
    "KNN": KNeighborsClassifier()
}

Dictionary to hold metrics
metrics = {}

Train and evaluate each model
for model_name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Store metrics
    metrics[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }


     

Convert metrics dictionary to DataFrame
results = pd.DataFrame(metrics).T  # Transpose to get models as rows
print(results)


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

 Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'KNN': KNeighborsClassifier()
}

Initialize a list to hold results
results_list = []

Evaluate each model using cross-validation
for model_name, model in models.items():
    Cross-validated accuracy
    accuracies = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
    accuracy = accuracies.mean()

    # Fit the model and predict to get other metrics
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Append results to the list
    results_list.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

Create a DataFrame from the results list
results = pd.DataFrame(results_list)

Display the results
print(results)




### Output:
![image](https://github.com/user-attachments/assets/575d3f27-6e21-4d97-9685-11c7d1453c64)

![image](https://github.com/user-attachments/assets/a0005982-7de3-4068-91b9-0fb12bb5613d)
Detection Accuracy: 96.7% Note: These metrics can be customized based on your actual performance evaluations.

### Result:
Thus the system was trained successfully and the prediction was carried out.
