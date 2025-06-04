# Power System Load Type Classification Project

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')

# 2. Load data

# Update this path if needed!
file_path = 'load_data.csv'

try:
    df = pd.read_csv(file_path)
    print(f"File '{file_path}' loaded successfully.")
except FileNotFoundError:
    print(f"File '{file_path}' not found. Please check the path and file name.")

# 3. Inspect data

print("\nColumns in dataset:")
print(df.columns)

# Remove leading/trailing spaces in column names if any
df.columns = df.columns.str.strip()

print("\nColumns after stripping spaces:")
print(df.columns)

print("\nFirst 5 rows of data:")
display(df.head())

# 4. Data Preprocessing

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing values if any (using forward fill here)
df.fillna(method='ffill', inplace=True)

# Convert 'Date' column to datetime format (make sure it exists)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
else:
    print("Warning: 'Date' column not found in data.")

# Sort data by Date if available
if 'Date' in df.columns:
    df = df.sort_values(by='Date').reset_index(drop=True)

# 5. Exploratory Data Analysis (EDA)

# Check distribution of target variable 'Load_Type'
if 'Load_Type' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='Load_Type')
    plt.title('Load Type Distribution')
    plt.show()
else:
    print("Warning: 'Load_Type' column not found.")

# Show summary statistics
print("\nSummary statistics:")
display(df.describe())

# Pairplot for numeric features if Load_Type exists
if 'Load_Type' in df.columns:
    features = ['Usage_kWh', 'Lagging Current reactive power', 'Leading Current reactive power', 'CO2', 'NSM']
    # Filter features that exist in df.columns
    features = [f for f in features if f in df.columns]

    if features:
        sns.pairplot(df, hue='Load_Type', vars=features)
        plt.show()
else:
    print("Skipping pairplot as 'Load_Type' column not found.")

# Correlation matrix heatmap for numeric features
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if numeric_cols:
    plt.figure(figsize=(8,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.show()

# 6. Feature Engineering

# Extract month from Date (if Date exists)
if 'Date' in df.columns:
    df['Month'] = df['Date'].dt.month
else:
    # If no Date, create Month as NaN or drop this step
    print("Skipping month extraction as 'Date' not found.")

# Drop 'Date' column (optional)
if 'Date' in df.columns:
    df.drop('Date', axis=1, inplace=True)

# 7. Encode target variable

if 'Load_Type' in df.columns:
    le = LabelEncoder()
    df['Load_Type_Encoded'] = le.fit_transform(df['Load_Type'])
    print("\nLoad Type classes and encoded values:")
    for i, cls in enumerate(le.classes_):
        print(f"{cls} --> {i}")
else:
    print("Target 'Load_Type' not found, cannot encode.")

# 8. Train-test split based on Month (last month as test set)

if 'Month' in df.columns and 'Load_Type_Encoded' in df.columns:
    last_month = df['Month'].max()
    train_data = df[df['Month'] != last_month]
    test_data = df[df['Month'] == last_month]

    X_train = train_data.drop(['Load_Type', 'Load_Type_Encoded'], axis=1)
    y_train = train_data['Load_Type_Encoded']

    X_test = test_data.drop(['Load_Type', 'Load_Type_Encoded'], axis=1)
    y_test = test_data['Load_Type_Encoded']

    print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # 9. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 10. Model Training (Random Forest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # 11. Model Evaluation
    y_pred = model.predict(X_test_scaled)

    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Feature importance plot
    feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
    plt.figure(figsize=(8,5))
    feat_imp.sort_values().plot(kind='barh')
    plt.title('Feature Importance')
    plt.show()

else:
    print("Train-test split cannot be done due to missing 'Month' or 'Load_Type_Encoded' columns.")
