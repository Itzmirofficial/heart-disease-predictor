import pandas as pd

# Define column names from UCI documentation
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'target'
]

# Load the dataset
df = pd.read_csv('processed.cleveland.data', names=column_names)

# Show first 5 rows
print(df.head())

# Check for missing values
print("\nMissing values (marked with '?'):")
print(df[df.eq('?').any(axis=1)])

# Show dataset info
print("\nInfo:")
print(df.info())
# Replace '?' with NaN and convert numeric columns
df.replace('?', pd.NA, inplace=True)

# Convert all columns to float
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values (simplest approach for now)
df.dropna(inplace=True)

print(f"\nCleaned Data Shape: {df.shape}")
print(df.head())
print(df['target'].value_counts())
# Convert to binary classification
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Show accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
import seaborn as sns

# Plot feature importances
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()
