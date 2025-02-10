from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the dataset
iris = load_iris()

# Features (input data)
X = iris.data

# Target (output data)
Y = iris.target

feature_names = iris.feature_names  # Corrected
target_names = iris.target_names

print("Features\n", feature_names)  # Corrected
print("Target Names\n", target_names)
print("First 5 samples:\n", X[:5])  # Corrected

# Create DataFrame
data = pd.DataFrame(iris.data, columns=feature_names)

# Check for missing values
print("Missing values per feature:\n", data.isnull().sum())  # Corrected

# Check for duplicates
print("Number of duplicate rows:", data.duplicated().sum())

# Remove duplicates
data.drop_duplicates(inplace=True)

# Re-check for missing values
print("Missing values per feature:\n", data.isnull().sum())  # Corrected

# Re-check for duplicates
print("Number of duplicate rows:", data.duplicated().sum())

#! Data Preprocessing -> Standardize Features
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
# Dit the scaler on the training data and transform both training and testing sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFirst 5 rows of scaled training data:\n", X_train_scaled[:5])

#! Step 3: Train and evaluate the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

# Make predictions on the test data
Y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"\nAccuracy : {accuracy * 100:.2f}%")

# Generate a classification report
print("\nClassification Report:\n", classification_report(Y_test, Y_pred, target_names=target_names))

cm = confusion_matrix(Y_test, Y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()