import csv
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).parent / "data" / "landmarks.csv"
MODEL_PATH = Path(__file__).parent / "models" / "gesture_classifier.joblib"


# -------------------------------------------------
# 1. Load the dataset
# -------------------------------------------------

features = []
labels = []

with DATA_PATH.open("r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        labels.append(row["label"])

        sample = [
            float(row[f"feature_{index}"])
            for index in range(63)
        ]

        features.append(sample)


X = np.array(features)
y = np.array(labels)

print(f"Total samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Labels: {Counter(y)}")


if len(set(y)) < 2:
    raise ValueError(
        "At least two different gesture labels are required."
    )


# -------------------------------------------------
# 2. Split into training and testing data
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# -------------------------------------------------
# 3. Train the Random Forest model
# -------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train, y_train)


# -------------------------------------------------
# 4. Evaluate the model
# -------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))


# -------------------------------------------------
# 5. Save the trained model
# -------------------------------------------------

MODEL_PATH.parent.mkdir(exist_ok=True)

model_data = {
    "model": model,
    "labels": list(model.classes_),
    "feature_count": 63
}

joblib.dump(model_data, MODEL_PATH)

print(f"\nModel saved successfully at:")
print(MODEL_PATH)
