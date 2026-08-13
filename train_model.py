"""Train and evaluate the V2 two-hand gesture classifier."""

import csv
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from feature_extractor import TOTAL_FEATURES


PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "landmarks_v2.csv"
MODEL_PATH = PROJECT_DIR / "models" / "gesture_classifier_v2.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20

METADATA_COLUMNS = ["label", "signer_id", "session_id"]
FEATURE_COLUMNS = [
    f"feature_{index}"
    for index in range(TOTAL_FEATURES)
]
EXPECTED_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS


def load_dataset():
    """Load and validate the V2 landmark dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run collect_data.py and collect at least two labels first."
        )

    features = []
    labels = []
    signer_ids = []
    session_ids = []

    with DATA_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("The dataset does not contain a CSV header.")

        missing_columns = [
            column
            for column in EXPECTED_COLUMNS
            if column not in reader.fieldnames
        ]

        if missing_columns:
            preview = ", ".join(missing_columns[:8])

            if len(missing_columns) > 8:
                preview += ", ..."

            raise ValueError(
                "landmarks_v2.csv has an incompatible structure.\n"
                f"Missing columns: {preview}\n"
                f"The V2 trainer expects {TOTAL_FEATURES} features."
            )

        for row_number, row in enumerate(reader, start=2):
            label = row["label"].strip().upper()
            signer_id = row["signer_id"].strip().upper()
            session_id = row["session_id"].strip()

            if not label:
                raise ValueError(f"Row {row_number}: label is empty.")

            if not signer_id:
                raise ValueError(f"Row {row_number}: signer_id is empty.")

            if not session_id:
                raise ValueError(f"Row {row_number}: session_id is empty.")

            try:
                sample = [
                    float(row[column])
                    for column in FEATURE_COLUMNS
                ]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Row {row_number}: one or more features are invalid."
                ) from error

            if not np.isfinite(sample).all():
                raise ValueError(
                    f"Row {row_number}: features contain NaN or infinity."
                )

            features.append(sample)
            labels.append(label)
            signer_ids.append(signer_id)
            session_ids.append(session_id)

    if not features:
        raise ValueError("The dataset contains no samples.")

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
        np.asarray(signer_ids),
        np.asarray(session_ids),
    )


def find_valid_group_split(X, y, groups):
    """
    Find a group-disjoint split that contains every label on both sides.

    A valid signer split means no signer occurs in both training and testing.
    A valid session split means no recording session occurs on both sides.
    """

    unique_groups = np.unique(groups)

    if len(unique_groups) < 2:
        return None

    required_labels = set(y.tolist())

    splitter = GroupShuffleSplit(
        n_splits=500,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    best_split = None
    best_score = None

    for train_indices, test_indices in splitter.split(X, y, groups):
        train_labels = set(y[train_indices].tolist())
        test_labels = set(y[test_indices].tolist())

        if train_labels != required_labels or test_labels != required_labels:
            continue

        test_fraction_error = abs(
            (len(test_indices) / len(y)) - TEST_SIZE
        )

        # Prefer a test split whose label proportions are close to the
        # complete dataset, after preferring a size near 20 percent.
        distribution_error = 0.0

        for label in required_labels:
            overall_ratio = np.mean(y == label)
            test_ratio = np.mean(y[test_indices] == label)
            distribution_error += abs(overall_ratio - test_ratio)

        score = (test_fraction_error, distribution_error)

        if best_score is None or score < best_score:
            best_score = score
            best_split = (train_indices, test_indices)

    return best_split


def make_random_split(X, y):
    """Create a stratified random fallback split for early development."""

    number_of_classes = len(np.unique(y))
    desired_test_count = max(
        number_of_classes,
        math.ceil(len(y) * TEST_SIZE),
    )

    # Keep at least one training sample for every class.
    maximum_test_count = len(y) - number_of_classes
    test_count = min(desired_test_count, maximum_test_count)

    if test_count < number_of_classes:
        raise ValueError(
            "There are not enough samples to create a stratified split. "
            "Collect more samples for every label."
        )

    return train_test_split(
        np.arange(len(y)),
        test_size=test_count,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def choose_split(X, y, signer_ids, session_ids):
    """Choose the strongest evaluation split supported by current data."""

    unique_signers = np.unique(signer_ids)

    if len(unique_signers) >= 2:
        split = find_valid_group_split(X, y, signer_ids)

        if split is not None:
            return (*split, "SIGNER_DISJOINT", True)

        print(
            "\nWARNING: A signer-disjoint split could not place every "
            "label in both training and testing."
        )

    # Include signer ID in the group key in case two people use the same
    # human-readable session name.
    recording_groups = np.asarray(
        [
            f"{signer_id}::{session_id}"
            for signer_id, session_id in zip(signer_ids, session_ids)
        ]
    )

    if len(np.unique(recording_groups)) >= 2:
        split = find_valid_group_split(X, y, recording_groups)

        if split is not None:
            return (*split, "SESSION_DISJOINT", True)

        print(
            "\nWARNING: A session-disjoint split could not place every "
            "label in both training and testing."
        )

    train_indices, test_indices = make_random_split(X, y)

    return (
        train_indices,
        test_indices,
        "RANDOM_FRAME_FALLBACK",
        False,
    )


def print_counter(title, values):
    """Print counts in a stable, readable order."""

    print(f"\n{title}:")

    for name, count in sorted(Counter(values).items()):
        print(f"  {name}: {count}")


def main():
    # -------------------------------------------------
    # 1. Load and inspect the dataset
    # -------------------------------------------------

    X, y, signer_ids, session_ids = load_dataset()

    label_counts = Counter(y)

    print("ISL Gesture Classifier V2")
    print("-------------------------")
    print(f"Dataset: {DATA_PATH}")
    print(f"Total samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of labels: {len(label_counts)}")
    print(f"Number of signers: {len(set(signer_ids))}")
    print(f"Number of sessions: {len(set(session_ids))}")

    print_counter("Samples per label", y)
    print_counter("Samples per signer", signer_ids)
    print_counter("Samples per session", session_ids)

    if X.shape[1] != TOTAL_FEATURES:
        raise ValueError(
            f"Expected {TOTAL_FEATURES} features, found {X.shape[1]}."
        )

    if len(label_counts) < 2:
        found_labels = ", ".join(sorted(label_counts))
        raise ValueError(
            "At least two different labels are required for training.\n"
            f"Currently found: {found_labels}\n"
            "Collect another test gesture using collect_data.py."
        )

    small_classes = {
        label: count
        for label, count in label_counts.items()
        if count < 2
    }

    if small_classes:
        raise ValueError(
            "Every label needs at least two samples. "
            f"Too small: {small_classes}"
        )

    # -------------------------------------------------
    # 2. Choose an evaluation split
    # -------------------------------------------------

    (
        train_indices,
        test_indices,
        split_strategy,
        leakage_safe_evaluation,
    ) = choose_split(X, y, signer_ids, session_ids)

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    print("\nEvaluation split")
    print("----------------")
    print(f"Strategy: {split_strategy}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    if split_strategy == "SIGNER_DISJOINT":
        print(
            "Training signers: "
            f"{sorted(set(signer_ids[train_indices]))}"
        )
        print(
            "Testing signers: "
            f"{sorted(set(signer_ids[test_indices]))}"
        )
    elif split_strategy == "SESSION_DISJOINT":
        train_groups = {
            f"{signer_ids[index]}::{session_ids[index]}"
            for index in train_indices
        }
        test_groups = {
            f"{signer_ids[index]}::{session_ids[index]}"
            for index in test_indices
        }

        print(f"Training recording groups: {sorted(train_groups)}")
        print(f"Testing recording groups: {sorted(test_groups)}")
    else:
        print(
            "WARNING: This split mixes neighboring frames. Its accuracy "
            "is only a pipeline check, not real-world model accuracy."
        )

    # -------------------------------------------------
    # 3. Train the classifier
    # -------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # -------------------------------------------------
    # 4. Evaluate the classifier
    # -------------------------------------------------

    predictions = model.predict(X_test)
    ordered_labels = sorted(model.classes_.tolist())

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=ordered_labels,
        average="macro",
        zero_division=0,
    )

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=ordered_labels,
            zero_division=0,
        )
    )

    print(f"Label order: {ordered_labels}")
    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=ordered_labels,
        )
    )

    # -------------------------------------------------
    # 5. Save model plus its feature/evaluation schema
    # -------------------------------------------------

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "model": model,
        "labels": ordered_labels,
        "feature_count": TOTAL_FEATURES,
        "feature_columns": FEATURE_COLUMNS,
        "schema_version": 2,
        "split_strategy": split_strategy,
        "leakage_safe_evaluation": leakage_safe_evaluation,
        "metrics": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        },
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "dataset_name": DATA_PATH.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(model_data, MODEL_PATH)

    print("\nModel saved successfully at:")
    print(MODEL_PATH)

    if not leakage_safe_evaluation:
        print(
            "\nIMPORTANT: Keep this model for development only. "
            "Collect complete labels from multiple signers so the next "
            "evaluation can use a signer-disjoint split."
        )


if __name__ == "__main__":
    main()