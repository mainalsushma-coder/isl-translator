import csv
from collections import Counter
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
from sklearn.model_selection import StratifiedGroupKFold


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "sequences_v3"
    / "manifest.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gesture_sequence_baseline_v3.joblib"
)

SEQUENCE_LENGTH = 48
FRAME_FEATURE_COUNT = 132
KEYFRAME_COUNT = 8
MIN_DETECTION_RATIO = 0.30
RANDOM_STATE = 42


# =========================================================
# Convert a sequence into motion-aware model features
# =========================================================

def sequence_to_model_features(sequence):
    """
    Convert one (48, 132) sequence into a fixed feature vector.

    The vector includes:

    - Eight ordered keyframes
    - Mean, standard deviation, minimum and maximum
    - First frame, last frame and overall displacement
    - Velocity statistics
    - Acceleration statistics
    """

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FRAME_FEATURE_COUNT,
    ):
        raise ValueError(
            f"Expected {(SEQUENCE_LENGTH, FRAME_FEATURE_COUNT)}, "
            f"received {sequence.shape}"
        )

    keyframe_indices = np.rint(
        np.linspace(
            0,
            SEQUENCE_LENGTH - 1,
            KEYFRAME_COUNT,
        )
    ).astype(int)

    keyframes = sequence[keyframe_indices].reshape(-1)

    sequence_mean = np.mean(sequence, axis=0)
    sequence_std = np.std(sequence, axis=0)
    sequence_min = np.min(sequence, axis=0)
    sequence_max = np.max(sequence, axis=0)

    first_frame = sequence[0]
    last_frame = sequence[-1]
    displacement = last_frame - first_frame

    velocity = np.diff(sequence, axis=0)

    velocity_mean = np.mean(velocity, axis=0)
    velocity_mean_absolute = np.mean(
        np.abs(velocity),
        axis=0,
    )
    velocity_std = np.std(velocity, axis=0)
    velocity_max_absolute = np.max(
        np.abs(velocity),
        axis=0,
    )

    acceleration = np.diff(velocity, axis=0)

    acceleration_mean_absolute = np.mean(
        np.abs(acceleration),
        axis=0,
    )
    acceleration_std = np.std(
        acceleration,
        axis=0,
    )
    acceleration_max_absolute = np.max(
        np.abs(acceleration),
        axis=0,
    )

    model_features = np.concatenate(
        [
            keyframes,
            sequence_mean,
            sequence_std,
            sequence_min,
            sequence_max,
            first_frame,
            last_frame,
            displacement,
            velocity_mean,
            velocity_mean_absolute,
            velocity_std,
            velocity_max_absolute,
            acceleration_mean_absolute,
            acceleration_std,
            acceleration_max_absolute,
        ]
    )

    return model_features.astype(np.float32)


# =========================================================
# Load generated sequence files
# =========================================================

def load_dataset():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    model_features = []
    labels = []
    signer_ids = []
    source_paths = []

    skipped_low_detection = 0
    skipped_invalid = 0

    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            detection_ratio = float(
                row["detection_ratio"]
            )

            if detection_ratio < MIN_DETECTION_RATIO:
                skipped_low_detection += 1
                continue

            sequence_path = (
                PROJECT_ROOT / row["sequence_path"]
            )

            if not sequence_path.exists():
                print(
                    f"WARNING: Missing sequence: "
                    f"{sequence_path}"
                )

                skipped_invalid += 1
                continue

            try:
                with np.load(
                    sequence_path,
                    allow_pickle=False,
                ) as data:
                    sequence = np.asarray(
                        data["sequence"],
                        dtype=np.float32,
                    )

                features = sequence_to_model_features(
                    sequence
                )

            except Exception as error:
                print(
                    f"WARNING: Could not load "
                    f"{sequence_path.name}: {error}"
                )

                skipped_invalid += 1
                continue

            model_features.append(features)
            labels.append(row["label"])
            signer_ids.append(row["signer_id"])
            source_paths.append(str(sequence_path))

    if not model_features:
        raise ValueError(
            "No valid sequence samples were found."
        )

    X = np.asarray(
        model_features,
        dtype=np.float32,
    )

    y = np.asarray(labels)
    groups = np.asarray(signer_ids)

    return (
        X,
        y,
        groups,
        source_paths,
        skipped_low_detection,
        skipped_invalid,
    )


# =========================================================
# Main training pipeline
# =========================================================

def main():
    print("ISL Sequence Classifier V3")
    print("--------------------------")
    print(f"Manifest: {MANIFEST_PATH}")

    (
        X,
        y,
        groups,
        source_paths,
        skipped_low_detection,
        skipped_invalid,
    ) = load_dataset()

    unique_labels = sorted(set(y))
    unique_signers = sorted(set(groups))

    print(f"Total usable sequences: {len(X)}")
    print(f"Frames per sequence: {SEQUENCE_LENGTH}")
    print(
        f"Features per frame: {FRAME_FEATURE_COUNT}"
    )
    print(
        f"Model features per sequence: {X.shape[1]}"
    )
    print(f"Number of labels: {len(unique_labels)}")
    print(f"Number of signers: {len(unique_signers)}")
    print(
        f"Skipped for low detection: "
        f"{skipped_low_detection}"
    )
    print(
        f"Skipped because invalid: {skipped_invalid}"
    )

    print("\nSequences per label:")

    label_counts = Counter(y)

    for label in unique_labels:
        print(f"  {label}: {label_counts[label]}")

    if len(unique_labels) < 2:
        raise ValueError(
            "At least two labels are required."
        )

    if len(unique_signers) < 5:
        raise ValueError(
            "At least five signers are required "
            "for a signer-disjoint evaluation."
        )

    # -----------------------------------------------------
    # Signer-disjoint train/test split
    # -----------------------------------------------------

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    groups_train = groups[train_indices]
    groups_test = groups[test_indices]

    train_signers = sorted(set(groups_train))
    test_signers = sorted(set(groups_test))

    overlap = set(train_signers).intersection(
        test_signers
    )

    if overlap:
        raise RuntimeError(
            f"Signer leakage detected: {overlap}"
        )

    missing_train_labels = (
        set(unique_labels) - set(y_train)
    )

    missing_test_labels = (
        set(unique_labels) - set(y_test)
    )

    if missing_train_labels or missing_test_labels:
        raise RuntimeError(
            "The split did not preserve every label.\n"
            f"Missing from training: "
            f"{sorted(missing_train_labels)}\n"
            f"Missing from testing: "
            f"{sorted(missing_test_labels)}"
        )

    print("\nSigner-disjoint evaluation")
    print("--------------------------")
    print(f"Training sequences: {len(X_train)}")
    print(f"Testing sequences: {len(X_test)}")
    print(f"Training signers: {len(train_signers)}")
    print(f"Testing signers: {len(test_signers)}")
    print(f"Signer overlap: {len(overlap)}")

    print("\nHeld-out testing signers:")

    for signer_id in test_signers:
        print(f"  {signer_id}")

    # -----------------------------------------------------
    # Train the classifier
    # -----------------------------------------------------

    classifier = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
        max_features="sqrt",
    )

    print("\nTraining motion-aware baseline...")

    classifier.fit(
        X_train,
        y_train,
    )

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=unique_labels,
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=unique_labels,
    )

    print("Label order:")
    print(unique_labels)

    print("\nConfusion matrix:")
    print(matrix)

    # -----------------------------------------------------
    # Save model and preprocessing information
    # -----------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_data = {
        "model": classifier,
        "labels": unique_labels,
        "sequence_length": SEQUENCE_LENGTH,
        "frame_feature_count": FRAME_FEATURE_COUNT,
        "model_feature_count": X.shape[1],
        "keyframe_count": KEYFRAME_COUNT,
        "feature_transform": (
            "TEMPORAL_SUMMARY_V1"
        ),
        "minimum_detection_ratio": (
            MIN_DETECTION_RATIO
        ),
        "training_signers": train_signers,
        "testing_signers": test_signers,
        "training_sample_count": len(X_train),
        "testing_sample_count": len(X_test),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "source": "MENDELEY_EMERGENCY_ISL",
    }

    joblib.dump(
        model_data,
        MODEL_PATH,
        compress=3,
    )

    print("\nModel saved successfully at:")
    print(MODEL_PATH)

    print(
        "\nIMPORTANT: This is a signer-disjoint "
        "dataset baseline. We must still test it "
        "using our live webcam and verify the ISL "
        "variants before presenting it as an "
        "official ISL translator."
    )


if __name__ == "__main__":
    main()