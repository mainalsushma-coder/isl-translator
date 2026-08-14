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

from train_sequence_model import (
    FRAME_FEATURE_COUNT,
    KEYFRAME_COUNT,
    MIN_DETECTION_RATIO,
    RANDOM_STATE,
    SEQUENCE_LENGTH,
    sequence_to_model_features,
)


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

BASELINE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gesture_sequence_baseline_v3.joblib"
)

ADAPTED_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gesture_sequence_adapted_v3.joblib"
)

EXTERNAL_SOURCE = "MENDELEY_EMERGENCY_ISL"
ADAPTATION_SOURCE = "WEBCAM_ADAPTATION"

# Give each adaptation clip more importance without
# duplicating the file multiple times.
ADAPTATION_SAMPLE_WEIGHT = 5.0


# =========================================================
# Load every sequence
# =========================================================

def load_all_sequences():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    features = []
    labels = []
    signers = []
    sources = []
    paths = []

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
                continue

            sequence_path = (
                PROJECT_ROOT / row["sequence_path"]
            )

            if not sequence_path.exists():
                print(
                    f"WARNING: Missing sequence: "
                    f"{sequence_path}"
                )
                continue

            with np.load(
                sequence_path,
                allow_pickle=False,
            ) as data:
                sequence = np.asarray(
                    data["sequence"],
                    dtype=np.float32,
                )

            model_features = (
                sequence_to_model_features(sequence)
            )

            features.append(model_features)
            labels.append(row["label"])
            signers.append(row["signer_id"])
            sources.append(row["source"])
            paths.append(str(sequence_path))

    if not features:
        raise ValueError(
            "No usable sequences were found."
        )

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
        np.asarray(signers),
        np.asarray(sources),
        paths,
    )


# =========================================================
# Main training pipeline
# =========================================================

def main():
    print("ISL Adapted Sequence Classifier V3")
    print("----------------------------------")

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "The baseline model is required so that "
            "we can reuse exactly the same held-out "
            f"testing signers:\n{BASELINE_MODEL_PATH}"
        )

    baseline_data = joblib.load(
        BASELINE_MODEL_PATH
    )

    held_out_signers = set(
        baseline_data["testing_signers"]
    )

    (
        X,
        y,
        groups,
        sources,
        paths,
    ) = load_all_sequences()

    external_mask = (
        sources == EXTERNAL_SOURCE
    )

    adaptation_mask = (
        sources == ADAPTATION_SOURCE
    )

    external_train_mask = (
        external_mask
        & ~np.isin(
            groups,
            list(held_out_signers),
        )
    )

    external_test_mask = (
        external_mask
        & np.isin(
            groups,
            list(held_out_signers),
        )
    )

    if not np.any(adaptation_mask):
        raise ValueError(
            "No WEBCAM_ADAPTATION samples were found."
        )

    X_external_train = X[external_train_mask]
    y_external_train = y[external_train_mask]
    groups_external_train = groups[
        external_train_mask
    ]

    X_adaptation = X[adaptation_mask]
    y_adaptation = y[adaptation_mask]
    groups_adaptation = groups[
        adaptation_mask
    ]

    X_test = X[external_test_mask]
    y_test = y[external_test_mask]
    groups_test = groups[external_test_mask]

    X_train = np.concatenate(
        [
            X_external_train,
            X_adaptation,
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            y_external_train,
            y_adaptation,
        ],
        axis=0,
    )

    groups_train = np.concatenate(
        [
            groups_external_train,
            groups_adaptation,
        ],
        axis=0,
    )

    sample_weights = np.concatenate(
        [
            np.ones(
                len(X_external_train),
                dtype=np.float32,
            ),
            np.full(
                len(X_adaptation),
                ADAPTATION_SAMPLE_WEIGHT,
                dtype=np.float32,
            ),
        ]
    )

    unique_labels = sorted(set(y))
    adaptation_signers = sorted(
        set(groups_adaptation)
    )

    signer_overlap = (
        set(groups_train)
        .intersection(set(groups_test))
    )

    if signer_overlap:
        raise RuntimeError(
            f"Signer leakage detected: "
            f"{sorted(signer_overlap)}"
        )

    missing_train_labels = (
        set(unique_labels) - set(y_train)
    )

    missing_test_labels = (
        set(unique_labels) - set(y_test)
    )

    if missing_train_labels or missing_test_labels:
        raise RuntimeError(
            "Every label must be present in training "
            "and testing.\n"
            f"Missing from training: "
            f"{sorted(missing_train_labels)}\n"
            f"Missing from testing: "
            f"{sorted(missing_test_labels)}"
        )

    print(f"All usable sequences: {len(X)}")
    print(
        f"External training sequences: "
        f"{len(X_external_train)}"
    )
    print(
        f"Webcam adaptation sequences: "
        f"{len(X_adaptation)}"
    )
    print(
        f"External testing sequences: "
        f"{len(X_test)}"
    )
    print(
        f"Features per sequence: {X.shape[1]}"
    )
    print(
        f"Adaptation sample weight: "
        f"{ADAPTATION_SAMPLE_WEIGHT}"
    )
    print(f"Signer overlap: {len(signer_overlap)}")

    print("\nAdaptation samples per label:")

    adaptation_counts = Counter(y_adaptation)

    for label in sorted(adaptation_counts):
        print(
            f"  {label}: "
            f"{adaptation_counts[label]}"
        )

    print("\nAdaptation signers:")

    for signer in adaptation_signers:
        print(f"  {signer}")

    print("\nHeld-out external testing signers:")

    for signer in sorted(held_out_signers):
        print(f"  {signer}")

    # -----------------------------------------------------
    # Train adapted classifier
    # -----------------------------------------------------

    classifier = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
        max_features="sqrt",
    )

    print("\nTraining adapted model...")

    classifier.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    # -----------------------------------------------------
    # Evaluate only on untouched external signers
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

    print(
        f"\nOriginal baseline accuracy: "
        f"{baseline_data['accuracy'] * 100:.2f}%"
    )

    print(
        f"Adapted external-test accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Adapted external-test Macro F1: "
        f"{macro_f1:.4f}"
    )

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
    # Save as a separate model
    # -----------------------------------------------------

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
        "training_signers": sorted(
            set(groups_train)
        ),
        "testing_signers": sorted(
            held_out_signers
        ),
        "adaptation_signers": (
            adaptation_signers
        ),
        "external_training_sample_count": (
            len(X_external_train)
        ),
        "adaptation_sample_count": (
            len(X_adaptation)
        ),
        "testing_sample_count": len(X_test),
        "adaptation_sample_weight": (
            ADAPTATION_SAMPLE_WEIGHT
        ),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "source": (
            "MENDELEY_EMERGENCY_ISL_PLUS_"
            "WEBCAM_ADAPTATION"
        ),
    }

    ADAPTED_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_data,
        ADAPTED_MODEL_PATH,
        compress=3,
    )

    print("\nAdapted model saved at:")
    print(ADAPTED_MODEL_PATH)

    print(
        "\nThe external test set remained untouched. "
        "The webcam clips were used only for training."
    )


if __name__ == "__main__":
    main()