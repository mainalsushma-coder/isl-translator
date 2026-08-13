"""
Shared feature extraction functions.

Both collect_data.py and predict_live.py will use this file.
This ensures that training and live prediction calculate
features in exactly the same way.
"""

NUM_LANDMARKS = 21
COORDINATES_PER_LANDMARK = 3

FEATURES_PER_HAND = NUM_LANDMARKS * COORDINATES_PER_LANDMARK

# Left hand: 63
# Right hand: 63
# Presence values: 2
TOTAL_FEATURES = (FEATURES_PER_HAND * 2) + 2


def get_handedness_name(handedness_categories):
    """
    Return 'Left' or 'Right' from MediaPipe handedness output.
    """

    if not handedness_categories:
        return None

    category = handedness_categories[0]

    name = category.category_name

    if not name:
        return None

    name = name.capitalize()

    if name not in {"Left", "Right"}:
        return None

    return name


def split_hands(result):
    """
    Put detected hands into fixed Left and Right positions.

    Returns:
        {
            "Left": landmarks or None,
            "Right": landmarks or None
        }
    """

    hands = {
        "Left": None,
        "Right": None
    }

    unresolved_hands = []

    detected_landmarks = result.hand_landmarks or []
    detected_handedness = result.handedness or []

    for index, landmarks in enumerate(detected_landmarks):

        if index < len(detected_handedness):
            hand_name = get_handedness_name(
                detected_handedness[index]
            )
        else:
            hand_name = None

        # MediaPipe handedness is reversed for our mirrored webcam frame
        if hand_name == "Left":
            hand_name = "Right"
        elif hand_name == "Right":
            hand_name = "Left"

        if hand_name and hands[hand_name] is None:
            hands[hand_name] = landmarks
        else:
            unresolved_hands.append(landmarks)

    # Fallback in case MediaPipe does not return handedness.
    #
    # The camera image is mirrored in collect_data.py.
    # Therefore:
    # screen-left generally corresponds to the user's right hand,
    # screen-right generally corresponds to the user's left hand.

    unresolved_hands.sort(
        key=lambda hand: hand[0].x
    )

    for landmarks in unresolved_hands:

        wrist_x = landmarks[0].x

        if wrist_x < 0.5 and hands["Right"] is None:
            hands["Right"] = landmarks

        elif hands["Left"] is None:
            hands["Left"] = landmarks

        elif hands["Right"] is None:
            hands["Right"] = landmarks

    return hands


def extract_two_hand_features(result):
    """
    Convert detected hands into a fixed-length feature vector.

    Feature layout:

        Left hand:
            21 landmarks × x, y, z = 63 features

        Right hand:
            21 landmarks × x, y, z = 63 features

        Presence:
            left_present, right_present = 2 features

        Total:
            63 + 63 + 2 = 128 features

    Missing hands are represented using zeros.
    """

    hands = split_hands(result)

    left_hand = hands["Left"]
    right_hand = hands["Right"]

    if left_hand is None and right_hand is None:
        return None, hands

    # -------------------------------------------------
    # Select normalization anchor
    # -------------------------------------------------

    if left_hand is not None and right_hand is not None:

        left_wrist = left_hand[0]
        right_wrist = right_hand[0]

        anchor_x = (left_wrist.x + right_wrist.x) / 2
        anchor_y = (left_wrist.y + right_wrist.y) / 2
        anchor_z = (left_wrist.z + right_wrist.z) / 2

    elif left_hand is not None:

        wrist = left_hand[0]

        anchor_x = wrist.x
        anchor_y = wrist.y
        anchor_z = wrist.z

    else:

        wrist = right_hand[0]

        anchor_x = wrist.x
        anchor_y = wrist.y
        anchor_z = wrist.z

    # -------------------------------------------------
    # Calculate relative coordinates
    # -------------------------------------------------

    relative_hands = {
        "Left": [],
        "Right": []
    }

    for hand_name, landmarks in hands.items():

        if landmarks is None:
            continue

        for landmark in landmarks:
            relative_hands[hand_name].append(
                (
                    landmark.x - anchor_x,
                    landmark.y - anchor_y,
                    landmark.z - anchor_z
                )
            )

    all_relative_points = (
        relative_hands["Left"] +
        relative_hands["Right"]
    )

    # Normalize for different distances from the camera
    scale = max(
        max(abs(x), abs(y), abs(z))
        for x, y, z in all_relative_points
    )

    if scale == 0:
        scale = 1

    # -------------------------------------------------
    # Build fixed-length feature vector
    # -------------------------------------------------

    features = []

    for hand_name in ["Left", "Right"]:

        relative_points = relative_hands[hand_name]

        if relative_points:

            for x, y, z in relative_points:
                features.extend(
                    [
                        x / scale,
                        y / scale,
                        z / scale
                    ]
                )

        else:
            # Hand was not detected
            features.extend(
                [0.0] * FEATURES_PER_HAND
            )

    # Add hand-presence information
    features.append(
        1.0 if left_hand is not None else 0.0
    )

    features.append(
        1.0 if right_hand is not None else 0.0
    )

    if len(features) != TOTAL_FEATURES:
        raise RuntimeError(
            f"Expected {TOTAL_FEATURES} features, "
            f"but received {len(features)}."
        )

    return features, hands
