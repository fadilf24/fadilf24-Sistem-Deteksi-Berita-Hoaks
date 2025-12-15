import numpy as np
from typing import Any, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================
# Split Data
# ======================================================
def split_data(
    X: Any,
    y: Any,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Any, Any, Any]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ======================================================
# Train Fuzzy Classifier (Centroid-based)
# ======================================================
def train_fuzzy_classifier(
    X_train: Any,
    y_train: Any
) -> Dict[str, Any]:

    classes = np.unique(y_train)
    centroids = {}

    for cls in classes:
        # Mean TF-IDF vector per class (AMAN UNTUK SPARSE)
        centroids[cls] = X_train[y_train == cls].mean(axis=0)

    return {
        "centroids": centroids,
        "classes": classes
    }


# ======================================================
# Predict Fuzzy (Cosine Similarity)
# ======================================================
def predict_fuzzy(
    model: Dict[str, Any],
    X_test: Any
) -> Tuple[np.ndarray, Dict[str, float]]:

    centroids = model["centroids"]
    classes = model["classes"]

    predictions = []
    membership_sum = {cls: 0.0 for cls in classes}

    for i in range(X_test.shape[0]):
        x = X_test[i]

        similarities = {}
        for cls in classes:
            centroid = centroids[cls]

            # COSINE SIMILARITY (FIX DIMENSION ISSUE)
            sim = cosine_similarity(x, centroid)[0][0]
            similarities[cls] = sim

        predicted_class = max(similarities, key=similarities.get)
        predictions.append(predicted_class)

        for cls, score in similarities.items():
            membership_sum[cls] += score

    # Normalisasi ke persen
    total = sum(membership_sum.values())
    distribution = {
        cls: round((val / total) * 100, 2) if total > 0 else 0.0
        for cls, val in membership_sum.items()
    }

    return np.array(predictions), distribution
