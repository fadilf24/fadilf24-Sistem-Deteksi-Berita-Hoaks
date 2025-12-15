import numpy as np
from typing import Any, Tuple, Dict
from sklearn.model_selection import train_test_split


# ======================================================
# Split Data
# ======================================================
def split_data(
    X: Any,
    y: Any,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Any, Any, Any]:
    """
    Membagi data menjadi data latih dan data uji
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ======================================================
# Train Fuzzy Classifier
# ======================================================
def train_fuzzy_classifier(
    X_train: Any,
    y_train: Any
) -> Dict[str, Any]:
    """
    Melatih model fuzzy berbasis centroid TF-IDF per kelas
    """
    if X_train.shape[0] == 0:
        raise ValueError("Data latih kosong")

    classes = np.unique(y_train)
    centroids = {}

    for cls in classes:
        centroids[cls] = X_train[y_train == cls].mean(axis=0)

    return {
        "centroids": centroids,
        "classes": classes
    }


# ======================================================
# Predict Fuzzy
# ======================================================
def predict_fuzzy(
    model: Dict[str, Any],
    X_test: Any
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Prediksi fuzzy berbasis cosine similarity
    """
    centroids = model["centroids"]
    classes = model["classes"]

    predictions = []
    membership_scores = {cls: 0.0 for cls in classes}

    for x in X_test:
        similarities = {}

        for cls in classes:
            centroid = centroids[cls]
            num = np.dot(x, centroid.T)
            denom = np.linalg.norm(x) * np.linalg.norm(centroid)

            similarity = num / denom if denom != 0 else 0
            similarities[cls] = similarity

        predicted_class = max(similarities, key=similarities.get)
        predictions.append(predicted_class)

        for cls, score in similarities.items():
            membership_scores[cls] += score

    # Normalisasi derajat keanggotaan
    total = sum(membership_scores.values())
    distribution = {
        cls: round((score / total) * 100, 2) if total != 0 else 0
        for cls, score in membership_scores.items()
    }

    return np.array(predictions), distribution
