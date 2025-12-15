import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict, Any

def split_data(
    X: Any,
    y: Any,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Any, Any, Any]:
    """
    Membagi dataset menjadi data latih dan data uji.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

class FuzzyClassifier:
    """
    Fuzzy Classification berbasis cosine similarity.
    Menghasilkan derajat keanggotaan (membership value)
    untuk setiap kelas.
    """

    def __init__(self):
        self.class_centroids = {}
        self.classes = None

    def fit(self, X_train: Any, y_train: Any) -> None:
        if X_train.shape[0] == 0:
            raise ValueError("Data latih kosong, tidak bisa melatih model.")

        self.classes = np.unique(y_train)

        # Hitung centroid tiap kelas
        for cls in self.classes:
            self.class_centroids[cls] = np.mean(
                X_train[y_train == cls], axis=0
            )

    def fuzzy_predict(self, X_test: Any) -> np.ndarray:
        if X_test.shape[0] == 0:
            raise ValueError("Data uji kosong, tidak bisa melakukan prediksi.")

        fuzzy_results = []

        for x in X_test:
            similarities = []

            for cls in self.classes:
                sim = cosine_similarity(
                    x, self.class_centroids[cls]
                )[0][0]
                similarities.append(sim)

            similarities = np.array(similarities)

            # Normalisasi fuzzy (jumlah = 1)
            membership = similarities / np.sum(similarities)
            fuzzy_results.append(membership)

        return np.array(fuzzy_results)

    def predict(self, X_test: Any) -> np.ndarray:
        """
        Menghasilkan label akhir berdasarkan
        nilai membership tertinggi.
        """
        fuzzy_values = self.fuzzy_predict(X_test)

def train_fuzzy_classifier(X_train: Any, y_train: Any) -> FuzzyClassifier:
    """
    Melatih model Fuzzy Classification.
    """
    model = FuzzyClassifier()
    model.fit(X_train, y_train)
    return model

def predict_fuzzy(model: FuzzyClassifier, X_test: Any) -> np.ndarray:
    """
    Melakukan prediksi label menggunakan Fuzzy Classification.
    """
    return model.predict(X_test)

def prediction_distribution(
    y_pred: np.ndarray,
    label_encoder: LabelEncoder
) -> Dict[str, float]:
    """
    Menghitung distribusi hasil prediksi dalam persentase tiap kelas.
    """

    if y_pred.size == 0:
        return {"empty": 0.0}

    unique, counts = np.unique(y_pred, return_counts=True)
    total = len(y_pred)

    return {
        label_encoder.inverse_transform([label])[0]:
        round((count / total) * 100, 2)
        for label, count in zip(unique, counts)
    }

        return self.classes[np.argmax(fuzzy_values, axis=1)]
