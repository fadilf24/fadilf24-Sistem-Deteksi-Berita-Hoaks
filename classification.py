import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict

def split_data(X, y, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Membagi data fitur dan label menjadi data latih dan data uji.

    Args:
        X: Fitur (fitur TF-IDF atau fitur lainnya)
        y: Label
        test_size: Proporsi data uji
        random_state: Seed acak untuk replikasi

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_naive_bayes(X_train, y_train) -> MultinomialNB:
    """
    Melatih model Naive Bayes (MultinomialNB) dengan data latih.

    Args:
        X_train: Data fitur latih
        y_train: Data label latih

    Returns:
        Model MultinomialNB yang telah dilatih
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def predict_naive_bayes(model: MultinomialNB, X_test) -> np.ndarray:
    """
    Melakukan prediksi menggunakan model Naive Bayes.

    Args:
        model: Model MultinomialNB yang telah dilatih
        X_test: Data fitur uji

    Returns:
        Array hasil prediksi
    """
    return model.predict(X_test)

def prediction_distribution(y_pred: np.ndarray, label_encoder: LabelEncoder) -> Dict[str, float]:
    """
    Menghitung distribusi persentase prediksi untuk masing-masing kelas.

    Args:
        y_pred: Array hasil prediksi
        label_encoder: Encoder label untuk mengembalikan label asli

    Returns:
        Dictionary dengan label asli dan persentase prediksi
    """
    unique, counts = np.unique(y_pred, return_counts=True)
    total = len(y_pred)
    percentages = {
        label_encoder.inverse_transform([cls])[0]: round((count / total) * 100, 2)
        for cls, count in zip(unique, counts)
    }
    return percentages
