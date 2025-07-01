from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Membagi data fitur dan label menjadi data latih dan uji.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_naive_bayes(X_train, y_train):
    """
    Melatih model Naive Bayes (MultinomialNB).
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def predict_naive_bayes(model, X_test):
    """
    Melakukan prediksi menggunakan model Naive Bayes terlatih.
    """
    return model.predict(X_test)
def prediction_distribution(y_pred, label_encoder):
    """
    Menghitung distribusi prediksi Hoax dan Non-Hoax dalam persen.

    Args:
        y_pred: array hasil prediksi
        label_encoder: instance dari LabelEncoder yang telah fit
    Returns:
        dict berisi label dan persentase prediksi
    """
    unique, counts = np.unique(y_pred, return_counts=True)
    total = len(y_pred)
    percentages = {
        label_encoder.inverse_transform([cls])[0]: round((count / total) * 100, 2)
        for cls, count in zip(unique, counts)
    }
    return percentages
