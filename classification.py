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
