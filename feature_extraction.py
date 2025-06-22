import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def combine_text_columns(df, col1='T_judul', col2='T_konten'):
    """
    Gabungkan dua kolom token menjadi satu string untuk setiap baris.
    Default: kolom 'T_judul' dan 'T_konten'.
    """
    df['gabungan'] = df[col1].apply(lambda x: ' '.join(x)) + ' ' + df[col2].apply(lambda x: ' '.join(x))
    return df

def tfidf_transform(text_series):
    """
    Transformasi teks (series) menjadi TF-IDF matrix.
    Mengembalikan matrix fitur dan vectorizer untuk digunakan kembali.
    """
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(text_series)
    return features, vectorizer

def tfidf_to_dataframe(tfidf_matrix, vectorizer):
    """
    Konversi TF-IDF matrix ke DataFrame agar mudah dilihat atau diekspor.
    """
    return pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

def save_tfidf_output(df_preprocessed, tfidf_df, preprocessed_path='preprocessed_data.csv', tfidf_path='tfidf_output.csv'):
    """
    Simpan hasil preprocessing dan TF-IDF ke file CSV.
    """
    df_preprocessed.to_csv(preprocessed_path, index=False)
    tfidf_df.to_csv(tfidf_path, index=False)
    print(f"Preprocessed data saved to {preprocessed_path}")
    print(f"TF-IDF output saved to {tfidf_path}")
