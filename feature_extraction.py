import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def combine_text_columns(df, col1='T_judul', col2='T_konten'):
    """
    Gabungkan dua kolom token menjadi satu string untuk setiap baris.
    Default: kolom 'T_judul' dan 'T_konten'.
    """
    df[col1] = df[col1].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    df[col2] = df[col2].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    df['gabungan'] = df[col1] + ' ' + df[col2]
    return df

def tfidf_transform(text_series):
    """
    Transformasi teks (series) menjadi TF-IDF matrix.
    Mengembalikan matrix fitur dan vectorizer untuk digunakan kembali.
    """
    # Bersihkan nilai NaN dan string kosong
    text_series = text_series.fillna('').astype(str)
    text_series = text_series.apply(lambda x: x.strip())
    text_series = text_series[text_series != '']

    # Validasi isi dokumen
    if text_series.empty:
        raise ValueError("Semua dokumen kosong setelah preprocessing atau hanya berisi stop words.")

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
