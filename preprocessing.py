import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK tokenizer
nltk.download('punkt')
nltk.download('stopwords')  # Hapus baris ini jika kamu tidak pakai stopwords

# Inisialisasi Sastrawi Stopword & Stemmer
stopword_factory = StopWordRemoverFactory()
stop_words = set(stopword_factory.get_stop_words())

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

def load_and_clean_data(df1, df2):
    """
    Menggabungkan, membersihkan kolom, dan menghapus missing value.
    """
    df2_renamed = df2.rename(columns={
        'Judul': 'judul',
        'Konten': 'narasi',
        'Label': 'label'
    })
    kolom_tidak_dipakai = ['ID', 'Tanggal', 'tanggal', 'Link', 'nama file gambar']
    
    df = pd.concat([df1, df2_renamed], ignore_index=True)
    df = df.drop(columns=[col for col in kolom_tidak_dipakai if col in df.columns])
    
    # Ganti '?' jadi NaN
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].replace('?', np.nan)
    
    # Drop kolom yang mengandung NaN
    df = df.dropna(axis=1)
    
    return df

def cleansing(text):
    """
    Membersihkan teks: URL, angka, tanda baca, whitespace.
    """
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def preprocess_dataframe(df):
    """
    Melakukan seluruh proses preprocessing pada dataframe.
    """
    df['judul'] = df['judul'].apply(cleansing)
    df['narasi'] = df['narasi'].apply(cleansing)

    df['judul_token'] = df['judul'].apply(word_tokenize)
    df['narasi_token'] = df['narasi'].apply(word_tokenize)

    df['T_judul'] = df['judul_token'].apply(remove_stopwords)
    df['T_konten'] = df['narasi_token'].apply(remove_stopwords)

    df['T_judul'] = df['T_judul'].apply(stemming_tokens)
    df['T_konten'] = df['T_konten'].apply(stemming_tokens)

    df['T_judul'] = df['T_judul'].apply(lambda tokens: filter_token_length(tokens, 3, 25))
    df['T_konten'] = df['T_konten'].apply(lambda tokens: filter_token_length(tokens, 3, 25))

    return df

def remove_stopwords(tokens):
    """
    Menghapus stopword menggunakan Sastrawi.
    """
    return [word for word in tokens if word not in stop_words]

def stemming_tokens(tokens):
    """
    Melakukan stemming pada token list.
    """
    kalimat = ' '.join(tokens)
    kalimat_stem = stemmer.stem(kalimat)
    return kalimat_stem.split()

def filter_token_length(tokens, min_len=4, max_len=25):
    """
    Filter token berdasarkan panjang karakter.
    """
    return [token for token in tokens if min_len <= len(token) <= max_len]

def preprocess_text(text):
    """
    Untuk preprocessing teks single (misalnya user input di Streamlit).
    """
    text = cleansing(text)
    tokens = word_tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming_tokens(tokens)
    tokens = filter_token_length(tokens)
    return ' '.join(tokens)
