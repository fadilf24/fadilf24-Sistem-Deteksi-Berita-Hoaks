import re
import pandas as pd
import numpy as np
from nltk.tokenize import wordpunct_tokenize  # Tidak perlu unduh punkt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def init_stopwords():
    factory = StopWordRemoverFactory()
    return set(factory.get_stop_words())

def init_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stop_words = init_stopwords()
stemmer = init_stemmer()

def cleansing(text):
    """
    Membersihkan teks dari URL, angka, simbol, dan whitespace.
    """
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def stemming_tokens(tokens):
    kalimat = ' '.join(tokens)
    kalimat_stem = stemmer.stem(kalimat)
    return kalimat_stem.split()

def filter_token_length(tokens, min_len=4, max_len=25):
    return [token for token in tokens if min_len <= len(token) <= max_len]

def preprocess_text(text):
    """
    Preprocessing teks tunggal (user input dari Streamlit).
    Output: string teks yang sudah bersih.
    """
    if not isinstance(text, str):
        text = str(text)
    text = cleansing(text)
    tokens = wordpunct_tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming_tokens(tokens)
    tokens = filter_token_length(tokens)
    return ' '.join(tokens)

def preprocess_dataframe(df):
    """
    Preprocessing dataframe untuk kolom 'judul' dan 'narasi'.
    Digabungkan menjadi kolom 'text', lalu diproses menjadi kolom 'T_text'.
    """
    df['judul'] = df['judul'].astype(str)
    df['narasi'] = df['narasi'].astype(str)

    df['text'] = df['judul'] + ' ' + df['narasi']
    df['T_text'] = df['text'].apply(preprocess_text)
    
    return df

def load_and_clean_data(df1, df2):
    """
    Menggabungkan data dari dua sumber dan membersihkan kolom tidak perlu.
    Digunakan sebelum TF-IDF dan klasifikasi.
    """
    df2_renamed = df2.rename(columns={
        'Judul': 'judul',
        'Konten': 'narasi',
        'Label': 'label'
    })

    kolom_tidak_dipakai = ['ID', 'Tanggal', 'tanggal', 'Link', 'nama file gambar']
    df = pd.concat([df1, df2_renamed], ignore_index=True)

    df = df.drop(columns=[col for col in kolom_tidak_dipakai if col in df.columns], errors='ignore')

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].replace('?', np.nan)

    df = df.dropna(subset=['judul', 'narasi', 'label'])
    
    df['text'] = df['judul'].astype(str) + ' ' + df['narasi'].astype(str)
    return df
