import re
import pandas as pd
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi global untuk stemmer dan stopword
def init_stopwords():
    factory = StopWordRemoverFactory()
    return set(factory.get_stop_words())

def init_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stop_words = init_stopwords()
stemmer = init_stemmer()

# -------------------------------------------
# Pembersihan dan Normalisasi Teks
# -------------------------------------------

def cleansing(text):
    """
    Membersihkan teks dari URL, angka, simbol, dan karakter non-huruf.
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

# -------------------------------------------
# Fungsi Preprocessing Utama
# -------------------------------------------

def preprocess_text(text):
    """
    Melakukan semua tahapan preprocessing pada teks tunggal:
    cleansing → tokenizing → stopword removal → stemming → filtering
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
    Melakukan preprocessing dataframe pada kolom judul + narasi → T_text.
    Digunakan untuk menyiapkan data sebelum TF-IDF dan klasifikasi.
    """
    df['judul'] = df['judul'].astype(str)
    df['narasi'] = df['narasi'].astype(str)
    df['text'] = df['judul'] + ' ' + df['narasi']
    df['T_text'] = df['text'].apply(preprocess_text)
    return df

# -------------------------------------------
# Fungsi Gabung Data
# -------------------------------------------

def load_and_clean_data(df1, df2):
    """
    Menggabungkan dua dataframe (df1, df2) dan memastikan kolom sesuai:
    - Menstandarkan kolom df2 jika perlu
    - Menghapus kolom yang tidak diperlukan
    - Menggabungkan dan membersihkan NaN
    """
    # Rename jika kolom df2 masih menggunakan nama asli
    expected_columns = ['judul', 'narasi', 'label']
    df2 = df2.rename(columns={
        'Judul': 'judul',
        'Konten': 'narasi',
        'Isi': 'narasi',
        'Label': 'label',
        'T_judul': 'judul',
        'T_isi': 'narasi',
        'T_label': 'label'
    })

    # Gabungkan dataframe
    df = pd.concat([df1, df2], ignore_index=True)

    # Hapus kolom tidak penting jika ada
    kolom_tidak_dipakai = ['ID', 'Tanggal', 'tanggal', 'Link', 'nama file gambar']
    df = df.drop(columns=[col for col in kolom_tidak_dipakai if col in df.columns], errors='ignore')

    # Ganti tanda '?' jadi NaN jika ada
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].replace('?', np.nan)

    # Drop baris yang tidak lengkap
    df.dropna(subset=['judul', 'narasi', 'label'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
