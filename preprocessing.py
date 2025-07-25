import re
import pandas as pd
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# -------------------------------------------
# Inisialisasi: Stopword & Stemmer
# -------------------------------------------
def init_stopwords():
    factory = StopWordRemoverFactory()
    return set(factory.get_stop_words())

def init_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stop_words = init_stopwords()
stemmer = init_stemmer()

# -------------------------------------------
# Tahapan 6-11: Pembersihan dan Normalisasi
# -------------------------------------------

def cleansing(text):
    """Menghapus URL, angka, simbol, dan karakter non-huruf."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()  # (7) Case folding

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def stemming_tokens(tokens):
    kalimat = ' '.join(tokens)
    hasil = stemmer.stem(kalimat)
    return hasil.split()

def filter_token_length(tokens, min_len=4, max_len=25):
    return [token for token in tokens if min_len <= len(token) <= max_len]

# -------------------------------------------
# Fungsi Preprocessing Utama (6-11)
# -------------------------------------------

def preprocess_text(text):
    """Melakukan preprocessing lengkap terhadap 1 teks."""
    if not isinstance(text, str):
        text = str(text)
    text = cleansing(text)  # (6 + 7)
    tokens = wordpunct_tokenize(text)  # (8) Tokenizing
    tokens = remove_stopwords(tokens)  # (9)
    tokens = stemming_tokens(tokens)   # (10)
    tokens = filter_token_length(tokens)  # (11)
    return ' '.join(tokens)

# -------------------------------------------
# Fungsi Gabungan Dataset (1-5)
# -------------------------------------------

def load_and_clean_data(df1, df2):
    """
    1. Penambahan atribut label untuk dataset Detik
    2. Pemilihan atribut yang akan digunakan
    3. Penyesuaian nama kolom
    4. Penggabungan dataset
    5. Penambahan atribut text (judul + narasi)
    """

    # (3) Normalisasi nama kolom
    kolom_alias = {
        'Judul': 'judul',
        'T_judul': 'judul',
        'T_isi': 'narasi',
        'Konten': 'narasi',
        'Isi': 'narasi',
        'Label': 'label',
        'T_label': 'label'
    }
    df1.rename(columns=kolom_alias, inplace=True)
    df2.rename(columns=kolom_alias, inplace=True)

    # (1) Penambahan label manual jika belum ada (misalnya dataset Detik)
    if 'label' not in df2.columns:
        df2['label'] = 'non-hoaks'  # Misalnya data dari detik diasumsikan valid

    # (4) Gabungkan kedua dataset
    df = pd.concat([df1, df2], ignore_index=True)

    # (2) Hapus kolom yang tidak relevan
    kolom_tidak_dipakai = ['ID', 'Tanggal', 'tanggal', 'Link', 'nama file gambar']
    df.drop(columns=[col for col in kolom_tidak_dipakai if col in df.columns], errors='ignore', inplace=True)

    # (2) Ganti tanda '?' dengan NaN jika ada
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].replace('?', np.nan)

    # (2) Hapus baris yang tidak lengkap
    df.dropna(subset=['judul', 'narasi', 'label'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # (5) Gabungkan judul + narasi jadi kolom baru 'text'
    df['judul'] = df['judul'].astype(str)
    df['narasi'] = df['narasi'].astype(str)
    df['text'] = df['judul'] + ' ' + df['narasi']

    # (6-11) Proses teks menjadi kolom 'T_text'
    df['T_text'] = df['text'].apply(preprocess_text)

    return df
