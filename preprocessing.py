import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import wordpunct_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# Pastikan stopwords Bahasa Indonesia tersedia
try:
    stopword_list = stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')
    stopword_list = stopwords.words('indonesian')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def load_and_clean_data(df1, df2):
    # Ubah nama kolom agar seragam
    df2 = df2.rename(columns={
        'Judul': 'judul',
        'Isi': 'narasi',
        'Konten': 'narasi',
        'T_judul': 'judul',
        'T_isi': 'narasi',
        'T_label': 'label',
        'Label': 'label'
    })

    # ✅ Tambahkan kolom 'label' jika tidak tersedia
    if 'label' not in df2.columns:
        df2['label'] = 'non-hoaks'

    # Ambil hanya kolom yang relevan
    expected_cols = ['judul', 'narasi', 'label']
    df2 = df2[[col for col in expected_cols if col in df2.columns]]

    # Gabungkan df1 dan df2
    df = pd.concat([df1, df2], ignore_index=True)

    # Hapus kolom yang tidak diperlukan jika ada
    kolom_tidak_dipakai = ['ID', 'Tanggal', 'tanggal', 'Link', 'nama file gambar']
    df.drop(columns=[col for col in kolom_tidak_dipakai if col in df.columns], inplace=True, errors='ignore')

    # Bersihkan tanda tanya atau karakter tidak valid
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].replace('?', np.nan)

    # Drop data yang kosong di kolom penting
    df.dropna(subset=['judul', 'narasi', 'label'], inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df

def clean_text(text):
    # Lowercase
    text = text.lower()

    # Hapus URL, angka, simbol, dan karakter aneh
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenisasi
    tokens = wordpunct_tokenize(text)

    # Stopword removal dan stemming
    filtered = [stemmer.stem(word) for word in tokens if word not in stopword_list]

    return " ".join(filtered)

def preprocess_dataframe(df):
    df['isi_bersih'] = df['judul'] + " " + df['narasi']
    df['isi_bersih'] = df['isi_bersih'].apply(clean_text)
    return df
