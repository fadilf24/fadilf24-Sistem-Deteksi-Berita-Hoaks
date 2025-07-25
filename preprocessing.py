import re
import pandas as pd
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# -----------------------------------
# Inisialisasi global: Stopword dan Stemmer
# -----------------------------------

def init_stopwords():
    factory = StopWordRemoverFactory()
    return set(factory.get_stop_words())

def init_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stop_words = init_stopwords()
stemmer = init_stemmer()

# -----------------------------------
# Fungsi Preprocessing Teks Tunggal
# -----------------------------------

def cleansing(text):
    """
    Membersihkan teks dari URL, angka, simbol, dan karakter non-alfabet.
    """
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Hapus URL
    text = re.sub(r"\d+", "", text)                     # Hapus angka
    text = re.sub(r"[^\w\s]", "", text)                 # Hapus simbol
    text = re.sub(r"[^a-zA-Z\s]", "", text)             # Hapus karakter non-alfabet
    text = re.sub(r"\s+", " ", text).strip()            # Hapus spasi berlebih
    return text.lower()                                 # Case folding

def tokenize(text):
    """
    Tokenisasi kata menggunakan wordpunct_tokenize.
    """
    return wordpunct_tokenize(text)

def remove_stopwords(tokens):
    """
    Menghapus stopword dari token-token.
    """
    return [word for word in tokens if word not in stop_words]

def stemming_tokens(tokens):
    """
    Melakukan stemming pada token-token.
    """
    kalimat = " ".join(tokens)
    hasil_stem = stemmer.stem(kalimat)
    return hasil_stem.split()

def filter_token_length(tokens, min_len=4, max_len=25):
    """
    Menyaring token berdasarkan panjang minimal dan maksimal.
    """
    return [t for t in tokens if min_len <= len(t) <= max_len]

def preprocess_text(text):
    """
    Proses lengkap preprocessing:
    1. Cleansing
    2. Tokenizing
    3. Stopword removal
    4. Stemming
    5. Filter panjang token
    """
    if not isinstance(text, str):
        text = str(text)

    text = cleansing(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming_tokens(tokens)
    tokens = filter_token_length(tokens)

    return " ".join(tokens)

# -----------------------------------
# Fungsi Preprocessing untuk DataFrame
# -----------------------------------

def load_and_clean_data(df1, df2):
    """
    Menggabungkan dan membersihkan dua dataset:
    - Tambah label 'Non-Hoax' jika kolom label belum ada
    - Penyesuaian nama kolom
    - Hapus kolom tidak penting
    - Hapus baris dengan data kosong
    """
    # Tambahkan kolom label jika belum ada
    if 'label' not in df2.columns:
        df2['label'] = 'Non-Hoax'

    # Penyesuaian nama kolom df2
    df2 = df2.rename(columns={
        "Judul": "judul",
        "Isi": "narasi",
        "Konten": "narasi",
        "T_judul": "judul",
        "T_isi": "narasi",
        "T_label": "label",
        "Label": "label"
    })

    # Hanya ambil kolom yang diperlukan
    expected_cols = ["judul", "narasi", "label"]
    df2 = df2[[col for col in expected_cols if col in df2.columns]]

    # Gabungkan dataset
    df = pd.concat([df1, df2], ignore_index=True)

    # Hapus kolom tidak dipakai jika ada
    kolom_tidak_dipakai = ["ID", "Tanggal", "tanggal", "Link", "nama file gambar"]
    df = df.drop(columns=[col for col in kolom_tidak_dipakai if col in df.columns], errors="ignore")

    # Ganti tanda '?' jadi NaN, lalu drop baris yang tidak lengkap
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace("?", np.nan)
    df.dropna(subset=["judul", "narasi", "label"], inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df

def preprocess_dataframe(df):
    """
    Preprocessing DataFrame:
    - Gabungkan judul + narasi → kolom 'text'
    - Terapkan preprocess_text ke kolom 'text' → simpan di 'T_text'
    """
    df["judul"] = df["judul"].astype(str)
    df["narasi"] = df["narasi"].astype(str)
    df["text"] = df["judul"] + " " + df["narasi"]
    df["T_text"] = df["text"].apply(preprocess_text)
    return df
def preprocess_with_steps(text):
    """
    Menjalankan semua tahapan preprocessing dan menyimpan hasil setiap langkah.
    """
    if not isinstance(text, str):
        text = str(text)

    hasil = {}

    # 6. Cleansing
    cleansed = cleansing(text)
    hasil["cleansing"] = cleansed

    # 7. Case folding (sudah dilakukan dalam cleansing)
    hasil["case_folding"] = cleansed.lower()

    # 8. Tokenizing
    tokens = tokenize(cleansed)
    hasil["tokenizing"] = tokens

    # 9. Stopword removal
    no_stopwords = remove_stopwords(tokens)
    hasil["stopword_removal"] = no_stopwords

    # 10. Stemming
    stemmed = stemming_tokens(no_stopwords)
    hasil["stemming"] = stemmed

    # 11. Filter token length
    filtered = filter_token_length(stemmed)
    hasil["filter_tokens"] = filtered

    # Hasil akhir
    hasil["final"] = " ".join(filtered)

    return hasil
