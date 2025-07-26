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
    Tahap 6: Cleansing - Membersihkan teks dari URL, angka, simbol, dan karakter non-alfabet.
    """
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Hapus URL
    text = re.sub(r"\d+", "", text)                     # Hapus angka
    text = re.sub(r"[^\w\s]", "", text)                 # Hapus simbol
    text = re.sub(r"[^a-zA-Z\s]", "", text)             # Hapus karakter non-alfabet
    text = re.sub(r"\s+", " ", text).strip()            # Hapus spasi berlebih
    return text.lower()                                 # Tahap 7: Case folding

def tokenize(text):
    """
    Tahap 8: Tokenizing - Tokenisasi kata menggunakan wordpunct_tokenize.
    """
    return wordpunct_tokenize(text)

def remove_stopwords(tokens):
    """
    Tahap 9: Stopword removal - Menghapus kata-kata umum yang tidak bermakna penting.
    """
    return [word for word in tokens if word not in stop_words]

def stemming_tokens(tokens):
    """
    Tahap 10: Stemming - Mengembalikan kata ke bentuk dasar menggunakan Sastrawi.
    """
    kalimat = " ".join(tokens)
    hasil_stem = stemmer.stem(kalimat)
    return hasil_stem.split()

def filter_token_length(tokens, min_len=4, max_len=25):
    """
    Tahap 11: Filter token - Menyaring token berdasarkan panjang huruf.
    """
    return [t for t in tokens if min_len <= len(t) <= max_len]

def preprocess_text(text):
    """
    Proses lengkap preprocessing teks:
    6. Cleansing
    7. Case folding (di dalam cleansing)
    8. Tokenizing
    9. Stopword removal
    10. Stemming
    11. Filter token panjang
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
    Proses gabungan data:
    1. Penambahan atribut label jika belum ada di df2
    2. Pemilihan atribut: 'judul', 'narasi', 'label'
    3. Penyesuaian nama kolom
    4. Penggabungan df1 dan df2
    """

    # 1. Tambahkan kolom label jika belum ada
    if 'label' not in df2.columns:
        df2['label'] = 'Non-Hoax'
    else:
        df2['label'] = df2['label'].fillna('Non-Hoax')

    # 3. Penyesuaian nama kolom df2
    df2 = df2.rename(columns={
        "Judul": "judul",
        "Isi": "narasi",
        "Konten": "narasi",
        "T_judul": "judul",
        "T_isi": "narasi",
        "T_label": "label",
        "Label": "label"
    })

    # 2. Pilih kolom yang diperlukan
    expected_cols = ["judul", "narasi", "label"]
    df2 = df2[[col for col in expected_cols if col in df2.columns]]

    # Pastikan df1 juga hanya punya kolom yang sama
    df1 = df1[[col for col in expected_cols if col in df1.columns]]

    # Bersihkan kolom duplikat
    df1 = df1.loc[:, ~df1.columns.duplicated()]
    df2 = df2.loc[:, ~df2.columns.duplicated()]

    # 4. Gabungkan kedua DataFrame
    df = pd.concat([df1, df2], ignore_index=True)

    # Hapus kolom tidak penting jika masih ada
    kolom_tidak_dipakai = ["ID", "Tanggal", "tanggal", "Link", "nama file gambar"]
    df = df.drop(columns=[col for col in kolom_tidak_dipakai if col in df.columns], errors="ignore")

    # Bersihkan nilai '?'
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace("?", np.nan)

    # Drop baris yang tidak lengkap
    df.dropna(subset=["judul", "narasi", "label"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def preprocess_dataframe(df):
    """
    Tahapan lanjutan:
    5. Buat kolom 'text' dari gabungan 'judul' dan 'narasi'
    6–11. Lakukan preprocessing ke kolom 'text', simpan ke 'T_text'
    """
    df["judul"] = df["judul"].astype(str)
    df["narasi"] = df["narasi"].astype(str)

    # 5. Gabungkan kolom judul + narasi
    df["text"] = df["judul"] + " " + df["narasi"]

    # 6-11. Terapkan preprocessing
    df["T_text"] = df["text"].apply(preprocess_text)

    return df

def preprocess_with_steps(text):
    hasil = {}
    hasil['original'] = text
    hasil['cleansing'] = cleansing(text)
    hasil['case_folding'] = case_folding(hasil['cleansing'])
    hasil['tokenizing'] = tokenizing(hasil['case_folding'])
    hasil['stopword_removal'] = stopword_removal(hasil['tokenizing'])
    hasil['stemming'] = stemming(hasil['stopword_removal'])
    hasil['filtering'] = filter_tokens(hasil['stemming'])
    return hasil
