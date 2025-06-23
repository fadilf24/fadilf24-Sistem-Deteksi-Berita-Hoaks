import re
import pandas as pd
import numpy as np
from nltk.tokenize import wordpunct_tokenize  # GANTI tokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Tidak perlu download 'punkt' lagi
# nltk.download('punkt')  # Hapus ini karena Streamlit Cloud tidak support

# Inisialisasi Sastrawi Stopword & Stemmer
stopword_factory = StopWordRemoverFactory()
stop_words = set(stopword_factory.get_stop_words())

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

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
    
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].replace('?', np.nan)
    
    df = df.dropna(axis=1)
    return df

def preprocess_text(text):
    """
    Preprocessing untuk single text (user input).
    Output: string hasil preprocessing.
    """
    text = cleansing(text)
    tokens = wordpunct_tokenize(text)  # Gunakan tokenizer ringan
    tokens = remove_stopwords(tokens)
    tokens = stemming_tokens(tokens)
    tokens = filter_token_length(tokens)
    return ' '.join(tokens)
