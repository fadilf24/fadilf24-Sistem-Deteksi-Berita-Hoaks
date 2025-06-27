import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import preprocess_text, preprocess_dataframe, load_and_clean_data
from feature_extraction import combine_text_columns, tfidf_transform
from interpretation import configure_gemini, analyze_with_gemini

st.set_page_config(page_title="Deteksi Berita Hoaks", page_icon="🗰️")
st.title("Deteksi Berita Hoaks (Naive Bayes + Gemini LLM)")

# -----------------------
# 🔍 Sidebar Navigasi
# -----------------------
st.sidebar.title("🔍 Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", (
    "Deteksi Hoaks", 
    "Dataset", 
    "Preprocessing", 
    "Evaluasi Model"
))

# -----------------------
# 🚀 Load & Preprocess
# -----------------------

@st.cache_data
def load_dataset():
    df1 = pd.read_csv("Data_latih.csv")
    df2 = pd.read_csv("detik_data.csv")
    return df1, df2

@st.cache_data
def prepare_data(df1, df2):
    df = load_and_clean_data(df1, df2)
    df = preprocess_dataframe(df)
    df = combine_text_columns(df)

    # Pastikan label diubah jadi angka: 1 = hoaks, 0 = non-hoaks
    label_map = {"Hoax": 1, "Non-Hoax": 0, 1: 1, 0: 0}
    df["label"] = df["label"].map(label_map)
    df = df[df["label"].notna()]  # Hapus baris dengan label tidak valid
    df["label"] = df["label"].astype(int)
    return df

@st.cache_data
def extract_features_and_model(df):
    X, vectorizer = tfidf_transform(df["gabungan"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, vectorizer, X_test, y_test, y_pred

# Inisialisasi data
try:
    df1, df2 = load_dataset()
    df = prepare_data(df1, df2)
    model, vectorizer, X_test, y_test, y_pred = extract_features_and_model(df)
except Exception as e:
    st.error(f"Gagal memuat atau memproses data: {e}")
    st.stop()

# -----------------------
# 🏠 Deteksi Hoaks
# -----------------------
if menu == "Deteksi Hoaks":
    st.subheader("Masukkan Teks Berita")
    user_input = st.text_area("Contoh: Pemerintah mengumumkan vaksin palsu beredar di Jakarta...")

    if st.button("Analisis Berita"):
        if not user_input.strip():
            st.warning("Teks tidak boleh kosong.")
        else:
            processed = preprocess_text(user_input)
            vectorized = vectorizer.transform([processed])
            prediction = model.predict(vectorized)[0]

            label_map = {1: "Hoax", 0: "Non-Hoax"}
            st.success(f"✅ Prediksi: {label_map[prediction]}")

            try:
                api_key = "AIzaSyDFRv6-gi44fDsJvR_l4E8N2Fxd45oGozU"
                configure_gemini(api_key)
                result = analyze_with_gemini(user_input, true_label="Unknown", predicted_label=label_map[prediction])
                st.success("Hasil Interpretasi LLM:")
                st.text(result)
            except Exception as e:
                st.error(f"❌ Error saat menggunakan Gemini: {e}")

# -----------------------
# 📂 Dataset
# -----------------------
elif menu == "Dataset":
    st.subheader("Dataset Kaggle (Data_latih.csv):")
    st.write(df1.head())
    st.subheader("Dataset Detik.com (detik_data.csv):")
    st.write(df2.head())
    st.subheader("Dataset Gabungan:")
    st.write(df[['T_judul', 'T_konten', 'label']].head())

# -----------------------
# ⚙️ Preprocessing
# -----------------------
elif menu == "Preprocessing":
    st.subheader("Hasil Preprocessing:")
    st.write(df[['T_judul', 'T_konten']].head())
    st.subheader("Gabungan Judul + Konten:")
    st.write(df[['gabungan']].head())

# -----------------------
# 📊 Evaluasi Model
# -----------------------
elif menu == "Evaluasi Model":
    st.subheader("Evaluasi Model:")
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="Akurasi", value=f"{acc*100:.2f}%")

    st.subheader("Laporan Klasifikasi:")
    report = classification_report(y_test, y_pred, target_names=["Non-Hoax", "Hoax"])
    st.text(report)

    st.subheader("📈 Visualisasi Prediksi:")
    df_eval = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    df_eval["Hasil"] = np.where(df_eval["Actual"] == df_eval["Predicted"], "Benar", "Salah")

    fig, ax = plt.subplots()
    sns.countplot(data=df_eval, x="Hasil", palette="pastel", ax=ax)
    ax.set_title("Distribusi Prediksi Benar vs Salah")
    st.pyplot(fig)

    st.subheader("🔍 Contoh Data Salah Prediksi:")
    salah = df_eval[df_eval["Hasil"] == "Salah"]
    if not salah.empty:
        st.dataframe(salah.head())
    else:
        st.success("Semua prediksi benar!")
