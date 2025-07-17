import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import io
import re
import json
import uuid
from datetime import datetime
import pytz
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from streamlit_option_menu import option_menu
from fpdf import FPDF
import firebase_admin
from firebase_admin import credentials, db

from preprocessing import preprocess_text, preprocess_dataframe, load_and_clean_data
from feature_extraction import combine_text_columns, tfidf_transform
from interpretation import configure_gemini, analyze_with_gemini

st.set_page_config(page_title="Deteksi Berita Hoaks", page_icon="🔎", layout="wide")

# ✅ Konfigurasi Firebase
firebase_cred = dict(st.secrets["FIREBASE_KEY"])
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_cred)
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://deteksi-hoaks-streamlit-default-rtdb.asia-southeast1.firebasedatabase.app/"
    })

def simpan_ke_firebase(data):
    # Tambahkan timestamp WIB
    tz = pytz.timezone("Asia/Jakarta")
    waktu_wib = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    data["timestamp"] = waktu_wib
    ref = db.reference("prediksi_hoaks")
    ref.child(str(uuid.uuid4())).set(data)

def read_predictions_from_firebase():
    try:
        ref = db.reference("prediksi_hoaks")
        data = ref.get()
        return pd.DataFrame(data.values()) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal membaca data dari Firebase: {e}")
        return pd.DataFrame()

# ✅ Sidebar Navigasi
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Deteksi Hoaks", "Dataset", "Preprocessing", "Evaluasi Model", "Riwayat Prediksi"],
        icons=["search", "folder", "tools", "bar-chart", "clock-history"],
        default_index=0,
        orientation="vertical"
    )

st.title("📰 Deteksi Berita Hoaks (Naive Bayes + LLM)")

@st.cache_data
def load_dataset():
    return pd.read_csv("Data_latih.csv"), pd.read_csv("detik_data.csv")

@st.cache_data
def prepare_data(df1, df2):
    df = load_and_clean_data(df1, df2)
    df = preprocess_dataframe(df)
    df = combine_text_columns(df)
    label_map = {"Hoax": 1, "Non-Hoax": 0, 1: 1, 0: 0}
    df["label"] = df["label"].map(label_map)
    df = df[df["label"].notna()]
    df["label"] = df["label"].astype(int)
    return df

@st.cache_data
def extract_features_and_model(df):
    X, vectorizer = tfidf_transform(df["gabungan"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, vectorizer, X_test, y_test, y_pred

def is_valid_text(text):
    words = re.findall(r'\w+', text)
    return len(words) >= 5 and any(len(word) > 3 for word in words)

# ✅ Load Data dan Model
try:
    df1, df2 = load_dataset()
    df = prepare_data(df1, df2)
    model, vectorizer, X_test, y_test, y_pred = extract_features_and_model(df)
except Exception as e:
    st.error(f"Gagal memuat atau memproses data:\n{e}")
    st.stop()

hasil_semua = []

# ✅ Menu Deteksi Hoaks
if selected == "Deteksi Hoaks":
    st.subheader("Masukkan Teks Berita:")
    user_input = st.text_area("Contoh: Pemerintah mengumumkan vaksin palsu beredar di Jakarta...", height=200)

    if st.button("Analisis Berita"):
        if not user_input.strip():
            st.warning("Teks tidak boleh kosong.")
        elif not is_valid_text(user_input):
            st.warning("Masukkan teks yang lengkap dan valid, bukan hanya satu kata atau karakter acak.")
        else:
            with st.spinner("Memproses teks dan memprediksi..."):
                processed = preprocess_text(user_input)
                vectorized = vectorizer.transform([processed])
                prediction = model.predict(vectorized)[0]
                probas = model.predict_proba(vectorized)[0]
                label_map = {1: "Hoax", 0: "Non-Hoax"}
                pred_label = label_map[prediction]

            st.success(f"Prediksi: **{pred_label}**")

            st.subheader("Keyakinan Model:")
            df_proba = pd.DataFrame({"Label": ["Non-Hoax", "Hoax"], "Probabilitas": probas})
            fig = px.pie(df_proba, names="Label", values="Probabilitas", title="Distribusi Probabilitas Prediksi",
                         color_discrete_sequence=["green", "red"])
            st.plotly_chart(fig, use_container_width=True)

            try:
                result = analyze_with_gemini(text=user_input, predicted_label=pred_label, used_links=[],
                                             distribution={"Non-Hoax": f"{probas[0]*100:.1f}", "Hoax": f"{probas[1]*100:.1f}"})
                with st.expander("Hasil Interpretasi LLM"):
                    st.write(result.get('output_mentah', 'Tidak tersedia'))

                hasil_baru = {
                    "Input": user_input,
                    "Preprocessed": processed,
                    "Prediksi Model": pred_label,
                    "Probabilitas Non-Hoax": f"{probas[0]*100:.2f}%",
                    "Probabilitas Hoax": f"{probas[1]*100:.2f}%",
                    "Kebenaran LLM": result.get("kebenaran"),
                    "Alasan LLM": result.get("alasan"),
                    "Ringkasan Berita": result.get("ringkasan"),
                    "Perbandingan": result.get("perbandingan_kebenaran"),
                    "Penjelasan Koreksi": result.get("penjelasan_koreksi")
                }

                simpan_ke_firebase(hasil_baru)
                hasil_semua.append(pd.DataFrame([hasil_baru]))
                st.success("Hasil disimpan ke Firebase Realtime Database")

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat menggunakan LLM:\n{e}")

    if hasil_semua:
        df_hasil = pd.concat(hasil_semua, ignore_index=True)
        csv = df_hasil.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Unduh Hasil (.csv)", data=csv, file_name="hasil_deteksi_berita.csv", mime="text/csv")

# ✅ Menu Dataset
elif selected == "Dataset":
    st.subheader("Dataset Gabungan:")
    st.dataframe(df[["T_judul", "T_konten", "label"]].head())

# ✅ Menu Preprocessing
elif selected == "Preprocessing":
    st.subheader("Hasil Preprocessing:")
    st.dataframe(df[["T_judul", "T_konten", "gabungan"]].head())

# ✅ Menu Evaluasi Model
elif selected == "Evaluasi Model":
    st.subheader("Evaluasi Model Naive Bayes")
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="Akurasi", value=f"{acc*100:.2f}%")

    st.subheader("Laporan Klasifikasi:")
    report = classification_report(y_test, y_pred, target_names=["Non-Hoax", "Hoax"])
    st.text(report)

    st.subheader("Visualisasi Prediksi:")
    df_eval = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    df_eval["Hasil"] = np.where(df_eval["Actual"] == df_eval["Predicted"], "Benar", "Salah")
    hasil_count = df_eval["Hasil"].value_counts().reset_index()
    hasil_count.columns = ["Hasil", "Jumlah"]
    fig_eval = px.pie(hasil_count, names="Hasil", values="Jumlah", title="Distribusi Prediksi Benar vs Salah",
                      color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_eval, use_container_width=True)

    st.subheader("Contoh Data Salah Prediksi:")
    salah = df_eval[df_eval["Hasil"] == "Salah"]
    st.dataframe(salah.head() if not salah.empty else pd.DataFrame([{"Info": "Semua prediksi benar!"}]))

# ✅ Menu Riwayat Prediksi
elif selected == "Riwayat Prediksi":
    st.subheader("Riwayat Prediksi")
    df_riwayat = read_predictions_from_firebase()
    if not df_riwayat.empty:
        kolom_utama = [
            "Input", "Prediksi Model", "Probabilitas Non-Hoax", "Probabilitas Hoax",
            "Kebenaran LLM", "Alasan LLM", "Ringkasan Berita", "Perbandingan", "Penjelasan Koreksi"
        ]
        tampilkan = [col for col in kolom_utama if col in df_riwayat.columns]
        st.dataframe(df_riwayat[tampilkan])
        csv_data = df_riwayat.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh Riwayat (.csv)", data=csv_data, file_name="riwayat_prediksi_firebase.csv", mime="text/csv")
    else:
        st.info("Belum ada data prediksi yang disimpan.")
