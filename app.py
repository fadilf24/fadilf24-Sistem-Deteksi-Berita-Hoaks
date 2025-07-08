import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from streamlit_option_menu import option_menu

from preprocessing import preprocess_text, preprocess_dataframe, load_and_clean_data
from feature_extraction import combine_text_columns, tfidf_transform
from interpretation import configure_gemini, analyze_with_gemini

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Berita Hoaks", page_icon="🔎", layout="wide")

# Sidebar Navigasi
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Deteksi Hoaks", "Dataset", "Preprocessing", "Evaluasi Model"],
        icons=["search", "folder", "tools", "bar-chart"],
        default_index=0,
        orientation="vertical"
    )

st.title("📰 Deteksi Berita Hoaks (Naive Bayes + Gemini LLM)")

# Load Data
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
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, vectorizer, X_test, y_test, y_pred

# Load semua
try:
    df1, df2 = load_dataset()
    df = prepare_data(df1, df2)
    model, vectorizer, X_test, y_test, y_pred = extract_features_and_model(df)
except Exception as e:
    st.error(f"Gagal memuat atau memproses data:\n{e}")
    st.stop()

# ----------------------- Halaman -----------------------
if selected == "Deteksi Hoaks":
    st.subheader("Masukkan Teks Berita:")
    user_input = st.text_area("Contoh: Pemerintah mengumumkan vaksin palsu beredar di Jakarta...", height=200)

    if st.button("Analisis Berita"):
        if not user_input.strip():
            st.warning("Teks tidak boleh kosong.")
        else:
            with st.spinner("Memproses teks dan memprediksi..."):
                processed = preprocess_text(user_input)
                vectorized = vectorizer.transform([processed])
                prediction = model.predict(vectorized)[0]
                label_map = {1: "Hoax", 0: "Non-Hoax"}
                pred_label = label_map[prediction]

            st.success(f"🧠 Prediksi: **{pred_label}**")

            # Plotly Pie Chart Keyakinan Model
            probas = model.predict_proba(vectorized)[0]
            class_labels = ["Non-Hoax", "Hoax"]
            st.subheader("📊 Keyakinan Model:")
            df_proba = pd.DataFrame({
                "Label": class_labels,
                "Probabilitas": probas
            })
            fig = px.pie(
                df_proba,
                names="Label",
                values="Probabilitas",
                title="Distribusi Probabilitas Prediksi",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig, use_container_width=True)

            # Interpretasi Gemini
            try:
                result = analyze_with_gemini(
                    text=user_input,
                    predicted_label=pred_label,
                    used_links=[],
                    distribution=None
                )

                with st.expander("📜 Lihat Output Lengkap Gemini"):
                    st.write(result['output_mentah'])

                if result["perbandingan_kebenaran"] == "sesuai":
                    st.success("✅ Interpretasi Gemini **sesuai** dengan prediksi model.")
                else:
                    st.warning("⚠️ Interpretasi Gemini **berbeda** dari prediksi model.")
                    st.markdown("#### 🤔 Penjelasan Perbedaan:")
                    st.info(result["penjelasan_koreksi"])

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat menggunakan Gemini:\n{e}")

            # Simpan hasil
            hasil_baru = pd.DataFrame([{
                "input": user_input,
                "preprocessed": processed,
                "prediksi": pred_label,
                "interpretasi": result
            }])

            try:
                hasil_baru.to_csv("hasil_prediksi.csv", mode="a", index=False, header=not os.path.exists("hasil_prediksi.csv"))
                st.success("✅ Hasil disimpan ke `hasil_prediksi.csv`")
            except Exception as e:
                st.warning(f"Gagal menyimpan hasil: {e}")

elif selected == "Dataset":
    st.subheader("Dataset Kaggle:")
    st.dataframe(df1.head())
    st.subheader("Dataset Detik.com:")
    st.dataframe(df2.head())
    st.subheader("Dataset Gabungan:")
    st.dataframe(df[["T_judul", "T_konten", "label"]].head())

elif selected == "Preprocessing":
    st.subheader("Hasil Preprocessing:")
    st.dataframe(df[["T_judul", "T_konten"]].head())
    st.subheader("Gabungan Judul + Konten:")
    st.dataframe(df[["gabungan"]].head())

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

    fig_eval = px.pie(
        hasil_count,
        names="Hasil",
        values="Jumlah",
        title="Distribusi Prediksi Benar vs Salah",
        color_discrete_sequence=px.colors.sequential.Blues
    )
    st.plotly_chart(fig_eval, use_container_width=True)

    st.subheader("Contoh Data Salah Prediksi:")
    salah = df_eval[df_eval["Hasil"] == "Salah"]
    if not salah.empty:
        st.dataframe(salah.head())
    else:
        st.success("🎉 Semua prediksi benar!")
