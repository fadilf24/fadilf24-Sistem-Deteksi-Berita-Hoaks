import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_elements import elements, mui, sync
from streamlit_elements.mui.icons import material as icon

from preprocessing import preprocess_text, preprocess_dataframe, load_and_clean_data
from feature_extraction import combine_text_columns, tfidf_transform
from interpretation import configure_gemini, analyze_with_gemini

# -----------------------
# Konfigurasi Aplikasi
# -----------------------
st.set_page_config(page_title="Deteksi Berita Hoaks", page_icon="🔎", layout="wide")
st.title("Deteksi Berita Hoaks (Naive Bayes + Gemini LLM)")

# -----------------------
# Sidebar Navigasi Collapse/Expand
# -----------------------
menu_options = [
    {"label": "Deteksi Hoaks", "key": "Deteksi Hoaks", "icon": icon.Search},
    {"label": "Dataset", "key": "Dataset", "icon": icon.Folder},
    {"label": "Preprocessing", "key": "Preprocessing", "icon": icon.Build},
    {"label": "Evaluasi Model", "key": "Evaluasi Model", "icon": icon.BarChart},
]

if "selected" not in st.session_state:
    st.session_state.selected = "Deteksi Hoaks"
if "sidebar_expanded" not in st.session_state:
    st.session_state.sidebar_expanded = True

with elements("sidebar"):
    with mui.Box(sx={"height": "100vh", "backgroundColor": "#0e1117"}):
        with mui.Stack(spacing=1, direction="column", sx={"p": 1, "alignItems": "center"}):
            mui.IconButton(
                icon.Menu() if not st.session_state.sidebar_expanded else icon.Close(),
                color="primary",
                onClick=lambda: st.session_state.update({"sidebar_expanded": not st.session_state.sidebar_expanded}),
                title="Toggle Sidebar"
            )
            for option in menu_options:
                with mui.Box(sx={"textAlign": "center"}):
                    mui.IconButton(
                        option["icon"](),
                        color="primary" if st.session_state.selected == option["key"] else "default",
                        onClick=lambda label=option["key"]: st.session_state.update({"selected": label}),
                        title=option["label"]
                    )
                    if st.session_state.sidebar_expanded:
                        mui.Typography(option["label"], variant="caption", sx={"color": "white"})

menu = st.session_state.get("selected", "Deteksi Hoaks")

# -----------------------
# Load & Preprocess Data
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
    label_map = {"Hoax": 1, "Non-Hoax": 0, 1: 1, 0: 0}
    df["label"] = df["label"].map(label_map)
    df = df[df["label"].notna()]
    df["label"] = df["label"].astype(int)
    return df

@st.cache_data
def extract_features_and_model(df):
    X, vectorizer = tfidf_transform(df["gabungan"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, vectorizer, X_test, y_test, y_pred

# -----------------------
# Inisialisasi
# -----------------------
try:
    df1, df2 = load_dataset()
    df = prepare_data(df1, df2)
    model, vectorizer, X_test, y_test, y_pred = extract_features_and_model(df)
except Exception as e:
    st.error(f"Gagal memuat atau memproses data:\n{e}")
    st.stop()

# -----------------------
# Halaman: Deteksi Hoaks
# -----------------------
if menu == "Deteksi Hoaks":
    st.subheader("Masukkan Teks Berita:")
    user_input = st.text_area(
        "Contoh: Pemerintah mengumumkan vaksin palsu beredar di Jakarta...", height=200
    )

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

            st.success(f"Prediksi: {pred_label}")

            # Probabilitas Model (Pie Chart)
            probas = model.predict_proba(vectorized)[0]
            class_labels = ["Non-Hoax", "Hoax"]

            st.subheader("Keyakinan Model (Pie Chart):")
            fig1, ax1 = plt.subplots()
            ax1.pie(probas, labels=class_labels, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)

            # Interpretasi dengan Gemini
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                with st.spinner("Menghasilkan interpretasi dengan Gemini..."):
                    configure_gemini(api_key)
                    result = analyze_with_gemini(
                        user_input, true_label="Unknown", predicted_label=pred_label
                    )
                st.markdown("### Hasil Interpretasi LLM:")
                st.text(result)
            except Exception as e:
                result = "Gagal interpretasi"
                st.error(f"Gagal menghasilkan interpretasi LLM:\n{e}")

            # Simpan ke CSV
            hasil_baru = pd.DataFrame([{
                "input": user_input,
                "preprocessed": processed,
                "prediksi": pred_label,
                "interpretasi": result
            }])

            try:
                hasil_baru.to_csv(
                    "hasil_prediksi.csv",
                    mode="a",
                    index=False,
                    header=not os.path.exists("hasil_prediksi.csv")
                )
                st.success("Hasil prediksi disimpan ke `hasil_prediksi.csv`")
            except Exception as e:
                st.warning(f"Gagal menyimpan hasil: {e}")

# -----------------------
# Halaman: Dataset
# -----------------------
elif menu == "Dataset":
    st.subheader("Dataset Kaggle (Data_latih.csv):")
    st.dataframe(df1.head())
    st.subheader("Dataset Detik.com (detik_data.csv):")
    st.dataframe(df2.head())
    st.subheader("Dataset Gabungan:")
    st.dataframe(df[["T_judul", "T_konten", "label"]].head())

# -----------------------
# Halaman: Preprocessing
# -----------------------
elif menu == "Preprocessing":
    st.subheader("Hasil Preprocessing:")
    st.dataframe(df[["T_judul", "T_konten"]].head())
    st.subheader("Gabungan Judul + Konten:")
    st.dataframe(df[["gabungan"]].head())

# -----------------------
# Halaman: Evaluasi Model
# -----------------------
elif menu == "Evaluasi Model":
    st.subheader("Evaluasi Model Naive Bayes")
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="Akurasi", value=f"{acc*100:.2f}%")

    st.subheader("Laporan Klasifikasi:")
    report = classification_report(
        y_test, y_pred, target_names=["Non-Hoax", "Hoax"]
    )
    st.text(report)

    st.subheader("Visualisasi Prediksi :")
    df_eval = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    df_eval["Hasil"] = np.where(df_eval["Actual"] == df_eval["Predicted"], "Benar", "Salah")
    hasil_count = df_eval["Hasil"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = sns.color_palette("pastel")[0:2]
    ax.pie(hasil_count, labels=hasil_count.index, colors=colors, autopct="%.1f%%", startangle=90)
    ax.set_title("Distribusi Prediksi Benar vs Salah", fontsize=14)
    ax.axis("equal")
    st.pyplot(fig)

    st.subheader("Contoh Data Salah Prediksi:")
    salah = df_eval[df_eval["Hasil"] == "Salah"]
    if not salah.empty:
        st.dataframe(salah.head())
    else:
        st.success("Semua prediksi benar!")
