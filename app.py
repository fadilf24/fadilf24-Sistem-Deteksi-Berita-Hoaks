import streamlit as st
import pandas as pd
import numpy as np
import re
import uuid
from datetime import datetime
import pytz
import plotly.express as px

from streamlit_option_menu import option_menu
import firebase_admin
from firebase_admin import credentials, db

# =========================
# 🔹 LOCAL MODULES
# =========================
from preprocessing import (
    preprocess_text,
    preprocess_dataframe,
    load_and_clean_data,
    preprocess_with_steps
)
from feature_extraction import combine_text_columns, tfidf_transform
from interpretation import analyze_with_gemini

from classification import split_data, train_fuzzy_classifier, predict_fuzzy

# =========================
# 🔹 PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Deteksi Berita Hoaks",
    page_icon="🔎",
    layout="wide"
)

st.title("📰 Deteksi Berita Hoaks (Fuzzy Classification + LLM)")

# =========================
# 🔹 FIREBASE CONFIG
# =========================
firebase_cred = dict(st.secrets["FIREBASE_KEY"])
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_cred)
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://deteksi-hoaks-streamlit-default-rtdb.asia-southeast1.firebasedatabase.app/"
    })

def simpan_ke_firebase(data: dict):
    tz = pytz.timezone("Asia/Jakarta")
    data["timestamp"] = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    ref = db.reference("prediksi_hoaks")
    ref.child(str(uuid.uuid4())).set(data)

def read_predictions_from_firebase():
    ref = db.reference("prediksi_hoaks")
    data = ref.get()
    return pd.DataFrame(data.values()) if data else pd.DataFrame()

# =========================
# 🔹 SIDEBAR
# =========================
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=[
            "Deteksi Hoaks",
            "Dataset",
            "Preprocessing",
            "Evaluasi Model",
            "Riwayat Prediksi"
        ],
        icons=["search", "folder", "tools", "bar-chart", "clock-history"],
        default_index=0
    )

# =========================
# 🔹 DATA LOADING
# =========================
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
def build_model(df):
    X, vectorizer = tfidf_transform(df["T_text"])
    y = df["label"].values

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_fuzzy_classifier(X_train, y_train)
    y_pred = predict_fuzzy(model, X_test)

    return model, vectorizer, X_test, y_test, y_pred

def is_valid_text(text: str) -> bool:
    words = re.findall(r"\w+", text)
    return len(words) >= 5 and any(len(word) > 3 for word in words)

# =========================
# 🔹 LOAD ALL
# =========================
try:
    df1, df2 = load_dataset()
    df = prepare_data(df1, df2)
    model, vectorizer, X_test, y_test, y_pred = build_model(df)
except Exception as e:
    st.error(f"Gagal memuat data atau model:\n{e}")
    st.stop()

# =========================
# 🔹 MENU: DETEKSI HOAKS
# =========================
if selected == "Deteksi Hoaks":
    st.subheader("✍️ Masukkan Teks Berita")
    user_input = st.text_area(
        "Contoh: Pemerintah mengumumkan vaksin palsu beredar di Jakarta...",
        height=200
    )

    if st.button("🔍 Analisis Berita"):
        if not user_input.strip():
            st.warning("Teks tidak boleh kosong.")
        elif not is_valid_text(user_input):
            st.warning("Teks terlalu pendek atau tidak valid.")
        else:
            with st.spinner("Memproses dan memprediksi..."):
                processed = preprocess_text(user_input)
                vectorized = vectorizer.transform([processed])

                fuzzy_values = model.fuzzy_predict(vectorized)[0]
                prediction = model.predict(vectorized)[0]

                label_map = {1: "Hoax", 0: "Non-Hoax"}
                pred_label = label_map[prediction]

                probas = {
                    "Non-Hoax": float(fuzzy_values[0]),
                    "Hoax": float(fuzzy_values[1])
                }

            st.success(f"🧠 Prediksi Model: **{pred_label}**")

            df_proba = pd.DataFrame({
                "Label": probas.keys(),
                "Membership": probas.values()
            })

            fig = px.pie(
                df_proba,
                names="Label",
                values="Membership",
                title="Distribusi Fuzzy Membership"
            )
            st.plotly_chart(fig, use_container_width=True)

            try:
                llm_result = analyze_with_gemini(
                    text=user_input,
                    predicted_label=pred_label,
                    used_links=[],
                    distribution={
                        k: f"{v*100:.2f}%"
                        for k, v in probas.items()
                    }
                )

                with st.expander("🤖 Interpretasi LLM"):
                    st.write(llm_result.get("output_mentah", "-"))

                hasil = {
                    "Input": user_input,
                    "Preprocessed": processed,
                    "Prediksi Model": pred_label,
                    "Fuzzy Non-Hoax": f"{probas['Non-Hoax']*100:.2f}%",
                    "Fuzzy Hoax": f"{probas['Hoax']*100:.2f}%",
                    "Kebenaran LLM": llm_result.get("kebenaran"),
                    "Alasan LLM": llm_result.get("alasan"),
                    "Ringkasan Berita": llm_result.get("ringkasan"),
                    "Perbandingan": llm_result.get("perbandingan_kebenaran"),
                    "Penjelasan Koreksi": llm_result.get("penjelasan_koreksi")
                }

                simpan_ke_firebase(hasil)
                st.success("✅ Hasil disimpan ke Firebase")

            except Exception as e:
                st.error(f"LLM Error:\n{e}")

# =========================
# 🔹 MENU: DATASET
# =========================
elif selected == "Dataset":
    st.subheader("📂 Dataset Hoaks")
    st.dataframe(df1)

    st.subheader("📂 Dataset Non-Hoaks (Detik)")
    st.dataframe(df2)

# =========================
# 🔹 MENU: PREPROCESSING
# =========================
elif selected == "Preprocessing":
    st.subheader("🔧 Tahapan Preprocessing")

    contoh_teks = df["text"].iloc[0]
    hasil = preprocess_with_steps(contoh_teks)

    df_steps = pd.DataFrame([
        {"Tahap": k, "Hasil": v}
        for k, v in hasil.items()
    ])

    st.dataframe(df_steps, use_container_width=True)

# =========================
# 🔹 MENU: EVALUASI MODEL
# =========================
elif selected == "Evaluasi Model":
    from sklearn.metrics import accuracy_score, classification_report

    st.subheader("📊 Evaluasi Fuzzy Classification")

    acc = accuracy_score(y_test, y_pred)
    st.metric("Akurasi", f"{acc*100:.2f}%")

    st.text(classification_report(
        y_test,
        y_pred,
        target_names=["Non-Hoax", "Hoax"]
    ))

# =========================
# 🔹 MENU: RIWAYAT
# =========================
elif selected == "Riwayat Prediksi":
    st.subheader("🕒 Riwayat Prediksi")

    df_hist = read_predictions_from_firebase()
    if not df_hist.empty:
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
        df_hist = df_hist.sort_values("timestamp", ascending=False)

        st.dataframe(df_hist, use_container_width=True)

        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Unduh Riwayat",
            csv,
            "riwayat_prediksi.csv",
            "text/csv"
        )
    else:
        st.info("Belum ada data prediksi.")


