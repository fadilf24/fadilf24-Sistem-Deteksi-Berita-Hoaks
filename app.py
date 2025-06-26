import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

from preprocessing import preprocess_text, preprocess_dataframe, load_and_clean_data
from feature_extraction import combine_text_columns, tfidf_transform
from interpretation import configure_gemini, analyze_with_gemini

st.set_page_config(page_title="Deteksi Berita Hoaks", page_icon="📰")
st.title("📰 Deteksi Berita Hoaks (Naive Bayes + Gemini LLM)")

# -----------------------
# 🔍 Sidebar Navigasi
# -----------------------
st.sidebar.title("🔍 Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", (
    "🏠 Deteksi Hoaks", 
    "📂 Dataset", 
    "⚙️ Preprocessing", 
    "📊 Evaluasi Model"
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
    label_map = {"hoaks": 1, "non-hoaks": 0, 1: 1, 0: 0}
    df["label"] = df["label"].map(label_map)
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

df1, df2 = load_dataset()
df = prepare_data(df1, df2)
model, vectorizer, X_test, y_test, y_pred = extract_features_and_model(df)

# -----------------------
# 🏠 Deteksi Hoaks
# -----------------------
if menu == "🏠 Deteksi Hoaks":
    st.subheader("✍️ Masukkan Teks Berita")
    user_input = st.text_area("Contoh: Pemerintah mengumumkan vaksin palsu beredar di Jakarta...")

    if st.button("🔍 Prediksi"):
        if not user_input.strip():
            st.warning("Teks tidak boleh kosong.")
        else:
            processed = preprocess_text(user_input)
            vectorized = vectorizer.transform([processed])
            prediction = model.predict(vectorized)[0]

            label_map = {1: "Hoax", 0: "Non-Hoax"}
            st.success(f"✅ Prediksi: {label_map[prediction]}")

    st.subheader("🧠 Interpretasi dengan Gemini LLM")
    user_input_llm = st.text_area("Masukkan teks berita untuk interpretasi:")
    if st.button("🔎 Interpretasi"):
        if not user_input_llm.strip():
            st.warning("Teks tidak boleh kosong.")
        else:
            try:
                api_key = "AIzaSyDFRv6-gi44fDsJvR_l4E8N2Fxd45oGozU"
                configure_gemini(api_key)
                result = analyze_with_gemini(user_input_llm, true_label="Unknown", predicted_label="Unknown")
                st.success("Hasil Interpretasi LLM:")
                st.text(result)
            except Exception as e:
                st.error(f"❌ Error saat menggunakan Gemini: {e}")

# -----------------------
# 📂 Dataset
# -----------------------
elif menu == "📂 Dataset":
    st.subheader("📁 Dataset 1 (Data_latih.csv):")
    st.write(df1.head())
    st.subheader("📁 Dataset 2 (detik_data.csv):")
    st.write(df2.head())
    st.subheader("📁 Dataset Gabungan:")
    st.write(df[['T_judul', 'T_konten', 'label']].head())

# -----------------------
# ⚙️ Preprocessing
# -----------------------
elif menu == "⚙️ Preprocessing":
    st.subheader("🧼 Hasil Preprocessing:")
    st.write(df[['T_judul', 'T_konten']].head())
    st.subheader("📚 Gabungan Judul + Konten:")
    st.write(df[['gabungan']].head())

# -----------------------
# 📊 Evaluasi Model
# -----------------------
elif menu == "📊 Evaluasi Model":
    st.subheader("📊 Evaluasi Model:")
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="Akurasi", value=f"{acc*100:.2f}%")

    st.subheader("📋 Laporan Klasifikasi:")
    report = classification_report(y_test, y_pred, target_names=["non-hoaks", "hoaks"])
    st.text(report)
