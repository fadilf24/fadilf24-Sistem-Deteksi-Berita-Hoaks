import streamlit as st
import pandas as pd
from preprocessing import preprocess_text  # digunakan untuk input teks
from feature_extraction import combine_text_columns, tfidf_transform
from classification import split_data, train_naive_bayes, predict_naive_bayes
from evaluation import evaluate_model, generate_classification_report
from interpretation import configure_gemini, analyze_with_gemini
from sklearn.preprocessing import LabelEncoder

st.title("Aplikasi Deteksi Berita Hoaks Menggunakan Naive Bayes & LLM (Gemini)")

#  API Key Gemini (JANGAN gunakan langsung di production)
api_key = "AIzaSyDFRv6-gi44fDsJvR_l4E8N2Fxd45oGozU"

# Load hasil preprocessing
try:
    df = pd.read_csv("hasil_preprocessing.csv")
    st.success("Berhasil memuat data hasil preprocessing!")
except FileNotFoundError:
    st.error("File hasil_preprocessing.csv tidak ditemukan.")
    st.stop()

# Tampilkan data awal
st.subheader("Data Setelah Preprocessing (CSV):")
st.write(df[['T_judul', 'T_konten']].head())

# Gabungkan teks
df = combine_text_columns(df)
st.subheader("Gabungan Judul & Konten:")
st.write(df[['gabungan']].head())

# TF-IDF
X_features, vectorizer = tfidf_transform(df['gabungan'])

# Encode label
le = LabelEncoder()
y = le.fit_transform(df['label'])

# Split data
X_train, X_test, y_train, y_test = split_data(X_features, y)

# Train & predict
model = train_naive_bayes(X_train, y_train)
y_pred = predict_naive_bayes(model, X_test)

# Evaluasi
metrics = evaluate_model(y_test, y_pred)
report = generate_classification_report(y_test, y_pred, target_names=le.classes_)

st.subheader("Hasil Evaluasi Model:")
st.json(metrics)

st.subheader("Laporan Klasifikasi Lengkap:")
st.text(report)

# Prediksi teks baru
st.subheader("Prediksi Berita Baru:")
user_input = st.text_area("Masukkan teks berita untuk diprediksi:")

if st.button("Prediksi"):
    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    prediction = predict_naive_bayes(model, input_vector)
    predicted_label = le.inverse_transform(prediction)
    st.success(f"Hasil Prediksi: {predicted_label[0]}")

# Interpretasi dengan Gemini
st.subheader("Interpretasi Pengetahuan dengan LLM (Gemini):")
user_input_llm = st.text_area("Masukkan teks berita untuk interpretasi LLM:")

if user_input_llm and st.button("Interpretasi dengan Gemini LLM"):
    try:
        configure_gemini(api_key)
        result = analyze_with_gemini(user_input_llm, true_label="Unknown", predicted_label="Unknown")
        st.success("Hasil Interpretasi LLM:")
        st.text(result)
    except Exception as e:
        st.error(f"Terjadi error saat menggunakan Gemini: {e}")
