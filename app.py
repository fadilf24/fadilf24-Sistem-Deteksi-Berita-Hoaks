import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from preprocessing import preprocess_text
from feature_extraction import combine_text_columns, tfidf_transform
from classification import split_data, train_naive_bayes, predict_naive_bayes
from evaluation import evaluate_model, generate_classification_report, plot_confusion_matrix
from interpretation import configure_gemini, analyze_with_gemini

st.set_page_config(page_title="Sistem Deteksi Hoaks")

st.title("Deteksi Berita Hoaks (Naive Bayes + LLM Gemini)")

# ✅ API Key Gemini (Hanya untuk demo)
api_key = "AIzaSyDFRv6-gi44fDsJvR_l4E8N2Fxd45oGozU"

# ✅ Load dataset hasil preprocessing
try:
    df_preprocessed = pd.read_csv("hasil_preprocessing.csv")
except FileNotFoundError:
    st.error(" File 'hasil_preprocessing.csv' tidak ditemukan.")
    st.stop()

# ✅ TF-IDF dan pelatihan model
df_preprocessed = combine_text_columns(df_preprocessed)
X_features, vectorizer = tfidf_transform(df_preprocessed["gabungan"])

# Encode label
le = LabelEncoder()
y = le.fit_transform(df_preprocessed["label"])

# Split, train, predict
X_train, X_test, y_train, y_test = split_data(X_features, y)
model = train_naive_bayes(X_train, y_train)
y_pred = predict_naive_bayes(model, X_test)

# ✅ SIDEBAR menu navigasi
st.sidebar.title("🔧 Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", (" Home", " Dataset", " Evaluasi Model"))

# ✅ HOME PAGE
if menu == " Home":
    st.subheader(" Masukkan Teks Berita untuk Deteksi:")

    user_input = st.text_area("Contoh: Pemerintah mengumumkan vaksin palsu beredar di Jakarta...")

    if st.button("🔍 Prediksi & Interpretasi"):
        if not user_input.strip():
            st.warning("Teks tidak boleh kosong.")
        else:
            processed = preprocess_text(user_input)
            input_vector = vectorizer.transform([processed])
            prediction = predict_naive_bayes(model, input_vector)
            predicted_label = le.inverse_transform(prediction)[0]

            st.success(f"✅ Prediksi: {predicted_label}")

            # Interpretasi LLM
            try:
                configure_gemini(api_key)
                result = analyze_with_gemini(
                    user_input,
                    true_label="Unknown",
                    predicted_label=predicted_label
                )
                st.subheader("🧠 Interpretasi LLM (Gemini):")
                st.text(result)
            except Exception as e:
                st.error(f"❌ Error saat menggunakan Gemini: {e}")

# ✅ DATASET PAGE
elif menu == " Dataset":
    try:
        df1 = pd.read_csv("Data_latih.csv")
        df2 = pd.read_csv("detik_data.csv")
        st.subheader("Dataset 1 (Data_latih.csv):")
        st.write(df1.head())
        st.subheader("Dataset 2 (detik_data.csv):")
        st.write(df2.head())
    except:
        st.warning("File Data_latih.csv atau detik_data.csv tidak ditemukan.")

# ✅ EVALUASI PAGE
elif menu == "Evaluasi Model":
    st.subheader("Evaluasi Model Naive Bayes")

    if st.button("Tampilkan Evaluasi"):
        metrics = evaluate_model(y_test, y_pred)
        report = generate_classification_report(y_test, y_pred, target_names=le.classes_)

        st.subheader("Hasil Evaluasi:")
        st.json(metrics)

        st.subheader("Laporan Klasifikasi:")
        st.text(report)

        st.subheader("Confusion Matrix:")
        fig = plot_confusion_matrix(y_test, y_pred, labels=le.classes_)  # Kembalikan figure dari fungsi plot_confusion_matrix
        st.pyplot(fig)

