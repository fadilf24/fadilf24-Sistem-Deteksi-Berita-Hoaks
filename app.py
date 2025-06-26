import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from preprocessing import preprocess_text, preprocess_dataframe, load_and_clean_data
from feature_extraction import combine_text_columns, tfidf_transform
from classification import split_data, train_naive_bayes, predict_naive_bayes
from evaluation import evaluate_model, generate_classification_report
from interpretation import configure_gemini, analyze_with_gemini

st.set_page_config(page_title="Deteksi Berita Hoaks", page_icon="📰")
st.title("📰 Deteksi Berita Hoaks (Naive Bayes + Gemini LLM)")

# ✅ Sidebar navigasi
st.sidebar.title("🔍 Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ("🏠 Deteksi Hoaks", "📂 Dataset", "⚙️ Preprocessing", "📊 Evaluasi Model"))

# ✅ Load dan proses data
df1 = pd.read_csv("Data_latih.csv")
df2 = pd.read_csv("detik_data.csv")
df = load_and_clean_data(df1, df2)
df = preprocess_dataframe(df)
df = combine_text_columns(df)
X_features, vectorizer = tfidf_transform(df["gabungan"])

le = LabelEncoder()
y = le.fit_transform(df["label"])

X_train, X_test, y_train, y_test = split_data(X_features, y)
model = train_naive_bayes(X_train, y_train)
y_pred = predict_naive_bayes(model, X_test)

# ✅ Halaman utama: Deteksi Hoaks
if menu == "🏠 Deteksi Hoaks":
    st.subheader("✍️ Masukkan Teks Berita")

    user_input = st.text_area("Contoh: Pemerintah mengumumkan vaksin palsu beredar di Jakarta...")

    if st.button("🔍 Prediksi"):
        if not user_input.strip():
            st.warning("Teks tidak boleh kosong.")
        else:
            processed = preprocess_text(user_input)
            input_vector = vectorizer.transform([processed])
            prediction = predict_naive_bayes(model, input_vector)
            predicted_label = le.inverse_transform(prediction)[0]
            st.success(f"✅ Prediksi: {predicted_label}")

    st.subheader("🧠 Interpretasi dengan Gemini LLM")
    user_input_llm = st.text_area("Masukkan teks berita untuk interpretasi:")
    if st.button("🔎 Interpretasi"):
        if not user_input_llm.strip():
            st.warning("Teks tidak boleh kosong.")
        else:
            try:
                api_key = "AIzaSyDFRv6-gi44fDsJvR_l4E8N2Fxd45oGozU"  # Ganti di production
                configure_gemini(api_key)
                result = analyze_with_gemini(user_input_llm, true_label="Unknown", predicted_label="Unknown")
                st.success("Hasil Interpretasi LLM:")
                st.text(result)
            except Exception as e:
                st.error(f"❌ Error saat menggunakan Gemini: {e}")

# ✅ Dataset Page
elif menu == "📂 Dataset":
    st.subheader("📁 Dataset 1 (data_latih.csv):")
    st.write(df1.head())
    st.subheader("📁 Dataset 2 (detik_data.csv):")
    st.write(df2.head())
    st.subheader("📁 Gabungan Setelah Preprocessing:")
    st.write(df[['T_judul', 'T_konten', 'label']].head())

# ✅ Preprocessing Page
elif menu == "⚙️ Preprocessing":
    st.subheader("🧼 Data Setelah Preprocessing:")
    st.write(df[['T_judul', 'T_konten']].head())
    st.subheader("🧾 Teks Gabungan (judul + konten):")
    st.write(df[['gabungan']].head())

# ✅ Evaluasi Page
elif menu == "📊 Evaluasi Model":
    metrics = evaluate_model(y_test, y_pred)
    report = generate_classification_report(y_test, y_pred, target_names=le.classes_)

    st.subheader("📊 Hasil Evaluasi:")
    st.json(metrics)
    st.subheader("📝 Laporan Klasifikasi:")
    st.text(report)
