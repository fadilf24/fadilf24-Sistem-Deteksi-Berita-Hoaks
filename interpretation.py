import google.generativeai as genai

def configure_gemini(api_key):
    """
    Konfigurasi API Key Gemini (Google Generative AI).
    """
    genai.configure(api_key=api_key)

def analyze_with_gemini(text, true_label, predicted_label):
    """
    Lakukan interpretasi teks menggunakan model Gemini LLM.
    """
    prompt = f"""
Teks berikut adalah sebuah berita.

Tugas Anda:
1. Tentukan apakah berita ini termasuk 'Hoax' atau 'Non-Hoax'. Jawaban harus diawali dengan: "Kebenaran: ..."
2. Jelaskan alasan mengapa Anda menilai demikian. Harus diawali dengan: "Alasan: ..."
3. Buat ringkasan isi berita maksimal 5 kalimat. Harus diawali dengan: "Ringkasan: ..."

Format output WAJIB:
Kebenaran: [Hoax/Non-Hoax]
Alasan: [penjelasan singkat]
Ringkasan: [ringkasan berita]

Label Asli: {true_label}
Prediksi Naive Bayes: {predicted_label}

Berita:
{text}
"""

    model = genai.GenerativeModel('gemini-2.0-flash')  # pastikan model ini tersedia
    response = model.generate_content(prompt)
    return response.text
