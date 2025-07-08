import google.generativeai as genai
import re

def configure_gemini(api_key):
    """
    Mengatur API key untuk Google Gemini.
    """
    genai.configure(api_key=api_key)

def analyze_with_gemini(text, predicted_label, used_links=None, example_url=None, distribution=None):
    """
    Menganalisis teks berita menggunakan Gemini berdasarkan hasil prediksi model Naive Bayes.
    Mengembalikan hasil interpretasi, serta menjelaskan jika hasil Gemini berbeda dengan prediksi model.
    """

    distribusi_str = ""
    if distribution:
        distribusi_str = "\nDistribusi Prediksi Model (dalam persen):\n"
        distribusi_str += "\n".join([f"- {label}: {percentage}%" for label, percentage in distribution.items()])

    prompt = f"""
Teks berikut adalah sebuah berita.

Tugas Anda:
1. Tentukan apakah berita ini termasuk 'Hoax' atau 'Non-Hoax'. Jawaban diawali dengan: "Kebenaran: ..."
2. Jelaskan secara singkat mengapa Anda menilai demikian. Jika Anda menggunakan referensi atau sumber, cantumkan juga sumbernya di bagian alasan. Jawaban diawali dengan: "Alasan: ..."
3. Buat ringkasan isi berita maksimal dalam 5 kalimat. Jawaban diawali dengan: "Ringkasan: ..."

Prediksi model Naive Bayes untuk berita ini: {predicted_label}
{distribusi_str}

Teks Berita:
{text}
"""

    if example_url:
        prompt += f"\nReferensi tambahan: {example_url}"

    # Gunakan model Gemini
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    response_text = response.text.strip()

    # Inisialisasi default
    kebenaran_val = None
    alasan_val = None
    ringkasan_val = None

    # Gunakan regex robust
    try:
        kebenaran_match = re.search(r"Kebenaran:\s*(Hoax|Non[- ]?Hoax)", response_text, re.IGNORECASE)
        if kebenaran_match:
            kebenaran_val = kebenaran_match.group(1).strip().replace("-", " ")

        alasan_match = re.search(r"Alasan:\s*(.*?)(?:\n(?:Ringkasan|$))", response_text, re.DOTALL | re.IGNORECASE)
        if alasan_match:
            alasan_val = alasan_match.group(1).strip()

        ringkasan_match = re.search(r"Ringkasan:\s*(.*?)(?:\n|$)", response_text, re.DOTALL | re.IGNORECASE)
        if ringkasan_match:
            ringkasan_val = ringkasan_match.group(1).strip()

    except Exception as e:
        # Jika parsing gagal, jangan hentikan aplikasi
        alasan_val = f"Gagal memproses respons Gemini: {e}"

    # Normalisasi label
    pred_label_clean = predicted_label.strip().lower().replace("-", " ") if predicted_label else ""
    gemini_label_clean = kebenaran_val.lower() if kebenaran_val else ""

    perbandingan = "sesuai" if pred_label_clean == gemini_label_clean else "berbeda"

    # Tambahkan penjelasan koreksi jika berbeda
    penjelasan_koreksi = None
    if perbandingan == "berbeda":
        penjelasan_koreksi = (
            f"Model memprediksi label: **{predicted_label}**, namun Gemini menyatakan kebenarannya adalah **{kebenaran_val}**.\n\n"
            f"Hal ini bisa terjadi karena model hanya mengandalkan representasi statistik (TF-IDF + Naive Bayes), "
            f"sedangkan Gemini melakukan analisis semantik yang lebih dalam.\n\n"
            f"**Alasan Gemini:** {alasan_val or 'Tidak tersedia'}"
        )

    return {
        "kebenaran": kebenaran_val,
        "alasan": alasan_val,
        "ringkasan": ringkasan_val,
        "output_mentah": response_text,
        "perbandingan_kebenaran": perbandingan,
        "penjelasan_koreksi": penjelasan_koreksi
    }
