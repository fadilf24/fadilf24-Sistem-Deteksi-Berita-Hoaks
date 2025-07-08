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

    # Ekstrak hasil Gemini
    kebenaran = re.search(r"Kebenaran:\s*(Hoax|Non[- ]?Hoax)", response_text, re.IGNORECASE)
    alasan = re.search(r"Alasan:\s*(.+?)(?:Ringkasan:|Link:|$)", response_text, re.DOTALL)
    ringkasan = re.search(r"Ringkasan:\s*(.+?)(?:Link:|$)", response_text, re.DOTALL)

    # Normalisasi label
    pred_label_clean = predicted_label.strip().lower().replace("-", " ")
    gemini_label_clean = kebenaran.group(1).strip().lower().replace("-", " ") if kebenaran else ""

    perbandingan = "sesuai" if pred_label_clean == gemini_label_clean else "berbeda"

    # Tambahkan penjelasan koreksi jika berbeda
    penjelasan_koreksi = None
    if perbandingan == "berbeda":
        penjelasan_koreksi = (
            f"Model memprediksi label: **{predicted_label}**, namun Gemini menyatakan kebenarannya adalah **{kebenaran.group(1)}**.\n\n"
            f"Hal ini bisa terjadi karena model hanya mengandalkan representasi statistik (TF-IDF + Naive Bayes), "
            f"sedangkan Gemini melakukan analisis semantik dan menggunakan sumber referensi dari situs turnbackhoax.id.\n\n"
            f"**Alasan Gemini:** {alasan.group(1).strip() if alasan else 'Tidak tersedia'}"
        )

    return {
        "kebenaran": kebenaran.group(1).replace("-", " ") if kebenaran else None,
        "alasan": alasan.group(1).strip() if alasan else None,
        "ringkasan": ringkasan.group(1).strip() if ringkasan else None,
        "link_asli": link_asli,
        "output_mentah": response_text,
        "perbandingan_kebenaran": perbandingan,
        "penjelasan_koreksi": penjelasan_koreksi
    }
