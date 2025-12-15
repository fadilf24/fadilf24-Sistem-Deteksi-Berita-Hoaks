import google.generativeai as genai
import re
from typing import Dict, Optional, List

def configure_gemini(api_key: str) -> None:
    """
    Mengatur API key untuk Google Gemini.
    """
    genai.configure(api_key=api_key)

def analyze_with_gemini(
    text: str,
    predicted_label: str,
    used_links: Optional[List[str]] = None,
    distribution: Optional[Dict[str, str]] = None
) -> Dict[str, Optional[str]]:
    """
    Menganalisis teks berita menggunakan Gemini LLM
    berdasarkan hasil Fuzzy Classification.

    Parameters:
    - text              : teks berita asli
    - predicted_label   : hasil akhir model fuzzy (Hoax / Non-Hoax)
    - distribution      : derajat keanggotaan fuzzy per kelas
                           contoh:
                           {
                               "Non-Hoax": "62.50%",
                               "Hoax": "37.50%"
                           }
    """

    # =========================
    # 🔹 Distribusi Fuzzy
    # =========================
    distribusi_str = ""
    if distribution:
        distribusi_str = "\nDistribusi Fuzzy Membership Model:\n"
        distribusi_str += "\n".join(
            [f"- {label}: {value}" for label, value in distribution.items()]
        )

    # =========================
    # 🔹 Prompt Gemini
    # =========================
    prompt = f"""
Teks berikut adalah sebuah berita.

Model deteksi otomatis menggunakan metode **Fuzzy Classification**
yang menghasilkan tingkat keyakinan (derajat keanggotaan) pada setiap kelas.

Hasil model:
- Prediksi akhir: {predicted_label}
{distribusi_str}

Tugas Anda:
1. Tentukan apakah berita ini termasuk **Hoax** atau **Non-Hoax**.
   Jawaban diawali dengan: **"Kebenaran:"**
2. Jelaskan secara singkat alasan penilaian Anda berdasarkan isi berita.
   Jawaban diawali dengan: **"Alasan:"**
3. Buat ringkasan isi berita maksimal 5 kalimat.
   Jawaban diawali dengan: **"Ringkasan:"**

Catatan:
- Model Fuzzy memungkinkan adanya ketidakpastian.
- Anda boleh berbeda pendapat dengan model bila isi berita menunjukkan hal lain.

Teks Berita:
{text}
"""

    # =========================
    # 🔹 Generate Respons
    # =========================
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        response_text = response.text.strip()
    except Exception as e:
        return {
            "kebenaran": None,
            "alasan": f"Error Gemini: {str(e)}",
            "ringkasan": None,
            "output_mentah": f"Error: {str(e)}",
            "perbandingan_kebenaran": None,
            "penjelasan_koreksi": None
        }

    # =========================
    # 🔹 Parsing Respons
    # =========================
    kebenaran_val = None
    alasan_val = None
    ringkasan_val = None

    try:
        kebenaran_match = re.search(
            r"Kebenaran:\s*(Hoax|Non[- ]?Hoax)",
            response_text,
            re.IGNORECASE
        )
        if kebenaran_match:
            kebenaran_val = (
                kebenaran_match.group(1)
                .replace("-", " ")
                .title()
            )

        alasan_match = re.search(
            r"Alasan:\s*(.*?)(?:\n(?:Ringkasan|$))",
            response_text,
            re.DOTALL | re.IGNORECASE
        )
        if alasan_match:
            alasan_val = alasan_match.group(1).strip()

        ringkasan_match = re.search(
            r"Ringkasan:\s*(.*?)(?:\n|$)",
            response_text,
            re.DOTALL | re.IGNORECASE
        )
        if ringkasan_match:
            ringkasan_val = ringkasan_match.group(1).strip()

    except Exception as e:
        alasan_val = f"Gagal memproses respons LLM: {e}"

    # ✅ FIX: Handle None values untuk perbandingan
    pred_clean = predicted_label.lower().replace("-", " ") if predicted_label else ""
    llm_clean = kebenaran_val.lower().replace("-", " ") if kebenaran_val else ""

    perbandingan = "sesuai" if pred_clean == llm_clean else "berbeda"

    penjelasan_koreksi = None
    if perbandingan == "berbeda":
        penjelasan_koreksi = (
            f"Model Fuzzy Classification memprediksi berita ini sebagai **{predicted_label}**, "
            f"namun hasil interpretasi oleh LLM menyatakan bahwa berita ini termasuk **{kebenaran_val}**.\n\n"
            f"Perbedaan ini dapat terjadi karena metode Fuzzy Classification bekerja berdasarkan "
            f"kemiripan pola kata dalam representasi numerik teks, sehingga masih memungkinkan "
            f"adanya ketidakpastian atau ambiguitas dalam klasifikasi.\n\n"
            f"Sebaliknya, LLM mampu memahami konteks, alur cerita, serta kewajaran informasi "
            f"secara semantik, sehingga dapat memberikan penilaian yang lebih mendekati "
            f"pemahaman manusia.\n\n"
            f"**Alasan dari LLM:** {alasan_val or 'Tidak tersedia'}"
        )
    
    return {
        "kebenaran": kebenaran_val,
        "alasan": alasan_val,
        "ringkasan": ringkasan_val,
        "output_mentah": response_text,
        "perbandingan_kebenaran": perbandingan,
        "penjelasan_koreksi": penjelasan_koreksi
    }
