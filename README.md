# 🚄 KAI Access Review Bombing & Anomaly Detector — Streamlit Dashboard

> **Riset Penambangan Data (Data Mining) — DIMAS-TI 2026**  
> **Fakultas Matematika dan Ilmu Pengetahuan Alam (FMIPA), Universitas Negeri Semarang (UNNES)**

Aplikasi dashboard interaktif berbasis **Streamlit** untuk mendeteksi serangan *review bombing* dan lonjakan anomali ulasan terkoordinasi pada aplikasi **Access by KAI** menggunakan kerangka kerja hibrida multi-view.

---

## 📌 Fitur Utama Dashboard

1. **📊 Executive Overview & Metrik KPI**:
   - Visualisasi distribusi 4 Kuadran Perilaku (Ulasan Organik, Krisis Layanan, Keluhan Umum/Spam, Review Bombing).
   - Analisis tren temporal harian dan per jam untuk memantau lonjakan abnormalitas.
   - Filter dinamis berdasarkan kuadran, rating bintang, dan ambang batas *Suspicion Score*.

2. **🔬 Multi-View Feature Explorer**:
   - **View 1 (Temporal & Goh-Barabási)**: Pemodelan proses stokastik jeda waktu ($\tau$) dan parameter *Burstiness* ($B \in [-1, 1]$).
   - **View 2 (Leksikon Fungsional Kereta Api)**: Analisis spesifisitas teknis (tiket, booking, refund, reschedule, kaipay, server down) versus makian/spam generik.
   - **View 3 (Co-Review Network & Louvain Community)**: Graf kemiripan ulasan dan deteksi komunitas sindikat terkoordinasi.

3. **🧪 Live Review Simulator & Tester**:
   - Pengujian ulasan baru secara real-time.
   - Normalisasi teks, pembersihan slang, perhitungan *Specificity Score*, dan klasifikasi kuadran otomatis.

4. **📑 Dokumentasi Riset & Makalah IEEE**:
   - Tampilan visualisasi komprehensif multi-panel hasil riset.
   - Grafik perbandingan benchmark multi-view framework vs baseline AutoML (FLAML).

5. **📋 Data Explorer & Export**:
   - Penelusuran data hasil deteksi anomali dengan pencarian teks dan fitur ekspor CSV langsung.

---

## 🚀 Cara Menjalankan

### Menggunakan Batch Script (Windows):
Cukup klik ganda file `run_dashboard.bat`.

### Menggunakan Terminal / Command Prompt:
```bash
cd C:\projects\kai-reviewbombing-dashboard
python -m streamlit run app.py
```
Dashboard akan otomatis terbuka di browser pada alamat `http://localhost:8501`.
