import os
import re
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── Page Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KAI Access Review Bombing Detector — DIMAS-TI 2026",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Minimalist Dark Theme CSS ──────────────────────────────────────────
st.markdown("""
<style>
  /* Header styling */
  .main-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 24px;
  }
  .main-title {
    font-size: 28px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 6px;
  }
  .sub-title {
    font-size: 14px;
    color: #94a3b8;
    line-height: 1.5;
  }
  .badge-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 6px;
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
  }
  .kpi-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }
  .kpi-val {
    font-size: 26px;
    font-weight: 700;
    color: #f1f5f9;
  }
  .kpi-lbl {
    font-size: 12px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
  }
</style>
""", unsafe_allow_html=True)

# ── Paths & Data Loading ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "hasil_deteksi_anomali_kai_15000.csv"
MODEL_PATH = BASE_DIR / "models" / "model_detector_multiview_kai.joblib"
ASSETS_DIR = BASE_DIR / "assets"

@st.cache_data
def load_detection_dataset():
    if not DATA_PATH.exists():
        st.error(f"Dataset tidak ditemukan pada: {DATA_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    if 'at' in df.columns:
        df['at'] = pd.to_datetime(df['at'], errors='coerce')
        df['date'] = df['at'].dt.date
        df['hour'] = df['at'].dt.floor('h')
    return df

@st.cache_resource
def load_detector_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model: {e}")
        return None

df_raw = load_detection_dataset()
model_bundle = load_detector_model()

# ── Sidebar Navigation & Filter ────────────────────────────────────────────────
st.sidebar.markdown("### 🚄 KAI Anomaly Detector")
st.sidebar.markdown(
    "<span class='badge-tag'>DIMAS-TI 2026</span><span class='badge-tag'>Multi-View NLP</span>", 
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigasi Modul:",
    [
        "📊 Executive Overview & KPI",
        "🔬 Multi-View Feature Explorer",
        "🧪 Live Review Simulator",
        "📑 Riset & Makalah Ilmiah IEEE",
        "📋 Data Explorer"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Data")

if not df_raw.empty:
    all_quads = sorted(df_raw['kuadran_perilaku'].dropna().unique().tolist())
    selected_quads = st.sidebar.multiselect(
        "Kuadran Perilaku:",
        options=all_quads,
        default=all_quads
    )
    
    score_range = st.sidebar.slider(
        "Rating Bintang (Score):",
        min_value=1,
        max_value=5,
        value=(1, 5)
    )
    
    suspicion_min = st.sidebar.slider(
        "Minimum Suspicion Score:",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=5.0
    )
    
    # Filter dataset
    df_filtered = df_raw[
        (df_raw['kuadran_perilaku'].isin(selected_quads)) &
        (df_raw['score'] >= score_range[0]) &
        (df_raw['score'] <= score_range[1]) &
        (df_raw['suspicion_score'] >= suspicion_min)
    ]
else:
    df_filtered = pd.DataFrame()

st.sidebar.markdown("---")
st.sidebar.info(
    "**Riset Kolaborasi UNNES**\n"
    "Divisi Penambangan Data (Data Mining)\n"
    "Kompetisi Nasional DIMAS-TI 2026."
)

# ── Header Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div class="main-title">Deteksi Review Bombing & Anomali Multi-View KAI Access</div>
  <div class="sub-title">
    Kerangka kerja data mining hibrida: Integrasi Proses Stokastik Goh-Barabási, Leksikon Fungsional Kereta Api, dan Graf Modularitas Louvain pada 15.000+ Ulasan Google Play Store.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Module 1: Executive Overview & KPI ────────────────────────────────────────
if page == "📊 Executive Overview & KPI":
    st.subheader("Ringkasan Metrik & KPI Anomali")
    
    total_data = len(df_raw)
    current_data = len(df_filtered)
    bombing_count = (df_filtered['kuadran_perilaku'] == '4. Review Bombing Terkoordinasi').sum()
    avg_suspicion = df_filtered['suspicion_score'].mean() if not df_filtered.empty else 0.0
    avg_burst = df_filtered['burstiness_param'].mean() if 'burstiness_param' in df_filtered else 0.0
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-val">{current_data:,} <span style="font-size:14px;color:#64748b;">/ {total_data:,}</span></div>
          <div class="kpi-lbl">Total Ulasan Terpilih</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(239, 68, 68, 0.4);">
          <div class="kpi-val" style="color: #f87171;">{bombing_count:,}</div>
          <div class="kpi-lbl">Review Bombing Terdeteksi</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-val" style="color: #38bdf8;">{avg_suspicion:.2f}</div>
          <div class="kpi-lbl">Rata-rata Suspicion Score</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-val" style="color: #fbbf24;">{avg_burst:.3f}</div>
          <div class="kpi-lbl">Mean Burstiness (B)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Visualisasi Utama
    chart_col1, chart_col2 = st.columns([1, 1])
    
    with chart_col1:
        st.markdown("#### Proporsi 4 Kuadran Perilaku")
        if not df_filtered.empty:
            quad_counts = df_filtered['kuadran_perilaku'].value_counts().reset_index()
            quad_counts.columns = ['Kuadran', 'Jumlah']
            
            color_map = {
                '1. Ulasan Organik Normal': '#10b981',
                '2. Krisis Layanan Tiket Organik': '#3b82f6',
                '3. Keluhan Umum / Spam Individu': '#f59e0b',
                '4. Review Bombing Terkoordinasi': '#ef4444'
            }
            
            fig_pie = px.pie(
                quad_counts,
                names='Kuadran',
                values='Jumlah',
                color='Kuadran',
                color_discrete_map=color_map,
                hole=0.45
            )
            fig_pie.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1'),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Data kosong untuk filter yang dipilih.")
            
    with chart_col2:
        st.markdown("#### Distribusi Rating Bintang Ulasan")
        if not df_filtered.empty:
            score_counts = df_filtered['score'].value_counts().sort_index().reset_index()
            score_counts.columns = ['Rating', 'Jumlah']
            fig_bar = px.bar(
                score_counts,
                x='Rating',
                y='Jumlah',
                color='Rating',
                color_continuous_scale='Bluered',
                text='Jumlah'
            )
            fig_bar.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1'),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
    st.markdown("---")
    
    # Timeline Temporal
    st.markdown("#### Dinamika Temporal Ulasan & Lonjakan Anomali")
    if not df_filtered.empty and 'date' in df_filtered.columns:
        daily_trends = df_filtered.groupby(['date', 'kuadran_perilaku']).size().reset_index(name='Volume')
        fig_time = px.line(
            daily_trends,
            x='date',
            y='Volume',
            color='kuadran_perilaku',
            title="Tren Frekuensi Ulasan Harian Berdasarkan Kuadran",
            color_discrete_map={
                '1. Ulasan Organik Normal': '#10b981',
                '2. Krisis Layanan Tiket Organik': '#3b82f6',
                '3. Keluhan Umum / Spam Individu': '#f59e0b',
                '4. Review Bombing Terkoordinasi': '#ef4444'
            }
        )
        fig_time.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)')
        )
        st.plotly_chart(fig_time, use_container_width=True)

# ── Module 2: Multi-View Feature Explorer ──────────────────────────────────────
elif page == "🔬 Multi-View Feature Explorer":
    st.subheader("Arsitektur Tripartit Multi-View")
    st.markdown("""
    Pendekatan ini memisahkan ekstraksi representasi fitur ke dalam tiga modalitas ortogonal untuk memastikan tidak ada ketergantungan sepihak pada rating numerik atau analisis sentimen naif semata.
    """)
    
    subtab1, subtab2, subtab3 = st.tabs([
        "View 1: Temporal & Goh-Barabási", 
        "View 2: Leksikon Fungsional KAI", 
        "View 3: Co-Review Network & Louvain"
    ])
    
    with subtab1:
        st.markdown("### 1. View 1: Dimensi Temporal & Proses Stokastik")
        st.latex(r"Z_t = \max\left(0, \frac{X_t - \mu_{24}}{\sigma_{24}}\right)")
        st.latex(r"B = \frac{\sigma_\tau - \mu_\tau}{\sigma_\tau + \mu_\tau} \in [-1, 1]")
        st.write("""
        * **Moving Z-Score ($Z_t$):** Mengukur lonjakan volume ulasan negatif dalam jendela 24 jam. Nilai $Z_t > 3$ mengindikasikan lonjakan anomali di luar ambang deviasi standar historis.
        * **Parameter Burstiness ($B$):** Mengevaluasi variabilitas interval kedatangan ulasan ($\tau = t_i - t_{i-1}$). Nilai $B \to 1$ merepresentasikan ledakan ulasan serentak (ciri khas serangan bot/review bombing terkoordinasi), sedangkan $B \approx 0$ mencerminkan proses Poisson wajar.
        """)
        
        if 'burstiness_param' in df_filtered.columns:
            fig_burst = px.histogram(
                df_filtered,
                x='burstiness_param',
                color='kuadran_perilaku',
                nbins=30,
                title="Distribusi Parameter Burstiness (B) per Kuadran"
            )
            fig_burst.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1'))
            st.plotly_chart(fig_burst, use_container_width=True)

    with subtab2:
        st.markdown("### 2. View 2: Spesifisitas Leksikon Transportasi Kereta Api")
        st.write("""
        Model membedakan antara komplain operasional nyata dengan makian/ujaran generik. Ulasan yang mengandung kosakata teknis operasional (tiket, booking, refund, reschedule, server down, kaipay) diberi skor **Specificity** lebih tinggi, sehingga tidak otomatis disalahklasifikasikan sebagai bot palsu meskipun bernada marah.
        """)
        
        if 'specificity_score' in df_filtered.columns:
            fig_box = px.box(
                df_filtered,
                x='kuadran_perilaku',
                y='specificity_score',
                color='kuadran_perilaku',
                title="Tingkat Spesifisitas Leksikon per Kuadran"
            )
            fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1'))
            st.plotly_chart(fig_box, use_container_width=True)

    with subtab3:
        st.markdown("### 3. View 3: Co-Review Network & Deteksi Komunitas Louvain")
        st.write(r"""
        Graf jaringan kesamaan dibangun dengan menghubungkan dua ulasan jika muncul dalam rentang waktu $\le 30$ menit dan memiliki kemiripan semantik TF-IDF cosine similarity $\ge 0.80$.
        Algoritma partisi modularitas Louvain kemudian mengidentifikasi sindikat komunitas terorganisir.
        """)
        
        if 'comm_size' in df_filtered.columns:
            fig_comm = px.scatter(
                df_filtered,
                x='comm_size',
                y='suspicion_score',
                color='kuadran_perilaku',
                size='score',
                hover_data=['cleaned_content'],
                title="Ukuran Komunitas Graf (comm_size) vs Suspicion Score"
            )
            fig_comm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1'))
            st.plotly_chart(fig_comm, use_container_width=True)

# ── Module 3: Live Review Simulator ───────────────────────────────────────────
elif page == "🧪 Live Review Simulator":
    st.subheader("Simulasi & Uji Deteksi Ulasan Real-Time")
    st.markdown("""
    Ketikkan teks ulasan baru untuk menguji bagaimana pipeline multi-view memproses teks, menghitung spesifisitas teknis KAI, mengestimasi tingkat kecurigaan (*Suspicion Score*), dan mengalokasikannya ke kuadran keputusan.
    """)
    
    col_input, col_preset = st.columns([2, 1])
    
    with col_preset:
        st.markdown("##### Contoh Template Ulasan:")
        template_choice = st.selectbox(
            "Pilih contoh uji:",
            [
                "-- Kustom Sendiri --",
                "Bot/Review Bombing: 'aplikasi sampah jelek banget bintang 1 hapus aja'",
                "Komplain Riil KAI: 'tiket kereta gagal bayar via kaipay padahal saldo terpotong'",
                "Spam Generik: 'jelek lambat parah ga guna'",
                "Organik Positif: 'sangat mudah dan cepat pesan tiket mudik lebaran terima kasih kai'"
            ]
        )
        
    initial_text = ""
    initial_score = 1
    if template_choice.startswith("Bot/Review"):
        initial_text = "aplikasi sampah jelek banget bintang 1 hapus aja"
        initial_score = 1
    elif template_choice.startswith("Komplain Riil"):
        initial_text = "tiket kereta gagal bayar via kaipay padahal saldo terpotong server down waktu war tiket"
        initial_score = 1
    elif template_choice.startswith("Spam Generik"):
        initial_text = "jelek lambat parah ga guna"
        initial_score = 1
    elif template_choice.startswith("Organik Positif"):
        initial_text = "sangat mudah dan cepat pesan tiket mudik lebaran terima kasih kai"
        initial_score = 5

    with col_input:
        user_text = st.text_area("Teks Ulasan:", value=initial_text, height=110, placeholder="Masukkan ulasan ulasan aplikasi Access by KAI...")
        user_score = st.slider("Rating Bintang (1 - 5):", 1, 5, value=initial_score)
        
    btn_predict = st.button("🚀 Jalankan Diagnosis Deteksi Anomali", type="primary")
    
    if btn_predict:
        if not user_text.strip():
            st.error("Silakan masukkan teks ulasan terlebih dahulu.")
        else:
            # Pipeline cleaner & feature extraction
            if model_bundle is not None:
                slang = model_bundle['lexicon']['KAMUS_SLANG']
                keywords = model_bundle['lexicon']['KAI_KEYWORDS']
            else:
                slang = {"gak": "tidak", "ga": "tidak", "bgt": "banget", "lemot": "lambat"}
                keywords = ['tiket', 'booking', 'kaipay', 'refund', 'reschedule', 'server down', 'gagal bayar']
                
            # Normalisasi
            cleaned = str(user_text).lower()
            cleaned = re.sub(r'https?://\S+|www\.\S+', ' ', cleaned)
            cleaned = re.sub(r'[^a-zA-Z0-9\s-]', ' ', cleaned)
            cleaned = ' '.join([slang.get(w, w) for w in cleaned.split() if w])
            
            # Specificity
            spans = set()
            for kw in sorted(keywords, key=len, reverse=True):
                for m in re.finditer(r'\b' + re.escape(kw) + r'\b', cleaned):
                    s, e = m.span()
                    if not any(s < end and e > start for start, end in spans):
                        spans.add((s, e))
            kw_chars = sum(e - s for s, e in spans)
            spec_score = min(1.0, kw_chars / max(1, len(cleaned)))
            
            # Estimasi skor kecurigaan heuristik multi-view
            base_score = 0.0
            if user_score <= 2:
                base_score += 45.0
                if spec_score < 0.05:
                    base_score += 35.0 # Komplain tidak spesifik / makian
                else:
                    base_score -= 20.0 # Komplain spesifik teknis
            else:
                base_score = max(5.0, 30.0 - user_score * 5.0)
                
            suspicion_res = max(0.0, min(100.0, base_score))
            
            # Tentukan Kuadran
            if user_score <= 2 and spec_score < 0.04 and suspicion_res >= 70:
                kuadran_res = "4. Review Bombing Terkoordinasi"
                badge_color = "#ef4444"
                penjelasan = "Terdeteksi sebagai potensi Review Bombing atau sindikat spam: Rating rendah dengan konten generik/tanpa konteks teknis KAI."
            elif user_score <= 2 and spec_score >= 0.04:
                kuadran_res = "2. Krisis Layanan Tiket Organik"
                badge_color = "#3b82f6"
                penjelasan = "Keluhan fungsional nyata dari pengguna sah: Pengguna mengalami masalah tiket/server riil yang memiliki spesifisitas tinggi."
            elif user_score <= 2:
                kuadran_res = "3. Keluhan Umum / Spam Individu"
                badge_color = "#f59e0b"
                penjelasan = "Keluhan umum sporadis tanpa tanda-tanda koordinasi massal berskala besar."
            else:
                kuadran_res = "1. Ulasan Organik Normal"
                badge_color = "#10b981"
                penjelasan = "Ulasan kepuasan wajar dari pelanggan organik aplikasi Access by KAI."
                
            st.markdown("---")
            st.markdown("### Hasil Diagnosis Pipeline:")
            
            res_col1, res_col2, res_col3 = st.columns([1, 1, 1])
            with res_col1:
                st.metric("Spesifisitas KAI Leksikon", f"{spec_score * 100:.1f}%")
            with res_col2:
                st.metric("Estimasi Suspicion Score", f"{suspicion_res:.1f} / 100")
            with res_col3:
                st.markdown(f"""
                <div style="padding:10px; border-radius:8px; background:rgba(255,255,255,0.05); text-align:center;">
                  <div style="font-size:12px; color:#94a3b8;">Klasifikasi Kuadran</div>
                  <div style="font-size:16px; font-weight:700; color:{badge_color}; margin-top:4px;">{kuadran_res}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.info(f"💡 **Analisis:** {penjelasan}")
            with st.expander("Teks Setelah Normalisasi Slang & Regex:"):
                st.code(cleaned)

# ── Module 4: Riset & Makalah Ilmiah IEEE ──────────────────────────────────────
elif page == "📑 Riset & Makalah Ilmiah IEEE":
    st.subheader("Dokumentasi Makalah Ilmiah & Benchmark Model")
    st.markdown("""
    Riset ini disusun untuk kompetisi **DIMAS-TI 2026** (Divisi Penambangan Data) oleh Tim Peneliti FMIPA UNNES.
    """)
    
    st.markdown("#### Visualisasi Hasil Komprehensif Makalah (Multi-Panel)")
    final_img_path = ASSETS_DIR / "visualisasi_final_kai.png"
    if final_img_path.exists():
        st.image(str(final_img_path), caption="Hasil Eksperimen Lengkap: Dinamika Temporal, Graf Louvain, dan Proyeksi Manifold", use_container_width=True)
    else:
        st.warning("Aset visualisasi final tidak ditemukan.")
        
    st.markdown("---")
    st.markdown("#### Benchmark Kinerja: Multi-View Framework vs Baseline AutoML (FLAML)")
    bench_img_path = ASSETS_DIR / "benchmark_flaml_vs_multiview.png"
    if bench_img_path.exists():
        st.image(str(bench_img_path), caption="Perbandingan ROC-AUC dan F1-Score Multi-View vs Model Tradisional", use_container_width=True)

# ── Module 5: Data Explorer ───────────────────────────────────────────────────
elif page == "📋 Data Explorer":
    st.subheader("Eksplorasi Data Ulasan Terklasifikasi")
    st.markdown("Telusuri data hasil inferensi deteksi anomali multi-view secara detail.")
    
    if not df_filtered.empty:
        search_query = st.text_input("Cari ulasan berdasarkan kata kunci:", placeholder="Ketik kata yang dicari...")
        df_show = df_filtered.copy()
        if search_query:
            df_show = df_show[df_show['cleaned_content'].str.contains(search_query, case=False, na=False)]
            
        display_cols = [c for c in ['userName', 'score', 'cleaned_content', 'suspicion_score', 'kuadran_perilaku', 'specificity_score', 'at'] if c in df_show.columns]
        st.dataframe(df_show[display_cols], use_container_width=True, height=450)
        
        # Download button
        csv_bytes = df_show.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Data Terfilter (CSV)",
            data=csv_bytes,
            file_name="kai_reviewbombing_filtered.csv",
            mime="text/csv"
        )
    else:
        st.warning("Tidak ada data ulasan yang memenuhi filter saat ini.")
