# kai_detector_pipeline.py
"""
Shared Pipeline Module for Access by KAI Multi-View Anomaly & Review Bombing Detection
Data Mining & Machine Learning Division
FMIPA Universitas Negeri Semarang (UNNES) & Universitas Negeri Gorontalo
"""

import os
import re
import json
import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass, field
from networkx.algorithms.community import louvain_communities, modularity as nx_modularity
from sklearn.feature_extraction.text import TfidfVectorizer

# [FIX-11] Default dictionary leksikon & slang
DEFAULT_KAMUS_SLANG = {
    "gak": "tidak", "ga": "tidak", "ngga": "tidak", "nggak": "tidak", "tdk": "tidak",
    "bgt": "banget", "bngt": "banget", "apk": "aplikasi", "app": "aplikasi",
    "lemot": "lambat", "lelet": "lambat", "bgus": "bagus", "trs": "terus",
    "dpt": "dapat", "pdhl": "padahal", "sy": "saya", "klo": "kalau", "kalo": "kalau",
    "tp": "tapi", "udh": "sudah", "udah": "sudah", "blm": "belum", "eror": "error",
    "refund": "pengembalian dana", "reschedule": "ubah jadwal", "restitusi": "restitusi",
    "boarding": "boarding pass", "tiketnya": "tiket", "kursinya": "kursi",
    "nyesel": "menyesal", "kecewaa": "kecewa", "bintang1": "bintang satu", "bintang5": "bintang lima"
}

DEFAULT_KAI_KEYWORDS = [
    'tiket', 'booking', 'kode booking', 'kursi', 'gerbong', 'nik ktp', 'nik penumpang', 'penumpang',
    'kaipay', 'kai pay', 'qris', 'transfer', 'saldo', 'restitusi', 'refund', 'pembatalan',
    'ubah jadwal', 'reschedule', 'stasiun', 'kereta', 'eksekutif', 'ekonomi', 'lokomotif',
    'antrean', 'war tiket', 'server down', 'server sibuk', 'gagal bayar', 'loading terus',
    'force close', 'keluar sendiri', 'boarding pass', 'e-boarding',
    'pengembalian dana', 'dana kembali', 'uang kembali'
]

def clean_text_fast(text, slang_dict=None):
    """Membersihkan teks ulasan, menghapus URL/karakter non-alfanumerik, dan normalisasi slang."""
    slang = slang_dict if slang_dict is not None else DEFAULT_KAMUS_SLANG
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text)
    words = text.split()
    normalized = [slang.get(w, w) for w in words if len(w) > 0]
    return ' '.join(normalized)

def calc_specificity(text, keywords_list=None):
    """Menghitung skor spesifisitas teknis KAI dengan word-boundary regex dan non-overlapping span."""
    keywords = keywords_list if keywords_list is not None else DEFAULT_KAI_KEYWORDS
    t_lower = str(text).lower()
    words = t_lower.split()
    if not words:
        return 0.0
    matched_spans = set()
    for kw in sorted(keywords, key=len, reverse=True):
        pattern = r'\b' + re.escape(kw) + r'\b'
        for m in re.finditer(pattern, t_lower):
            start, end = m.span()
            if not any(s < end and e > start for s, e in matched_spans):
                matched_spans.add((start, end))
    return len(matched_spans) / (len(words) + 2)

@dataclass
class FrozenModelState:
    tfidf: TfidfVectorizer
    scalers: dict          # dict of fitted MinMaxScaler per feature
    rolling_stats: dict    # precomputed hourly statistics from calibration
    lexicon: dict          # KAI_KEYWORDS and KAMUS_SLANG
    graph_cfg: dict        # time_window_sec, BEST_TAU, louvain_seed
    fusion_cfg: dict       # weights and hierarchical gates
    threshold_cfg: dict    # suspicion_threshold, comm_size_gate, community_purity_min
    dataset_hash: str
    _evaluated: bool = False

def extract_temporal_frozen(data, state_rolling=None):
    """Ekstraksi fitur dinamika temporal menggunakan baseline rolling statistics terbekukan."""
    d = data.copy().sort_values('at').reset_index(drop=True)
    cols_to_drop = ['hour_bucket', 'temporal_z_score', 'inter_arrival_sec', 'burstiness_param']
    d = d.drop(columns=[c for c in cols_to_drop if c in d.columns])
    d['hour_bucket'] = pd.to_datetime(d['at']).dt.floor('h')

    neg_reviews = d[d['score'] <= 2]
    hourly_neg = neg_reviews.groupby('hour_bucket').size().rename('neg_count')
    full_hours = pd.date_range(start=d['hour_bucket'].min(), end=d['hour_bucket'].max(), freq='h')
    hourly_neg = hourly_neg.reindex(full_hours, fill_value=0)
    hourly_neg.index.name = 'hour_bucket'

    roll_mean = hourly_neg.rolling(24, min_periods=1).mean()
    roll_std = hourly_neg.rolling(24, min_periods=1).std().replace(0, 1.0).fillna(1.0)
    z_series = ((hourly_neg - roll_mean) / roll_std).rename('temporal_z_score')

    d = d.merge(z_series.reset_index(), on='hour_bucket', how='left')
    d['temporal_z_score'] = d['temporal_z_score'].fillna(0.0).clip(lower=0.0)

    inter = pd.to_datetime(d['at']).diff().dt.total_seconds()
    med_inter = state_rolling.get('median_inter_arrival', 1800.0) if state_rolling else inter.median()
    d['inter_arrival_sec'] = inter.fillna(med_inter).clip(lower=0.1)

    def calc_b(s):
        mu, sigma = s.mean(), s.std()
        return 0.0 if (pd.isna(sigma) or (sigma + mu) == 0) else float((sigma - mu) / (sigma + mu))

    d['burstiness_param'] = (
        d['inter_arrival_sec'].rolling(15, min_periods=3).apply(calc_b, raw=False).fillna(0.0)
    )
    return d

def build_network_state(data, emb_sparse, tau_thresh, time_window_sec=1800, louvain_seed=42):
    """
    [FIX-11] Pembangunan Co-Review Graph dan Deteksi Komunitas Louvain dengan seed konsisten.
    """
    G = nx.Graph()
    n = len(data)
    for i in range(n): G.add_node(i)
    ts = pd.to_datetime(data['at']).values.astype('int64') // 10**9
    max_sim = np.zeros(n, dtype=np.float32)
    edges = []

    for i in range(n):
        j = i + 1
        neighbors = []
        while j < n and (ts[j] - ts[i]) <= time_window_sec:
            neighbors.append(j)
            j += 1
        if neighbors:
            sims = (emb_sparse[i] @ emb_sparse[neighbors].T).toarray()[0]
            max_sim[i] = max(max_sim[i], float(np.max(sims)))
            for k_idx, s in zip(neighbors, sims):
                max_sim[k_idx] = max(max_sim[k_idx], float(s))
                if s >= tau_thresh:
                    edges.append((i, k_idx, float(s)))

    G.add_weighted_edges_from(edges)
    deg_cent = nx.degree_centrality(G)
    clust_map = nx.clustering(G, weight='weight')

    if G.number_of_edges() > 0:
        # [FIX-11] Gunakan louvain_seed dari parameter/config, default 42
        comms = louvain_communities(G, weight='weight', seed=louvain_seed)
        part = {node: idx for idx, c in enumerate(comms) for node in c}
        mod_q = nx_modularity(G, comms, weight='weight')
    else:
        part = {node: node for node in range(n)}
        mod_q = 0.0

    c_counts = pd.Series(part).value_counts().to_dict()
    df_res = data.copy()
    df_res['max_temporal_sim'] = max_sim
    df_res['degree_centrality'] = [deg_cent[i] for i in range(n)]
    df_res['clustering_coeff'] = [clust_map[i] for i in range(n)]
    df_res['louvain_comm_id'] = [part[i] for i in range(n)]
    df_res['comm_size'] = df_res['louvain_comm_id'].map(c_counts).fillna(1).astype(int)

    is_neg_sim = (df_res['score'] <= 2) & (df_res['max_temporal_sim'] >= tau_thresh)
    comm_neg_counts = df_res.groupby('louvain_comm_id').apply(lambda g: (is_neg_sim.iloc[g.index]).mean()).to_dict()
    df_res['community_purity'] = df_res['louvain_comm_id'].map(comm_neg_counts).fillna(0.0).astype(float)

    return df_res, G, mod_q

def apply_frozen(state: FrozenModelState, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    [FIX-12] Transform-Only Inference Pipeline.
    Catatan: suspicion_score = secondary_score * 100, BUKAN probabilitas terkalibrasi.
    secondary_score = f_comm * (1 + 0.5*f_sim + 0.3*f_z) * (1 - 0.5*f_spec).
    Rentang teoritis: [0, 1.8] sehingga suspicion_score in [0, 180]. Digunakan untuk pemeringkatan.
    """
    # 1. Ekstraksi Fitur Waktu dengan Stats Terbekukan
    data_enriched = extract_temporal_frozen(dataframe, state.rolling_stats)
    
    # 2. Vektorisasi Teks MENGGUNAKAN TFIDF TERBEKUKAN (Transform Only!)
    X_emb_trans = state.tfidf.transform(data_enriched['cleaned_content'])
    
    # 3. Pembangunan Graf dengan BEST_TAU & louvain_seed Terbekukan [FIX-11]
    l_seed = state.graph_cfg.get('louvain_seed', 42)
    data_net, G_sub, Q_sub = build_network_state(
        data_enriched, X_emb_trans, state.graph_cfg['BEST_TAU'],
        time_window_sec=state.graph_cfg.get('time_window_sec', 1800),
        louvain_seed=l_seed
    )
    
    # 4. Normalisasi Fitur MENGGUNAKAN SCALER TERBEKUKAN (DILARANG FIT!)
    f_sim  = state.scalers['f_sim'].transform(data_net[['max_temporal_sim']]).clip(0, 1)
    f_comm = state.scalers['f_comm'].transform(data_net[['comm_size']]).clip(0, 1)
    f_z    = state.scalers['f_z'].transform(data_net[['temporal_z_score']]).clip(0, 1)
    f_clust= state.scalers['f_clust'].transform(data_net[['clustering_coeff']]).clip(0, 1)
    f_spec = state.scalers['f_spec'].transform(data_net[['specificity_score']]).clip(0, 1)
    f_score= state.scalers['f_score'].transform(data_net[['score']]).clip(0, 1)
    
    # 5. Formula Fusi Sekunder Hierarkis
    sec = f_comm * (1.0 + 0.5*f_sim + 0.3*f_z) * (1.0 - 0.5*f_spec)
    data_net['secondary_score'] = sec.flatten()
    data_net['suspicion_score'] = (data_net['secondary_score'] * 100).round(2)
    
    # 6. Alokasi Kuadran Keputusan [FIX-07: gunakan specificity_q2_min dari config]
    gate_c = state.threshold_cfg['comm_size_gate']
    purity_min = state.threshold_cfg['community_purity_min']
    tau_h = state.threshold_cfg['tau_high']
    sec_thresh = state.threshold_cfg['suspicion_threshold']
    spec_q2_min = state.threshold_cfg.get('specificity_q2_min', 0.04)
    
    def assign_quad(r):
        primary = (r['comm_size'] >= gate_c) and (r['score'] <= 2) and ((r['community_purity'] >= purity_min) or (r['max_temporal_sim'] >= tau_h))
        secondary = (r['secondary_score'] >= sec_thresh)
        if primary and secondary:
            return '4. Review Bombing Terkoordinasi'
        elif (r['score'] <= 2) and (r['specificity_score'] >= spec_q2_min):
            return '2. Krisis Layanan Tiket Organik'
        elif r['score'] <= 2:
            return '1. Keluhan Sporadis Organik'
        else:
            return '3. Kepuasan Layanan Organik'
            
    data_net['kuadran_perilaku'] = data_net.apply(assign_quad, axis=1)
    return data_net
