import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Config
st.set_page_config(
    page_title="COVID-19 Predictor Indonesia",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.8rem !important;
        font-weight: bold !important;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.8rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .stButton > button {
        width: 100%;
        height: 3.5rem;
        border-radius: 15px;
        font-size: 1.2rem;
        font-weight: bold;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        box-shadow: 0 8px 25px rgba(255,107,107,0.4);
    }
    .future-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 15px 25px;
        border-radius: 30px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load models dengan error handling"""
    try:
        scaler = joblib.load('models/scaler.pkl')
        kmeans_model = joblib.load('models/kmeans_model.pkl')
        rf_model = joblib.load('models/random_forest_model.pkl')
        st.success("✅ Models loaded successfully!")
        return scaler, kmeans_model, rf_model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.error("📁 Pastikan folder `models/` ada dengan 3 file .pkl")
        st.stop()

# Load models
scaler, kmeans_model, rf_model = load_models()

# PROVINSI dengan data realistis (populasi, urban, risk)
PROVINSI_DATA = {
    11: {'nama': 'ACEH', 'populasi': 5500000, 'urban': 0.30, 'risk_factor': 0.80, 'color': '#DDA0DD'},
    12: {'nama': 'SUMATERA UTARA', 'populasi': 15000000, 'urban': 0.50, 'risk_factor': 1.00, 'color': '#98D8C8'},
    13: {'nama': 'SUMATERA BARAT', 'populasi': 5600000, 'urban': 0.45, 'risk_factor': 0.90, 'color': '#F7DC6F'},
    14: {'nama': 'RIAU', 'populasi': 6500000, 'urban': 0.60, 'risk_factor': 1.00, 'color': '#BB8FCE'},
    15: {'nama': 'JAMBI', 'populasi': 3700000, 'urban': 0.40, 'risk_factor': 0.85, 'color': '#85C1E9'},
    16: {'nama': 'SUMATERA SELATAN', 'populasi': 8900000, 'urban': 0.45, 'risk_factor': 0.95, 'color': '#A9DFBF'},
    17: {'nama': 'BENGKULU', 'populasi': 2100000, 'urban': 0.35, 'risk_factor': 0.80, 'color': '#AED6F1'},
    18: {'nama': 'LAMPUNG', 'populasi': 9200000, 'urban': 0.40, 'risk_factor': 0.90, 'color': '#F9E79F'},
    19: {'nama': 'KEP. BANGKA BELITUNG', 'populasi': 1500000, 'urban': 0.55, 'risk_factor': 0.85, 'color': '#D7BDE2'},
    21: {'nama': 'KEP. RIAU', 'populasi': 2200000, 'urban': 0.75, 'risk_factor': 1.00, 'color': '#F5CBA7'},

    31: {'nama': 'DKI JAKARTA', 'populasi': 10700000, 'urban': 1.00, 'risk_factor': 1.50, 'color': '#FF6B6B'},
    32: {'nama': 'JAWA BARAT', 'populasi': 50000000, 'urban': 0.60, 'risk_factor': 1.30, 'color': '#4ECDC4'},
    33: {'nama': 'JAWA TENGAH', 'populasi': 37000000, 'urban': 0.50, 'risk_factor': 1.10, 'color': '#45B7D1'},
    34: {'nama': 'DI YOGYAKARTA', 'populasi': 3800000, 'urban': 0.90, 'risk_factor': 1.10, 'color': '#FFEAA7'},
    35: {'nama': 'JAWA TIMUR', 'populasi': 41000000, 'urban': 0.55, 'risk_factor': 1.20, 'color': '#96CEB4'},
    36: {'nama': 'BANTEN', 'populasi': 12500000, 'urban': 0.80, 'risk_factor': 1.40, 'color': '#F8C471'},

    51: {'nama': 'BALI', 'populasi': 4500000, 'urban': 0.70, 'risk_factor': 1.10, 'color': '#F1948A'},
    52: {'nama': 'NUSA TENGGARA BARAT', 'populasi': 5600000, 'urban': 0.35, 'risk_factor': 0.85, 'color': '#D5DBDB'},
    53: {'nama': 'NUSA TENGGARA TIMUR', 'populasi': 5600000, 'urban': 0.25, 'risk_factor': 0.80, 'color': '#A3E4D7'},

    61: {'nama': 'KALIMANTAN BARAT', 'populasi': 5600000, 'urban': 0.40, 'risk_factor': 0.85, 'color': '#FAD7A0'},
    62: {'nama': 'KALIMANTAN TENGAH', 'populasi': 2800000, 'urban': 0.35, 'risk_factor': 0.80, 'color': '#F5B7B1'},
    63: {'nama': 'KALIMANTAN SELATAN', 'populasi': 4300000, 'urban': 0.50, 'risk_factor': 0.90, 'color': '#D2B4DE'},
    64: {'nama': 'KALIMANTAN TIMUR', 'populasi': 4100000, 'urban': 0.65, 'risk_factor': 1.00, 'color': '#A2D9CE'},
    65: {'nama': 'KALIMANTAN UTARA', 'populasi': 750000, 'urban': 0.55, 'risk_factor': 0.85, 'color': '#AED6F1'},

    71: {'nama': 'SULAWESI UTARA', 'populasi': 2700000, 'urban': 0.55, 'risk_factor': 0.90, 'color': '#F9E79F'},
    72: {'nama': 'SULAWESI TENGAH', 'populasi': 3100000, 'urban': 0.35, 'risk_factor': 0.80, 'color': '#D7BDE2'},
    73: {'nama': 'SULAWESI SELATAN', 'populasi': 9400000, 'urban': 0.50, 'risk_factor': 1.00, 'color': '#F5CBA7'},
    74: {'nama': 'SULAWESI TENGGARA', 'populasi': 2800000, 'urban': 0.35, 'risk_factor': 0.85, 'color': '#A9DFBF'},
    75: {'nama': 'GORONTALO', 'populasi': 1200000, 'urban': 0.40, 'risk_factor': 0.80, 'color': '#AED6F1'},
    76: {'nama': 'SULAWESI BARAT', 'populasi': 1500000, 'urban': 0.30, 'risk_factor': 0.75, 'color': '#FADBD8'},

    81: {'nama': 'MALUKU', 'populasi': 1900000, 'urban': 0.35, 'risk_factor': 0.80, 'color': '#D6EAF8'},
    82: {'nama': 'MALUKU UTARA', 'populasi': 1300000, 'urban': 0.35, 'risk_factor': 0.75, 'color': '#D5F5E3'},

    91: {'nama': 'PAPUA', 'populasi': 4400000, 'urban': 0.30, 'risk_factor': 0.80, 'color': '#FCF3CF'},
    92: {'nama': 'PAPUA BARAT', 'populasi': 600000, 'urban': 0.45, 'risk_factor': 0.80, 'color': '#E8DAEF'},
    93: {'nama': 'PAPUA SELATAN', 'populasi': 550000, 'urban': 0.25, 'risk_factor': 0.75, 'color': '#D4E6F1'},
    94: {'nama': 'PAPUA TENGAH', 'populasi': 1400000, 'urban': 0.20, 'risk_factor': 0.75, 'color': '#D5F5E3'},
    95: {'nama': 'PAPUA PEGUNUNGAN', 'populasi': 1500000, 'urban': 0.15, 'risk_factor': 0.70, 'color': '#FCF3CF'},
    96: {'nama': 'PAPUA BARAT DAYA', 'populasi': 650000, 'urban': 0.45, 'risk_factor': 0.80, 'color': '#FADBD8'}
}

# Mapping 8 Cluster Berdasarkan Centroid Data Riil ML Kamu
CLUSTER_INTERPRETASI = {
    0: "🟢 **OPTIMAL** - Vaksinasi Tinggi & Angka Kematian Sangat Rendah",
    1: "🟢 **AMAN** - Efektivitas Cakupan Vaksin Baik & Risiko Terkendali",
    2: "🔴 **KRITIS** - Lonjakan Kematian Tinggi! Perlu Evaluasi Faskes",
    3: "🟢 **SANGAT OPTIMAL** - Proteksi Kelompok Wilayah Maksimal",
    4: "🟠 **TINGGI** - Risiko Kematian Moderat, Dorong Program Booster",
    5: "🟡 **WASPADA** - Vaksinasi Rendah, Potensi Bahaya Gelombang Baru",
    6: "🟢 **STABIL** - Imunitas Wilayah dan Angka Kasus Terjaga Baik",
    7: "🟡 **MODERAT** - Kesiapsiagaan Vaksinasi Perlu Dipertahankan"
}

# === HEADER ===
st.markdown('<h1 class="main-header">🦠 COVID-19 Future Predictor Indonesia</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#666; font-size:1.2rem;'>
    🔮 Prediksi vaksinasi & kematian <strong>2025-2030</strong> per provinsi<br>
    🤖 K-Means + Random Forest | 📊 Akurasi K-Means <strong>98.78%</strong>
</div>
""", unsafe_allow_html=True)

# === SIDEBAR ===
st.sidebar.header("🎯 **Parameter Prediksi**")
st.sidebar.markdown("---")

# Provinsi
prov_kode = st.sidebar.selectbox(
    "🏛️ **Provinsi**", 
    list(PROVINSI_DATA.keys()), 
    format_func=lambda x: f"**{x}** - {PROVINSI_DATA[x]['nama']}",
    index=0
)

# Tahun masa depan
tahun = st.sidebar.selectbox(
    "📅 **Tahun Prediksi**", 
    [2025, 2026, 2027, 2028, 2029, 2030],
    index=0
)

minggu = st.sidebar.slider("📋 **Minggu ke-**", 1, 52, 26)

# Trend adjustment
st.sidebar.markdown("---")
trend_boost = st.sidebar.slider(
    "📈 **Trend Vaksinasi** (%/tahun)", 
    5, 30, 15, 5,
    help="Kenaikan vaksinasi per tahun"
)

st.sidebar.markdown("---")
st.sidebar.markdown("*Populasi, urbanisasi, & risk factor otomatis diperhitungkan*")

# === PREDICTION BUTTON ===
if st.sidebar.button("🚀 **JALANKAN PREDIKSI**", use_container_width=True):
    prov_info = PROVINSI_DATA[prov_kode]
    
    with st.spinner("🔮 Menghitung prediksi masa depan..."):
        # 1. ESTIMASI DYNAMIC INPUT (Menggantikan hardcoded 0,0,0,0 agar RF bergerak dinamis)
        years_ahead = tahun - 2024
        pop_scale = prov_info['populasi'] / 10_000_000
        
        # Penyesuaian nilai mentah skala basis data klaster asli (vaksin: ~40-105, kematian: ~0.005-1.73)
        approx_vaksin = pop_scale * 65.0 * (1 + (trend_boost / 100) * years_ahead)
        approx_death = pop_scale * prov_info['risk_factor'] * 0.4 * max(0.1, 1 - 0.12 * years_ahead)
        
        approx_pct = (approx_death / max(0.001, approx_vaksin)) * 100
        approx_rasio = approx_vaksin / max(0.001, approx_death)
        
        # Susun array 7 fitur sesuai susunan model aslimu
        base_input = np.array([[prov_kode, tahun, minggu, approx_vaksin, approx_death, approx_pct, approx_rasio]])
        base_scaled = scaler.transform(base_input)
        cluster_pred = int(rf_model.predict(base_scaled)[0])
        
        # 2. GET CENTROID
        centroid_scaled = kmeans_model.cluster_centers_[cluster_pred]
        centroid_raw = scaler.inverse_transform([centroid_scaled])[0]
        
        # 3. PROVINSI-SPECIFIC FACTORS FOR DISPLAY
        urban_scale = prov_info['urban']
        risk_scale = prov_info['risk_factor']
        
        SCALING_FACTOR = 100_000
        vaksin_base = int(max(1, abs(centroid_raw[3])) * SCALING_FACTOR)
        death_base = int(max(1, abs(centroid_raw[4])) * SCALING_FACTOR)
        
        # 4. FINAL CALCULATIONS
        vaksin_prov = int(vaksin_base * pop_scale * urban_scale * 1.3)
        death_prov = int(death_base * pop_scale * risk_scale * 0.9)
        
        vaksin_final = int(vaksin_prov * (1 + trend_boost/100 * years_ahead))
        death_final = int(death_prov * max(0.3, 1 - 0.12 * years_ahead))
        
        rasio_vk = vaksin_final / max(1, death_final)
        mortality_pct = (death_final / vaksin_final) * 100
        
        # === RESULTS DISPLAY ===
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h2 style='color:white; margin:0;'>💉</h2>
                <h1 style='color:white; margin:10px 0 5px 0;'>{vaksin_final:,}</h1>
                <p style='margin:0; font-size:1.1rem'>{prov_info['nama']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container" style='background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);'>
                <h2 style='color:white; margin:0;'>☠️</h2>
                <h1 style='color:white; margin:10px 0 5px 0;'>{death_final:,}</h1>
                <p style='margin:0; font-size:1.1rem'>Kematian</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container" style='background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);'>
                <h2 style='color:white; margin:0;'>⭐</h2>
                <h1 style='color:white; margin:10px 0 5px 0;'>{cluster_pred}</h1>
                <p style='margin:0; font-size:1.1rem'>{CLUSTER_INTERPRETASI.get(cluster_pred, "Status Tidak Diketahui")}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container" style='background: linear-gradient(135deg, #F7DC6F 0%, #D68910 100%);'>
                <h2 style='color:white; margin:0;'>📊</h2>
                <h1 style='color:white; margin:10px 0 5px 0;'>{rasio_vk:.1f}</h1>
                <p style='margin:0; font-size:1.1rem'>Rasio V/K</p>
            </div>
            """, unsafe_allow_html=True)
        
        # === SUMMARY ===
        st.markdown("---")
        st.markdown(f"""
        <div class="future-badge">
            🔮 **PREDIKSI {tahun}: {prov_info['nama']}** | 
            {CLUSTER_INTERPRETASI.get(cluster_pred)}
        </div>
        """, unsafe_allow_html=True)
        
        # Detail table
        st.markdown("## 📋 **Detail Prediksi**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("💉 Vaksinasi Disarankan", f"{vaksin_final:,}", f"+{trend_boost*years_ahead}%")
            st.metric("☠️ Prediksi Kematian", f"{death_final:,}", f"-{12*years_ahead}%")
            st.metric("📊 Mortalitas Rate", f"{mortality_pct:.2f}%")
        
        with col2:
            st.metric("📈 Rasio Vaksin/Kematian", f"{rasio_vk:.1f}:1")
            st.metric("⭐ Cluster Risk", f"Cluster {cluster_pred}")
            st.metric("👥 Populasi Basis", f"{prov_info['populasi']:,}")
        
        # === CHARTS ===
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=['Vaksinasi', 'Kematian'],
                y=[vaksin_final, death_final],
                title=f"📊 {prov_info['nama']} - {tahun}",
                color=['Vaksinasi', 'Kematian'],
                color_discrete_sequence=['#00D4AA', '#FF6B6B'],
                text=[f"{vaksin_final:,}", f"{death_final:,}"]
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(showlegend=False, height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=rasio_vk,
                delta={'reference': 10},
                title={'text': f"Rasio V/K - {tahun}"},
                gauge={
                    'axis': {'range': [0, 150]},
                    'bar': {'color': "#4ECDC4"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 80], 'color': "yellow"},
                        {'range': [80, 150], 'color': "orange"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        # === PROVINSI COMPARISON ===
        st.markdown("## 🌍 **Perbandingan Antar Provinsi**")
        prov_compare = [31, 32, 34, 11, prov_kode]
        vaksin_list, death_list, prov_names = [], [], []
        
        for pk in prov_compare:
            if pk in PROVINSI_DATA:
                pinfo = PROVINSI_DATA[int(pk)]
                v = int(1000 * pinfo['populasi'] / 10_000_000)
                vaksin_list.append(v * 1.2)
                death_list.append(int(v * 0.08))
                prov_names.append(pinfo['nama'][:12])
        
        fig2 = px.bar(
            x=prov_names, y=vaksin_list,
            title="Vaksinasi Relatif (2025)",
            color=death_list,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig2, use_container_width=True)

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align:center; padding:2rem; color:#666; font-size:1.1rem;'>
    <strong>🤖 Powered by K-Means Clustering + Random Forest</strong><br>
    📊 Akurasi Model 98.78% | 🔮 Prediksi Masa Depan 2025-2030 | 🛠️ Streamlit Cloud
</div>
""", unsafe_allow_html=True)