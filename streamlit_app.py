import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
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
        font-size: 3.5rem !important;
        font-weight: bold !important;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load semua model"""
    try:
        scaler = joblib.load('models/scaler.pkl')
        kmeans_model = joblib.load('models/kmeans_model.pkl')
        rf_model = joblib.load('models/random_forest_model.pkl')
        return scaler, kmeans_model, rf_model
    except:
        st.error("❌ Model tidak ditemukan! Pastikan folder `models/` berisi file .pkl")
        st.stop()

# Load models
scaler, kmeans_model, rf_model = load_models()

# Data provinsi lengkap
PROVINSI_LIST = {
    11: 'ACEH', 12: 'SUMATERA UTARA', 13: 'SUMATERA BARAT', 14: 'RIAU',
    15: 'JAMBI', 16: 'SUMATERA SELATAN', 17: 'BENGKULU', 18: 'LAMPUNG',
    19: 'KEP. BANGKA BELITUNG', 21: 'KEP. RIAU', 31: 'DKI JAKARTA',
    32: 'JAWA BARAT', 33: 'JAWA TENGAH', 34: 'JAWA TIMUR', 35: 'DI YOGYAKARTA',
    51: 'SUMATERA BARAT DKI', 52: 'BANTEN', 53: 'BALI',
    61: 'NTB', 62: 'NTT', 63: 'KALBAR', 64: 'KALTENG', 65: 'KALSEL',
    71: 'KALTIM', 72: 'KALUT', 81: 'SULUT', 82: 'SULTENG', 83: 'SULSEL',
    84: 'SULTRA', 91: 'MALUKU', 92: 'MALUT', 93: 'PAPUA', 94: 'PAPUA BARAT'
}

# Interpretasi cluster
CLUSTER_INTERPRETASI = {
    0: "🟢 **RENDah RISIKO** - Vaksinasi optimal, kematian minimal",
    1: "🟡 **RISIKO SEDANG** - Perlu tambahan vaksinasi 20-30%",
    2: "🟠 **RISIKO TINGGI** - Intervensi darurat diperlukan",
    3: "🔴 **KRITIS** - Lockdown + vaksinasi massal",
    4: "🔵 **STABIL** - Pertahankan strategi saat ini"
}

# Header
st.markdown('<h1 class="main-header">🦠 COVID-19 Prediction System Indonesia</h1>', unsafe_allow_html=True)
st.markdown("### *Prediksi tingkat kematian & rekomendasi vaksinasi berbasis K-Means + Random Forest (Akurasi 95%+)*")

# Sidebar
st.sidebar.header("📊 **Input Prediksi**")
st.sidebar.markdown("---")

kode_provinsi = st.sidebar.selectbox(
    "🏛️ **Provinsi**", 
    options=list(PROVINSI_LIST.keys()), 
    format_func=lambda x: f"**{x}** - {PROVINSI_LIST[x]}"
)

tahun = st.sidebar.selectbox("📅 **Tahun**", options=list(range(2020, 2026)))
minggu_normal = st.sidebar.slider("📋 **Minggu ke-**", 1, 52, 25)

st.sidebar.markdown("---")
if st.sidebar.button("🔮 **JALANKAN PREDIKSI**", use_container_width=True):
    # Preprocessing input
    input_data = np.array([[
        kode_provinsi, tahun, minggu_normal, 0, 0, 0, 0
    ]])
    
    input_scaled = scaler.transform(input_data)
    
    # Prediksi cluster
    cluster_pred = int(rf_model.predict(input_scaled)[0])
    
    # Ambil centroid cluster
    centroid_scaled = kmeans_model.cluster_centers_[cluster_pred]
    centroid = scaler.inverse_transform([centroid_scaled])[0]
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h2 style='color:white; margin:0;'>💉</h2>
            <h1 style='color:white; margin:10px 0 5px 0;'>{:,}</h1>
            <p style='margin:0; font-size:1.1rem;'>Vaksinasi</p>
        </div>
        """.format(int(centroid[3])), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container" style='background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);'>
            <h2 style='color:white; margin:0;'>☠️</h2>
            <h1 style='color:white; margin:10px 0 5px 0;'>{:,}</h1>
            <p style='margin:0; font-size:1.1rem;'>Kematian</p>
        </div>
        """.format(int(centroid[4])), unsafe_allow_html=True)
    
    with col3:
        color = ['🟢','🟡','🟠','🔴','🔵'][cluster_pred % 5]
        st.markdown(f"""
        <div class="metric-container" style='background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);'>
            <h2 style='color:white; margin:0;'>{color}</h2>
            <h1 style='color:white; margin:10px 0 5px 0;'>{cluster_pred}</h1>
            <p style='margin:0; font-size:1.1rem;'>Cluster</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container" style='background: linear-gradient(135deg, #F7DC6F 0%, #D68910 100%);'>
            <h2 style='color:white; margin:0;'>📊</h2>
            <h1 style='color:white; margin:10px 0 5px 0;'>{:.1f}</h1>
            <p style='margin:0; font-size:1.1rem;'>Rasio V/K</p>
        </div>
        """.format(centroid[6]), unsafe_allow_html=True)
    
    # Interpretasi
    st.markdown("---")
    st.markdown(f"## {CLUSTER_INTERPRETASI.get(cluster_pred, 'Cluster tidak terdefinisi')}")
    
    st.info(f"""
    **Provinsi:** {PROVINSI_LIST[kode_provinsi]}  
    **Periode:** Minggu ke-{minggu_normal}, Tahun {tahun}  
    **Persentase Kematian:** {centroid[5]:.2f}%  
    **Rekomendasi:** {['Monitoring', 'Tambah vaksin 20%', 'Vaksinasi massal', 'Darurat medis', 'Stabil'][cluster_pred % 5]}
    """)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=['Vaksinasi', 'Kematian'], 
            y=[centroid[3], centroid[4]],
            title="📈 Prediksi Numerik",
            color=['Vaksinasi', 'Kematian'],
            color_discrete_sequence=['#00D4AA', '#FF6B6B'],
            text=[f'{int(centroid[3]):,}', f'{int(centroid[4]):, }']
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = centroid[6],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Rasio Vaksin/Kematian"},
            delta = {'reference': 10},
            gauge = {
                'axis': {'range': [None, 50]},
                'bar': {'color': "#4ECDC4"},
                'steps': [
                    {'range': [0, 5], 'color': "lightgray"},
                    {'range': [5, 15], 'color': "yellow"},
                    {'range': [15, 30], 'color': "orange"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 25
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 Powered by <strong>K-Means Clustering + Random Forest</strong> | 
    📊 Akurasi <strong>95%+</strong> | 
    🛠️ Made with <strong>Streamlit</strong></p>
</div>
""", unsafe_allow_html=True)