# app.py - Aplikasi Deteksi Objek Canggih (MySQL + UI Modern)
# Jalankan: streamlit run app.py

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

import database as db
from kamus_objek import terjemahkan, daftar_kelas_id, NAMA_OBJEK_ID

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Deteksi Objek AI - Praktis & Cerdas",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS KUSTOM - UI KEREN
# ============================================
def inject_css():
    st.markdown("""
    <style>
    /* Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }

    /* Teks dasar */
    .stApp, .stMarkdown, p, span, label, h1, h2, h3, h4, h5, h6 { color: #e8e8ff !important; }

    /* Sidebar glass */
    [data-testid="stSidebar"] {
        background: rgba(20, 20, 40, 0.75) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Kartu glass */
    .glass-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 22px;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        margin-bottom: 16px;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg,#667eea 0%,#764ba2 50%,#f093fb 100%);
        padding: 40px 20px;
        border-radius: 22px;
        text-align: center;
        color: white;
        box-shadow: 0 12px 40px rgba(118,75,162,0.45);
        margin-bottom: 26px;
        animation: fadeIn 0.8s ease-in-out;
    }
    .hero h1 { color: #fff !important; font-size: 2.1rem; margin: 0; font-weight: 700; }
    .hero p { color: #f5f0ff !important; margin-top: 8px; }

    @keyframes fadeIn { from {opacity:0; transform:translateY(12px);} to {opacity:1; transform:translateY(0);} }

    /* Tombol */
    .stButton > button {
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: #fff !important;
        border: none;
        border-radius: 12px;
        padding: 10px 22px;
        font-weight: 600;
        transition: all 0.25s ease;
        box-shadow: 0 4px 16px rgba(102,126,234,0.45);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(102,126,234,0.65);
        background: linear-gradient(135deg,#764ba2,#f093fb);
    }

    /* Metric card */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 16px;
        padding: 14px 18px;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricValue"] { color: #b8b0ff !important; font-weight: 700; }

    /* Input */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stSlider, .stTextArea textarea {
        background: rgba(255,255,255,0.07) !important;
        border-radius: 12px !important;
        color: #fff !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
    }

    /* Tab */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.06);
        border-radius: 12px 12px 0 0;
        color: #c9c4ff;
        padding: 10px 18px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg,#667eea,#764ba2) !important;
        color: #fff !important;
        box-shadow: 0 4px 14px rgba(102,126,234,0.5);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        color: #d0ccff !important;
    }

    /* Hasil deteksi badge */
    .objek-badge {
        display: inline-block;
        background: linear-gradient(135deg,#00c9a7,#00b4d8);
        color: #fff !important;
        padding: 8px 14px;
        border-radius: 12px;
        margin: 4px 4px 8px 0;
        font-weight: 600;
        font-size: 0.92rem;
        box-shadow: 0 4px 14px rgba(0,201,167,0.4);
        animation: pop 0.4s ease-out;
    }
    @keyframes pop { from { transform: scale(0.85); opacity:0;} to {transform:scale(1); opacity:1;} }

    /* Info/alert */
    .stAlert { border-radius: 14px !important; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.12) !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: rgba(0,0,0,0.3); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg,#667eea,#764ba2);
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ============================================
# INISIALISASI DATABASE
# ============================================
@st.cache_resource
def init_database():
    try:
        db.init_db()
        return True, "Terhubung ke MySQL"
    except Exception as e:
        return False, str(e)

db_ok, db_msg = init_database()

# ============================================
# SESSION STATE
# ============================================
defaults = {
    'login': False, 'username': None, 'nama': None, 'role': None,
    'hasil_gambar': None, 'deteksi': [], 'halaman': "login",
    'confidence_threshold': 0.25, 'selected_model': "yolov8n.pt",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================
# FUNGSI BANTU
# ============================================
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def deteksi_objek(model, gambar, nama_file, sumber, confidence=0.25, filter_id=None):
    arr = np.array(gambar.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    hasil = model(bgr, conf=confidence)

    detections = []
    if hasil[0].boxes:
        for box in hasil[0].boxes:
            kelas_en = model.names[int(box.cls[0])]
            kelas_id = terjemahkan(kelas_en)
            conf_score = float(box.conf[0])

            if filter_id and filter_id != "Semua" and kelas_id != filter_id:
                continue

            detections.append({
                "objek": kelas_id,
                "objek_id": kelas_en,
                "confidence": conf_score,
                "x1": float(box.xyxy[0][0]), "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]), "y2": float(box.xyxy[0][3]),
            })

    gambar_hasil = cv2.cvtColor(hasil[0].plot(), cv2.COLOR_BGR2RGB)

    # Simpan ke MySQL
    try:
        db.simpan_deteksi(
            st.session_state.username, nama_file,
            st.session_state.selected_model, confidence,
            detections, sumber
        )
    except Exception as e:
        st.warning(f"⚠️ Gagal simpan ke database: {e}")

    return gambar_hasil, detections

def download_image(image_array):
    img = Image.fromarray(image_array)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def render_badge(objek, confidence):
    return f"""<span class="objek-badge">🔍 {objek} — {confidence:.1%}</span>"""

# ============================================
# HALAMAN LOGIN
# ============================================
def halaman_login():
    st.markdown("""
    <div class="hero">
        <h1>🎯 Deteksi Objek AI</h1>
        <p>Deteksi Gambar, Buah, Hewan & Benda Praktis dengan Kecerdasan Buatan</p>
    </div>
    """, unsafe_allow_html=True)

    if not db_ok:
        st.error(f"❌ Database MySQL tidak terhubung: {db_msg}")
        st.info("Pastikan MySQL berjalan dan konfigurasi di file `.env` sudah benar.")
        return

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔓 Masuk ke Akun Anda")
        username = st.text_input("Username", placeholder="Masukkan username")
        password = st.text_input("Kata Sandi", type="password", placeholder="Masukkan kata sandi")

        if st.button("🔓 MASUK", use_container_width=True):
            user = db.check_login(username, password)
            if user:
                st.session_state.login = True
                st.session_state.username = user["username"]
                st.session_state.nama = user["nama"]
                st.session_state.role = user["role"]
                st.success(f"Selamat datang, {user['nama']}!")
                st.rerun()
            else:
                st.error("❌ Username atau kata sandi salah!")

        if st.button("📝 DAFTAR AKUN BARU", use_container_width=True):
            st.session_state.halaman = "register"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.info("🔑 Akun demo: **admin / admin123**")

# ============================================
# HALAMAN REGISTRASI
# ============================================
def halaman_register():
    st.markdown("""
    <div class="hero">
        <h1>📝 Daftar Akun Baru</h1>
        <p>Buat akun Anda dan mulai deteksi objek secara cerdas</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        with st.form("form_register"):
            username = st.text_input("Username", placeholder="Minimal 3 karakter")
            nama = st.text_input("Nama Lengkap", placeholder="Nama Anda")
            password = st.text_input("Kata Sandi", type="password", placeholder="Minimal 4 karakter")
            confirm = st.text_input("Konfirmasi Kata Sandi", type="password", placeholder="Ulangi kata sandi")

            if st.form_submit_button("📝 DAFTAR SEKARANG", use_container_width=True):
                if len(username) < 3:
                    st.error("Username minimal 3 karakter!")
                elif len(password) < 4:
                    st.error("Kata sandi minimal 4 karakter!")
                elif password != confirm:
                    st.error("Kata sandi tidak sama!")
                else:
                    ok, msg = db.register_user(username, password, nama)
                    if ok:
                        st.success("✅ " + msg)
                        st.balloons()
                        st.session_state.halaman = "login"
                        st.rerun()
                    else:
                        st.error(msg)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("◀ KEMBALI KE LOGIN", use_container_width=True):
            st.session_state.halaman = "login"
            st.rerun()

# ============================================
# HALAMAN DETEKSI
# ============================================
def halaman_deteksi():
    st.markdown("""
    <div class="hero">
        <h1>🎯 Deteksi Objek</h1>
        <p>Unggah gambar atau ambil foto untuk mendeteksi objek secara otomatis</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan Deteksi")
        model_option = st.selectbox(
            "Pilih Model AI",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            index=["yolov8n.pt","yolov8s.pt","yolov8m.pt"].index(st.session_state.selected_model),
            help="n=cepat, s=seimbang, m=akurat"
        )
        if model_option != st.session_state.selected_model:
            st.session_state.selected_model = model_option
            st.cache_resource.clear()

        confidence = st.slider(
            "Ambang Keyakinan",
            0.0, 1.0, st.session_state.confidence_threshold, 0.05,
            help="Semakin tinggi, deteksi semakin ketat"
        )
        st.session_state.confidence_threshold = confidence

        filter_id = st.selectbox("Filter Objek Spesifik", daftar_kelas_id())
        st.markdown("---")
        st.info(f"💡 Ambang keyakinan saat ini: **{confidence:.0%}**")

    model = load_model(st.session_state.selected_model)

    tab1, tab2 = st.tabs(["📤 Unggah Gambar", "📷 Kamera Langsung"])

    # ---- Tab Upload ----
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            uploaded = st.file_uploader("Unggah Gambar", type=['jpg','jpeg','png'], key="upload")
            if uploaded:
                gambar = Image.open(uploaded).convert("RGB")
                st.image(gambar, caption="Gambar Asli", use_container_width=True)
                if st.button("🚀 DETEKSI SEKARANG", type="primary", use_container_width=True):
                    with st.spinner("🔎 Sedang mendeteksi objek..."):
                        hasil, deteksi = deteksi_objek(
                            model, gambar, uploaded.name, "upload",
                            confidence,
                            None if filter_id == "Semua" else filter_id
                        )
                        st.session_state.hasil_gambar = hasil
                        st.session_state.deteksi = deteksi
                        st.rerun()

        with col2:
            if st.session_state.hasil_gambar is not None:
                st.image(st.session_state.hasil_gambar, caption="Hasil Deteksi", use_container_width=True)

                st.download_button(
                    "💾 Unduh Hasil Deteksi",
                    data=download_image(st.session_state.hasil_gambar),
                    file_name=f"hasil_deteksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )

                st.markdown("### 📋 Objek Terdeteksi")
                if st.session_state.deteksi:
                    badges = "".join(
                        render_badge(d["objek"], d["confidence"])
                        for d in st.session_state.deteksi
                    )
                    st.markdown(badges, unsafe_allow_html=True)

                    df = pd.DataFrame([
                        {"Objek": d["objek"], "Keyakinan": f"{d['confidence']:.1%}"}
                        for d in st.session_state.deteksi
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.warning("⚠️ Tidak ada objek terdeteksi. Coba turunkan ambang keyakinan.")

    # ---- Tab Kamera ----
    with tab2:
        st.markdown("### 📷 Deteksi dari Kamera")
        st.warning("⚠️ Ambil foto terlebih dahulu, lalu tekan tombol deteksi.")
        cam = st.camera_input("Ambil foto dari kamera")
        if cam:
            gambar = Image.open(cam).convert("RGB")
            c1, c2 = st.columns(2)
            with c1:
                st.image(gambar, caption="Foto Kamera", use_container_width=True)
            with c2:
                if st.button("🔍 DETEKSI FOTO INI", type="primary", use_container_width=True):
                    with st.spinner("🔎 Mendeteksi..."):
                        hasil, deteksi = deteksi_objek(
                            model, gambar,
                            f"kamera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                            "kamera", confidence,
                            None if filter_id == "Semua" else filter_id
                        )
                        st.image(hasil, caption="Hasil Deteksi", use_container_width=True)
                        if deteksi:
                            st.success(f"✅ Berhasil mendeteksi {len(deteksi)} objek!")
                            badges = "".join(
                                render_badge(d["objek"], d["confidence"]) for d in deteksi
                            )
                            st.markdown(badges, unsafe_allow_html=True)
                        else:
                            st.warning("Tidak ada objek terdeteksi.")

# ============================================
# HALAMAN DASHBOARD
# ============================================
def halaman_dashboard():
    st.markdown("""
    <div class="hero">
        <h1>📊 Dashboard Analitik</h1>
        <p>Ringkasan aktivitas deteksi Anda</p>
    </div>
    """, unsafe_allow_html=True)

    stats = db.statistik_global(st.session_state.username)
    total_det = stats["total_deteksi"] or 0
    total_obj = int(stats["total_objek"] or 0)
    rata = (total_obj / total_det) if total_det else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📸 Total Deteksi", total_det)
    c2.metric("🎯 Total Objek", total_obj)
    c3.metric("📈 Rata-rata/Deteksi", f"{rata:.1f}")
    c4.metric("👤 Pengguna", st.session_state.nama)

    st.markdown("---")

    if total_det == 0:
        st.info("Belum ada riwayat deteksi. Silakan lakukan deteksi terlebih dahulu.")
        return

    # Chart objek terbanyak
    obj_data = db.objek_terbanyak(st.session_state.username, 10)
    if obj_data:
        df_obj = pd.DataFrame(obj_data)
        fig = px.bar(
            df_obj, x="jumlah", y="objek", orientation="h",
            title="🏆 10 Objek Paling Sering Terdeteksi",
            labels={"jumlah": "Jumlah", "objek": "Objek"},
            color="jumlah", color_continuous_scale="Purples"
        )
        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8e8ff"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Riwayat
    st.markdown("### 📜 Riwayat Deteksi")
    history = db.ambil_history(st.session_state.username, 20)

    search = st.text_input("🔍 Cari berdasarkan nama file atau objek")
    for h in history:
        detail = db.ambil_detail(h["id"])
        objek_str = ", ".join([d["objek"] for d in detail]) if detail else "Tidak ada"
        if search and search.lower() not in h["nama_file"].lower() and search.lower() not in objek_str.lower():
            continue

        with st.expander(
            f"🕐 {h['waktu'].strftime('%Y-%m-%d %H:%M:%S')} | "
            f"📁 {h['nama_file']} | 🎯 {h['total_objek']} objek"
        ):
            st.write(f"**Model:** {h['model']}")
            st.write(f"**Sumber:** {h['sumber'].title()}")
            st.write(f"**Ambang Keyakinan:** {h['confidence_threshold']:.0%}")
            if detail:
                df = pd.DataFrame([
                    {"Objek": d["objek"], "Keyakinan": f"{d['confidence']:.1%}"}
                    for d in detail
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Ekspor Riwayat ke CSV", use_container_width=True):
            all_rows = []
            for h in history:
                detail = db.ambil_detail(h["id"])
                if detail:
                    for d in detail:
                        all_rows.append({
                            "Waktu": h["waktu"], "File": h["nama_file"],
                            "Model": h["model"], "Objek": d["objek"],
                            "Keyakinan": d["confidence"], "Sumber": h["sumber"]
                        })
            if all_rows:
                csv = pd.DataFrame(all_rows).to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="riwayat_deteksi.csv" style="color:#00e0c6;font-weight:600;">⬇️ Unduh CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("Tidak ada data untuk diekspor.")
    with col2:
        if st.button("🗑️ Hapus Semua Riwayat", use_container_width=True):
            db.hapus_history(st.session_state.username)
            st.success("Riwayat berhasil dihapus.")
            st.rerun()

# ============================================
# HALAMAN STATISTIK
# ============================================
def halaman_statistik():
    st.markdown("""
    <div class="hero">
        <h1>📈 Statistik Lanjutan</h1>
        <p>Analisis mendalam atas hasil deteksi Anda</p>
    </div>
    """, unsafe_allow_html=True)

    history = db.ambil_history(st.session_state.username, 100)
    if not history:
        st.info("Belum ada data. Silakan lakukan deteksi terlebih dahulu.")
        return

    # Tren harian
    df_h = pd.DataFrame([{
        "Waktu": h["waktu"], "Total": h["total_objek"], "Model": h["model"],
        "Sumber": h["sumber"]
    } for h in history])

    fig = px.line(
        df_h.sort_values("Waktu"), x="Waktu", y="Total",
        markers=True, title="📈 Tren Deteksi per Waktu",
        labels={"Total": "Jumlah Objek", "Waktu": "Waktu"}
    )
    fig.update_layout(
        height=400, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e8ff"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Performa model
    st.markdown("### 🚀 Performa per Model")
    perf = df_h.groupby("Model")["Total"].agg(["mean","sum","count"]).round(2)
    perf.columns = ["Rata-rata Objek", "Total Objek", "Jumlah Deteksi"]
    st.dataframe(perf, use_container_width=True)

# ============================================
# SIDEBAR
# ============================================
def sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2.4rem;">👤</div>
            <h3 style="margin:6px 0;color:#b8b0ff !important;">{st.session_state.nama}</h3>
            <p style="margin:0;color:#9d97c9 !important;">@{st.session_state.username}</p>
            <small style="color:#7d77a8;">Peran: {st.session_state.role}</small>
        </div>
        """, unsafe_allow_html=True)

        menu = st.radio(
            "MENU UTAMA",
            ["🎯 Deteksi Objek", "📊 Dashboard", "📈 Statistik"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        with st.expander("ℹ️ Tentang Aplikasi"):
            st.markdown("""
            **Deteksi Objek AI**
            - Model: YOLOv8 (80 kelas COCO)
            - Deteksi: orang, hewan, buah, kendaraan, benda
            - Penyimpanan: MySQL
            - Ekspor riwayat ke CSV
            """)

        st.markdown("---")
        if st.button("🚪 KELUAR", use_container_width=True):
            for k in ['login','username','nama','role','hasil_gambar']:
                st.session_state[k] = None if k != 'login' else False
            st.session_state.deteksi = []
            st.rerun()

        st.caption("© 2024 Deteksi Objek AI v3.0")

    return menu

# ============================================
# PROGRAM UTAMA
# ============================================
def main():
    if not st.session_state.login:
        if st.session_state.halaman == "register":
            halaman_register()
        else:
            halaman_login()
    else:
        menu = sidebar()
        if menu == "🎯 Deteksi Objek":
            halaman_deteksi()
        elif menu == "📊 Dashboard":
            halaman_dashboard()
        else:
            halaman_statistik()

if __name__ == "__main__":
    main()
