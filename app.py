# app.py - Aplikasi Deteksi Objek Canggih
# Jalankan dengan: py -m streamlit run app.py

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import hashlib
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# ============================================
# KONFIGURASI HALAMAN
# ============================================

st.set_page_config(
    page_title="Deteksi Objek AI Canggih",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# DATABASE USER SEDERHANA
# ============================================

USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r') as f:
            return json.load(f)
    else:
        users = {
            "admin": {
                "password": hashlib.sha256("admin123".encode()).hexdigest(),
                "nama": "Administrator",
                "role": "Admin"
            }
        }
        save_users(users)
        return users

def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def register_user(username, password, nama):
    users = load_users()
    if username in users:
        return False, "Username sudah digunakan!"
    users[username] = {
        "password": hashlib.sha256(password.encode()).hexdigest(),
        "nama": nama,
        "role": "User"
    }
    save_users(users)
    return True, "Registrasi berhasil!"

def check_login(username, password):
    users = load_users()
    if username in users:
        hashed = hashlib.sha256(password.encode()).hexdigest()
        return hashed == users[username]["password"]
    return False

# ============================================
# SESSION STATE
# ============================================

if 'login' not in st.session_state:
    st.session_state.login = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'nama' not in st.session_state:
    st.session_state.nama = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'hasil_gambar' not in st.session_state:
    st.session_state.hasil_gambar = None
if 'deteksi' not in st.session_state:
    st.session_state.deteksi = []
if 'halaman' not in st.session_state:
    st.session_state.halaman = "login"
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.25
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "yolov8n.pt"
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# ============================================
# FUNGSI
# ============================================

@st.cache_resource
def load_model(model_path):
    with st.spinner(f"Memuat model {model_path}..."):
        try:
            return YOLO(model_path)
        except:
            st.warning(f"Model {model_path} belum diunduh. Mengunduh sekarang...")
            return YOLO(model_path)

def deteksi_objek(model, gambar, nama_file, confidence=0.25, filter_class=None):
    gambar_array = np.array(gambar)
    gambar_bgr = cv2.cvtColor(gambar_array, cv2.COLOR_RGB2BGR)
    hasil = model(gambar_bgr, conf=confidence)
    
    detections = []
    if hasil[0].boxes:
        for box in hasil[0].boxes:
            class_name = model.names[int(box.cls[0])]
            conf_score = float(box.conf[0])
            
            # Filter jika diperlukan
            if filter_class and filter_class != "Semua":
                if class_name.lower() != filter_class.lower():
                    continue
                    
            detections.append({
                "objek": class_name,
                "confidence": conf_score,
                "x1": float(box.xyxy[0][0]),
                "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]),
                "y2": float(box.xyxy[0][3])
            })
    
    gambar_hasil = cv2.cvtColor(hasil[0].plot(), cv2.COLOR_BGR2RGB)
    
    # Simpan ke history
    st.session_state.history.append({
        'waktu': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'file': nama_file,
        'total': len(detections),
        'objek': ', '.join([d['objek'] for d in detections]) if detections else 'Tidak ada',
        'detail_objek': detections,
        'model': st.session_state.selected_model,
        'confidence_threshold': confidence
    })
    
    return gambar_hasil, detections

def download_image(image_array, filename):
    """Fungsi untuk mendownload gambar hasil deteksi"""
    img = Image.fromarray(image_array)
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def export_history():
    """Export history ke CSV atau JSON"""
    if not st.session_state.history:
        return None
    
    data = []
    for h in st.session_state.history:
        data.append({
            'Waktu': h['waktu'],
            'File': h['file'],
            'Total Objek': h['total'],
            'Objek': h['objek'],
            'Model': h.get('model', 'yolov8n.pt')
        })
    
    df = pd.DataFrame(data)
    return df

def get_class_list():
    """Mendapatkan daftar kelas dari model COCO"""
    return ['Semua', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# ============================================
# HALAMAN LOGIN
# ============================================

def halaman_login():
    st.markdown("""
    <h1 style='text-align:center; color:#667eea;'>🎯 DETEKSI OBJEK AI</h1>
    <p style='text-align:center;'>Sistem Deteksi Objek Cerdas dengan AI</p>
    <hr>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        username = st.text_input("Username", placeholder="Masukkan username")
        password = st.text_input("Password", type="password", placeholder="Masukkan password")
        
        if st.button("🔓 LOGIN", use_container_width=True):
            if check_login(username, password):
                users = load_users()
                st.session_state.login = True
                st.session_state.username = username
                st.session_state.nama = users[username]["nama"]
                st.rerun()
            else:
                st.error("❌ Username atau password salah!")
        
        if st.button("📝 DAFTAR AKUN BARU", use_container_width=True):
            st.session_state.halaman = "register"
            st.rerun()
        
        st.info("🔑 Akun demo: admin / admin123")

# ============================================
# HALAMAN REGISTRASI
# ============================================

def halaman_register():
    st.markdown("<h1 style='text-align:center'>📝 DAFTAR AKUN BARU</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("form_register"):
            username = st.text_input("Username", placeholder="Buat username")
            nama = st.text_input("Nama Lengkap", placeholder="Nama Anda")
            password = st.text_input("Password", type="password", placeholder="Buat password")
            confirm = st.text_input("Konfirmasi Password", type="password", placeholder="Ulangi password")
            
            if st.form_submit_button("📝 DAFTAR", use_container_width=True):
                if len(username) < 3:
                    st.error("Username minimal 3 karakter!")
                elif len(password) < 4:
                    st.error("Password minimal 4 karakter!")
                elif password != confirm:
                    st.error("Password tidak sama!")
                else:
                    ok, msg = register_user(username, password, nama)
                    if ok:
                        st.success(msg)
                        st.balloons()
                        st.info("Silakan login")
                        if st.button("KE HALAMAN LOGIN"):
                            st.session_state.halaman = "login"
                            st.rerun()
                    else:
                        st.error(msg)
        
        if st.button("◀ KEMBALI KE LOGIN", use_container_width=True):
            st.session_state.halaman = "login"
            st.rerun()

# ============================================
# HALAMAN DASHBOARD
# ============================================

def halaman_dashboard():
    st.markdown("## 📊 DASHBOARD ANALYTICS")
    
    # Metrics
    total_detections = len(st.session_state.history)
    total_objects = sum(h['total'] for h in st.session_state.history)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Deteksi", total_detections)
    col2.metric("Total Objek", total_objects)
    col3.metric("Rata-rata/Deteksi", f"{total_objects/total_detections:.1f}" if total_detections > 0 else "0")
    col4.metric("User", st.session_state.nama)
    
    if st.session_state.history:
        # Chart: Deteksi per hari
        df = export_history()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Statistik Objek Terdeteksi")
            # Hitung frekuensi objek
            all_objects = []
            for h in st.session_state.history:
                if h['detail_objek']:
                    for obj in h['detail_objek']:
                        all_objects.append(obj['objek'])
            
            if all_objects:
                obj_counts = pd.Series(all_objects).value_counts().head(10)
                fig = px.bar(x=obj_counts.values, y=obj_counts.index, orientation='h', 
                            title="Top 10 Objek Terbanyak", 
                            labels={'x': 'Jumlah', 'y': 'Objek'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Distribusi Confidence")
            all_confidences = []
            for h in st.session_state.history:
                if h['detail_objek']:
                    for obj in h['detail_objek']:
                        all_confidences.append(obj['confidence'])
            
            if all_confidences:
                fig = px.histogram(x=all_confidences, nbins=20, 
                                  title="Distribusi Confidence Score",
                                  labels={'x': 'Confidence', 'y': 'Frekuensi'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Riwayat dengan filter dan search
        st.markdown("---")
        st.markdown("### 📜 Riwayat Deteksi")
        
        col1, col2 = st.columns([3,1])
        with col2:
            search = st.text_input("🔍 Cari file atau objek")
        with col1:
            pass
        
        filtered_history = st.session_state.history.copy()
        if search:
            filtered_history = [h for h in filtered_history if 
                              search.lower() in h['file'].lower() or 
                              search.lower() in h['objek'].lower()]
        
        for h in filtered_history[-10:][::-1]:
            with st.expander(f"🕐 {h['waktu']} - 📁 {h['file']} - 🎯 {h['total']} objek"):
                st.write(f"**Model:** {h.get('model', 'yolov8n.pt')}")
                st.write(f"**Objek:** {h['objek']}")
                if h.get('detail_objek'):
                    df_detail = pd.DataFrame(h['detail_objek'])
                    st.dataframe(df_detail[['objek', 'confidence']], use_container_width=True)
        
        # Export data
        st.markdown("---")
        st.markdown("### 💾 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export ke CSV", use_container_width=True):
                df_export = export_history()
                csv = df_export.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="detection_history.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("File CSV siap download!")
        
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()
    else:
        st.info("Belum ada riwayat deteksi. Upload gambar untuk memulai!")

# ============================================
# HALAMAN DETEKSI
# ============================================

def halaman_deteksi():
    st.markdown("## 🎯 DETEKSI OBJEK")
    st.markdown("Upload gambar atau gunakan webcam untuk mendeteksi objek")
    
    # Sidebar settings untuk deteksi
    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan Deteksi")
        
        # Model selection
        model_option = st.selectbox(
            "Pilih Model AI",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            index=0,
            help="n=small/fast, s=medium, m=large/accurate"
        )
        if model_option != st.session_state.selected_model:
            st.session_state.selected_model = model_option
            st.cache_resource.clear()
        
        # Confidence threshold
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_threshold,
            step=0.05,
            help="Semakin tinggi, semakin akurat tapi lebih sedikit objek"
        )
        st.session_state.confidence_threshold = confidence
        
        # Filter class
        filter_class = st.selectbox("Filter Objek Spesifik", get_class_list())
        
        st.markdown("---")
        st.info(f"💡 Tips: Confidence {confidence:.0%} adalah pengaturan saat ini")
    
    model = load_model(st.session_state.selected_model)
    
    # Tab untuk upload dan camera
    tab1, tab2 = st.tabs(["📤 Upload Gambar", "📷 Live Camera"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded = st.file_uploader("Upload Gambar", type=['jpg', 'jpeg', 'png'], key="upload_multiple")
            if uploaded:
                gambar = Image.open(uploaded)
                st.image(gambar, caption="Gambar Asli", use_container_width=True)
                
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    if st.button("🚀 DETEKSI", type="primary", use_container_width=True):
                        with st.spinner("Mendeteksi..."):
                            hasil, deteksi = deteksi_objek(
                                model, gambar, uploaded.name, 
                                confidence, 
                                None if filter_class == "Semua" else filter_class
                            )
                            st.session_state.hasil_gambar = hasil
                            st.session_state.deteksi = deteksi
                            st.rerun()
        
        with col2:
            if st.session_state.hasil_gambar is not None:
                st.image(st.session_state.hasil_gambar, caption="Hasil Deteksi", use_container_width=True)
                
                # Download button
                img_bytes = download_image(st.session_state.hasil_gambar, "hasil_deteksi.png")
                st.download_button(
                    label="💾 Download Hasil Deteksi",
                    data=img_bytes,
                    file_name=f"deteksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                st.markdown("---")
                st.markdown("### 📋 Hasil Deteksi")
                if st.session_state.deteksi:
                    # Tampilkan dalam columns
                    cols = st.columns(3)
                    for idx, d in enumerate(st.session_state.deteksi):
                        with cols[idx % 3]:
                            st.success(f"✅ **{d['objek'].upper()}**")
                            st.progress(d['confidence'])
                            st.caption(f"Confidence: {d['confidence']:.1%}")
                else:
                    st.warning("⚠️ Tidak ada objek terdeteksi dengan confidence yang dipilih")
    
    with tab2:
        st.markdown("### Live Camera Detection")
        st.warning("⚠️ Untuk keamanan, deteksi live camera hanya berjalan saat tombol ditekan")
        
        camera_image = st.camera_input("Ambil foto dari webcam")
        
        if camera_image:
            gambar = Image.open(camera_image)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(gambar, caption="Foto dari Kamera", use_container_width=True)
            
            with col2:
                if st.button("🔍 Deteksi Foto Ini", type="primary"):
                    with st.spinner("Mendeteksi..."):
                        hasil, deteksi = deteksi_objek(
                            model, gambar, f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                            confidence,
                            None if filter_class == "Semua" else filter_class
                        )
                        st.image(hasil, caption="Hasil Deteksi", use_container_width=True)
                        
                        if deteksi:
                            st.success(f"✅ Mendeteksi {len(deteksi)} objek!")
                            for d in deteksi[:5]:  # Tampilkan 5 pertama
                                st.write(f"- {d['objek']} ({d['confidence']:.1%})")
                        else:
                            st.warning("Tidak ada objek terdeteksi")

# ============================================
# HALAMAN STATISTIK
# ============================================

def halaman_statistik():
    st.markdown("## 📈 Statistik Lanjutan")
    
    if not st.session_state.history:
        st.info("Belum ada data deteksi. Silakan lakukan deteksi objek terlebih dahulu.")
        return
    
    # Timeline
    st.markdown("### 📅 Timeline Deteksi")
    df = export_history()
    
    fig = px.line(df, x='Waktu', y='Total Objek', title='Trend Deteksi Objek per Waktu',
                 markers=True, labels={'Total Objek': 'Jumlah Objek', 'Waktu': 'Waktu Deteksi'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Performance
    st.markdown("### 🚀 Performa per Model")
    model_performance = df.groupby('Model')['Total Objek'].agg(['mean', 'sum', 'count']).round(2)
    model_performance.columns = ['Rata-rata Objek', 'Total Objek', 'Jumlah Deteksi']
    st.dataframe(model_performance, use_container_width=True)
    
    # Object co-occurrence
    st.markdown("### 🔗 Objek yang Sering Muncul Bersama")
    all_pairs = []
    for h in st.session_state.history:
        if h['detail_objek'] and len(h['detail_objek']) > 1:
            objects = [obj['objek'] for obj in h['detail_objek']]
            for i in range(len(objects)):
                for j in range(i+1, len(objects)):
                    all_pairs.append(f"{objects[i]} + {objects[j]}")
    
    if all_pairs:
        pair_counts = pd.Series(all_pairs).value_counts().head(10)
        fig = px.bar(x=pair_counts.values, y=pair_counts.index, orientation='h',
                    title="Top 10 Pasangan Objek yang Sering Muncul",
                    labels={'x': 'Frekuensi', 'y': 'Pasangan Objek'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# SIDEBAR
# ============================================

def sidebar():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.nama}")
        st.markdown(f"@{st.session_state.username}")
        st.markdown("---")
        
        menu = st.radio("MENU", ["🎯 Deteksi Objek", "📊 Dashboard", "📈 Statistik"])
        
        st.markdown("---")
        
        # Informasi
        with st.expander("ℹ️ Tentang Aplikasi"):
            st.markdown("""
            **Deteksi Objek AI Canggih**
            
            - Model: YOLOv8
            - Dataset: COCO (80 kelas)
            - Real-time detection
            - Export data & visualisasi
            """)
        
        st.markdown("---")
        if st.button("🚪 LOGOUT", use_container_width=True):
            for key in ['login', 'username', 'nama', 'hasil_gambar', 'deteksi']:
                if key in st.session_state:
                    st.session_state[key] = None if key != 'login' else False
            st.session_state.login = False
            st.rerun()
        
        st.caption("© 2024 Deteksi Objek AI Canggih | v2.0")
    
    return menu

# ============================================
# MAIN PROGRAM
# ============================================

def main():
    if not st.session_state.login:
        if st.session_state.halaman == "register":
            halaman_register()
        else:
            halaman_login()
    else:
        menu = sidebar()
        
        st.markdown("""
        <div style='background:linear-gradient(135deg,#667eea,#764ba2); padding:20px; border-radius:10px; color:white; text-align:center; margin-bottom:20px'>
            <h1>🎯 DETEKSI OBJEK AI</h1>
            <p>Sistem Deteksi Objek Cerdas dengan AI - Support Multi Model & Live Camera</p>
        </div>
        """, unsafe_allow_html=True)
        
        if menu == "🎯 Deteksi Objek":
            halaman_deteksi()
        elif menu == "📊 Dashboard":
            halaman_dashboard()
        else:
            halaman_statistik()

if __name__ == "__main__":
    main()