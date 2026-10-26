# database.py - Koneksi & Operasi MySQL
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "deteksi_objek_db"),
    "charset": "utf8mb4"
}

def get_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        raise Exception(f"Gagal koneksi ke MySQL: {e}")

def init_db():
    """Cek koneksi & seed admin jika kosong."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO users (username,password,nama,role) VALUES (%s,SHA2(%s,256),%s,%s)",
            ("admin", "admin123", "Administrator", "Admin")
        )
        conn.commit()
    cur.close(); conn.close()

# -------- USER --------
def check_login(username, password):
    conn = get_connection(); cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT * FROM users WHERE username=%s AND password=SHA2(%s,256)",
        (username, password)
    )
    row = cur.fetchone()
    cur.close(); conn.close()
    return row  # None kalau gagal

def register_user(username, password, nama):
    conn = get_connection(); cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM users WHERE username=%s", (username,))
        if cur.fetchone():
            return False, "Username sudah digunakan!"
        cur.execute(
            "INSERT INTO users (username,password,nama,role) VALUES (%s,SHA2(%s,256),%s,'User')",
            (username, password, nama)
        )
        conn.commit()
        return True, "Registrasi berhasil!"
    except Error as e:
        conn.rollback()
        return False, f"Gagal registrasi: {e}"
    finally:
        cur.close(); conn.close()

def get_all_users():
    conn = get_connection(); cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id,username,nama,role,created_at FROM users ORDER BY created_at DESC")
    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows

# -------- DETEKSI --------
def simpan_deteksi(username, nama_file, model, conf_threshold, detections, sumber="upload"):
    conn = get_connection(); cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO deteksi (username,nama_file,model,confidence_threshold,total_objek,sumber)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (username, nama_file, model, conf_threshold, len(detections), sumber)
        )
        deteksi_id = cur.lastrowid

        for d in detections:
            cur.execute(
                """INSERT INTO detail_deteksi
                   (deteksi_id,objek,objek_id,confidence,x1,y1,x2,y2)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (deteksi_id, d["objek"], d.get("objek_id", ""),
                 d["confidence"], d["x1"], d["y1"], d["x2"], d["y2"])
            )
        conn.commit()
        return deteksi_id
    except Error as e:
        conn.rollback()
        raise Exception(f"Gagal simpan deteksi: {e}")
    finally:
        cur.close(); conn.close()

def ambil_history(username, limit=100):
    conn = get_connection(); cur = conn.cursor(dictionary=True)
    cur.execute(
        """SELECT * FROM deteksi WHERE username=%s
           ORDER BY waktu DESC LIMIT %s""",
        (username, limit)
    )
    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows

def ambil_detail(deteksi_id):
    conn = get_connection(); cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM detail_deteksi WHERE deteksi_id=%s", (deteksi_id,))
    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows

def hapus_history(username):
    conn = get_connection(); cur = conn.cursor()
    cur.execute("DELETE FROM deteksi WHERE username=%s", (username,))
    conn.commit()
    cur.close(); conn.close()

def statistik_global(username):
    conn = get_connection(); cur = conn.cursor(dictionary=True)
    cur.execute(
        """SELECT COUNT(*) AS total_deteksi,
                  COALESCE(SUM(total_objek),0) AS total_objek
           FROM deteksi WHERE username=%s""",
        (username,)
    )
    row = cur.fetchone()
    cur.close(); conn.close()
    return row

def objek_terbanyak(username, limit=10):
    conn = get_connection(); cur = conn.cursor(dictionary=True)
    cur.execute(
        """SELECT d.objek, COUNT(*) AS jumlah
           FROM detail_deteksi d
           JOIN deteksi x ON x.id = d.deteksi_id
           WHERE x.username=%s
           GROUP BY d.objek
           ORDER BY jumlah DESC LIMIT %s""",
        (username, limit)
    )
    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows
