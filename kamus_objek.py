# kamus_objek.py - Terjemahan 80 kelas COCO ke Bahasa Indonesia
NAMA_OBJEK_ID = {
    "person": "Orang", "bicycle": "Sepeda", "car": "Mobil",
    "motorcycle": "Motor", "airplane": "Pesawat", "bus": "Bus",
    "train": "Kereta", "truck": "Truk", "boat": "Perahu",
    "traffic light": "Lampu Lalu Lintas", "fire hydrant": "Hydrant",
    "stop sign": "Rambu Stop", "parking meter": "Meteran Parkir",
    "bench": "Bangku", "bird": "Burung", "cat": "Kucing",
    "dog": "Anjing", "horse": "Kuda", "sheep": "Domba",
    "cow": "Sapi", "elephant": "Gajah", "bear": "Beruang",
    "zebra": "Zebra", "giraffe": "Jerapah", "backpack": "Tas Punggung",
    "umbrella": "Payung", "handbag": "Tas Tangan", "tie": "Dasi",
    "suitcase": "Koper", "frisbee": "Frisbee", "skis": "Ski",
    "snowboard": "Snowboard", "sports ball": "Bola", "kite": "Layang-layang",
    "baseball bat": "Pemukul Bisbol", "baseball glove": "Sarung Tangan Bisbol",
    "skateboard": "Skateboard", "surfboard": "Papan Surf",
    "tennis racket": "Raket Tenis", "bottle": "Botol",
    "wine glass": "Gelas Wine", "cup": "Cangkir", "fork": "Garpu",
    "knife": "Pisau", "spoon": "Sendok", "bowl": "Mangkuk",
    "banana": "Pisang", "apple": "Apel", "sandwich": "Sandwich",
    "orange": "Jeruk", "broccoli": "Brokoli", "carrot": "Wortel",
    "hot dog": "Hot Dog", "pizza": "Pizza", "donut": "Donat",
    "cake": "Kue", "chair": "Kursi", "couch": "Sofa",
    "potted plant": "Tanaman Pot", "bed": "Tempat Tidur",
    "dining table": "Meja Makan", "toilet": "Toilet",
    "tv": "Televisi", "laptop": "Laptop", "mouse": "Mouse",
    "remote": "Remote", "keyboard": "Keyboard",
    "cell phone": "Ponsel", "microwave": "Microwave",
    "oven": "Oven", "toaster": "Pemanggang Roti", "sink": "Wastafel",
    "refrigerator": "Kulkas", "book": "Buku", "clock": "Jam",
    "vase": "Vas", "scissors": "Gunting", "teddy bear": "Boneka Beruang",
    "hair drier": "Pengering Rambut", "toothbrush": "Sikat Gigi",
}

def terjemahkan(nama_en: str) -> str:
    return NAMA_OBJEK_ID.get(nama_en, nama_en.title())

def daftar_kelas_id():
    return ["Semua"] + list(NAMA_OBJEK_ID.values())
