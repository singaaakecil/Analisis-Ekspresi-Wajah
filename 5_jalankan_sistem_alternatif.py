import cv2
import os
from deepface import DeepFace

print("[INFO] Memulai sistem...")

# --- Konfigurasi Awal ---
MODEL_PATH = 'model/model_latih.xml'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
KAMERA_INDEX = 1

# ====================================================================
# PENTING: Sesuaikan ID dan Nama di bawah ini dengan data Anda
NAMA_PENGGUNA = {
    1: "faris",      # Ganti "Nama Anda" sesuai nama untuk ID 1
    2: "ocid",
    3: "dito",
    4: "ikrom",
    5: "musang"    # Ganti "Nama Teman" sesuai nama untuk ID 2
}
# ====================================================================

# --- Pemeriksaan File ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(CASCADE_PATH):
    print("[ERROR] File model atau cascade tidak ditemukan. Pastikan file ada di folder yang benar.")
    exit()

# --- Muat Model dan Inisialisasi ---
print("[INFO] Memuat model-model...")
recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read(MODEL_PATH)
face_detector = cv2.CascadeClassifier(CASCADE_PATH)

print("[INFO] Membuka kamera...")
kamera = cv2.VideoCapture(KAMERA_INDEX)
if not kamera.isOpened():
    print(f"[ERROR] Gagal mengakses kamera dengan index {KAMERA_INDEX}.")
    exit()

print("[INFO] Sistem siap. Jendela kamera akan terbuka. Tekan 'q' untuk keluar.")
print("[INFO] Catatan: Deteksi emosi pertama kali mungkin butuh waktu lebih lama untuk memuat model.")

# --- Loop Utama ---
while True:
    ret, frame = kamera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Pengenalan Wajah (menggunakan model LBPH kita)
        id_prediksi, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        nama = "Tidak Dikenal"
        if confidence < 75:
            nama = NAMA_PENGGUNA.get(id_prediksi, "Tidak Dikenal")

        cv2.putText(frame, nama, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Analisis Emosi (menggunakan deepface)
        try:
            # DeepFace membutuhkan frame berwarna (BGR)
            # enfore_detection=False karena kita sudah mendeteksi wajah
            analisis = DeepFace.analyze(frame[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False)

            # DeepFace mengembalikan list, kita ambil elemen pertama
            if isinstance(analisis, list) and len(analisis) > 0:
                top_emotion = analisis[0]['dominant_emotion']
                cv2.putText(frame, f"Emosi: {top_emotion}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        except Exception as e:
            # Kadang deepface gagal jika wajah terlalu kecil, jadi kita lewati saja
            pass

    cv2.imshow("Sistem Absensi Wajah & Deteksi Emosi (DeepFace)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Selesai ---
print("[INFO] Menutup sistem.")
kamera.release()
cv2.destroyAllWindows()