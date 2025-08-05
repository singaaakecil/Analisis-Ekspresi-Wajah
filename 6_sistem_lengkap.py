import cv2
import os
from deepface import DeepFace
import numpy as np

print("[INFO] Memulai sistem lengkap...")

# --- Konfigurasi Awal ---
# Model Face Recognition (LBPH)
MODEL_PATH = 'model/model_latih.xml'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
NAMA_PENGGUNA = {
    1: "Nama Anda",
    2: "Nama Teman",
}

# Model Object Detection (MobileNet-SSD)
PROTOTXT_PATH = 'model_objek/MobileNetSSD_deploy.prototxt.txt'
CAFFE_MODEL_PATH = 'model_objek/MobileNetSSD_deploy.caffemodel'
# Daftar kelas yang bisa dideteksi oleh MobileNet-SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
CONFIDENCE_THRESHOLD = 0.4 # Ambang batas kepercayaan untuk deteksi objek

KAMERA_INDEX = 1 # Ubah jika perlu

# --- Muat Semua Model ---
print("[INFO] Memuat model-model...")
# 1. Model Face Recognition
recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read(MODEL_PATH)
face_detector = cv2.CascadeClassifier(CASCADE_PATH)

# 2. Model Object Detection
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFE_MODEL_PATH)

# 3. Inisialisasi Detektor Emosi
print("[INFO] Menginisialisasi detektor emosi (DeepFace)...")
try:
    # Build model pertama kali untuk mempercepat loop
    DeepFace.build_model("Emotion")
    print("[INFO] Model emosi berhasil di-build.")
except Exception as e:
    print(f"[ERROR] Gagal inisialisasi DeepFace: {e}")
    exit()

# --- Buka Kamera ---
print("[INFO] Membuka kamera...")
kamera = cv2.VideoCapture(KAMERA_INDEX)
if not kamera.isOpened():
    print(f"[ERROR] Gagal mengakses kamera index {KAMERA_INDEX}.")
    exit()

print("[INFO] Sistem siap. Tekan 'q' untuk keluar.")

# --- Loop Utama ---
while True:
    ret, frame = kamera.read()
    if not ret:
        break

    (h_frame, w_frame) = frame.shape[:2]

    # --- A. Proses Deteksi Objek ---
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop hasil deteksi objek
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            # Abaikan deteksi 'person' agar tidak tumpang tindih dengan face recognition
            if CLASSES[idx] == "person":
                continue

            box = detections[0, 0, i, 3:7] * np.array([w_frame, h_frame, w_frame, h_frame])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence:.2%}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y_label = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # --- B. Proses Deteksi & Pengenalan Wajah ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id_prediksi, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        nama = "Tidak Dikenal"
        if confidence < 75:
            nama = NAMA_PENGGUNA.get(id_prediksi, "Tidak Dikenal")

        cv2.putText(frame, nama, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Analisis Emosi
        try:
            analisis = DeepFace.analyze(frame[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False)
            if isinstance(analisis, list) and len(analisis) > 0:
                top_emotion = analisis[0]['dominant_emotion']
                cv2.putText(frame, f"Emosi: {top_emotion}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        except:
            pass

    # Tampilkan frame hasil akhir
    cv2.imshow("Sistem AI Lengkap", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Selesai ---
print("[INFO] Menutup sistem.")
kamera.release()
cv2.destroyAllWindows()