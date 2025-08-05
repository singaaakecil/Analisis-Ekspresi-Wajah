import cv2
import os
import numpy as np
from PIL import Image

print("[INFO] Memulai proses training model...")

# --- Konfigurasi Awal ---
folder_latih = 'data_wajah/latih'
folder_model = 'model'

# Pastikan folder untuk menyimpan model sudah ada
os.makedirs(folder_model, exist_ok=True)

# --- Inisialisasi Model ---
# Menggunakan LBPH (Local Binary Patterns Histograms) Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Menggunakan Haar Cascade untuk mendeteksi wajah di dalam gambar latih
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# --- Fungsi untuk Mempersiapkan Data Latih ---
def get_images_and_labels(path):
    # Dapatkan semua path gambar di dalam folder latih
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    
    face_samples = []
    ids = []

    print(f"[INFO] Membaca {len(image_paths)} gambar latih...")
    
    for image_path in image_paths:
        try:
            # Buka gambar dan konversi ke grayscale menggunakan PIL
            pil_image = Image.open(image_path).convert('L')
            image_np = np.array(pil_image, 'uint8')
            
            # Ekstrak ID pengguna dari nama file
            # Format: Nama.ID.Nomor.jpg
            id_pengguna = int(os.path.split(image_path)[-1].split(".")[1])
            
            # Deteksi wajah dalam gambar
            faces = detector.detectMultiScale(image_np)
            
            for (x, y, w, h) in faces:
                # Tambahkan area wajah (ROI - Region of Interest) ke list sampel
                face_samples.append(image_np[y:y+h, x:x+w])
                # Tambahkan ID yang sesuai ke list ID
                ids.append(id_pengguna)
                
        except Exception as e:
            print(f"[PERINGATAN] Gagal memproses gambar {image_path}: {e}")

    return face_samples, ids


# --- Proses Training ---
faces, ids = get_images_and_labels(folder_latih)

# Lakukan training model dengan data wajah dan ID yang sudah dikumpulkan
# np.array(ids) diperlukan karena recognizer membutuhkan format array NumPy
recognizer.train(faces, np.array(ids))

# Simpan model yang sudah dilatih ke dalam file .xml
model_path = os.path.join(folder_model, 'model_latih.xml')
recognizer.write(model_path)

# --- Selesai ---
print(f"\n[INFO] Proses training selesai.")
print(f"[INFO] {len(np.unique(ids))} wajah berhasil ditraining.")
print(f"[INFO] Model disimpan di: {model_path}")