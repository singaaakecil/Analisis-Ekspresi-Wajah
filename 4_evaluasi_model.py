import cv2
import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

print("[INFO] Memulai proses evaluasi model...")

# --- Konfigurasi Awal ---
folder_uji = 'data_wajah/uji'
model_path = 'model/model_latih.xml'

# --- Muat Model yang Sudah Dilatih ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
print(f"[INFO] Model dari '{model_path}' berhasil dimuat.")

# --- Fungsi untuk Evaluasi ---
def evaluate_model(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    
    true_labels = []
    predicted_labels = []

    print(f"[INFO] Mengevaluasi {len(image_paths)} gambar uji...")

    for image_path in image_paths:
        try:
            # Buka gambar uji dan konversi ke grayscale
            pil_image = Image.open(image_path).convert('L')
            image_np = np.array(pil_image, 'uint8')
            
            # Dapatkan ID sebenarnya dari nama file
            id_aktual = int(os.path.split(image_path)[-1].split(".")[1])
            
            # Lakukan prediksi menggunakan model
            id_prediksi, confidence = recognizer.predict(image_np)
            
            # Tambahkan hasil ke list masing-masing
            true_labels.append(id_aktual)
            predicted_labels.append(id_prediksi)
            
            print(f"  - File: {os.path.basename(image_path)} -> Prediksi: {id_prediksi}, Sebenarnya: {id_aktual}")

        except Exception as e:
            print(f"[PERINGATAN] Gagal memproses gambar {image_path}: {e}")
            
    return true_labels, predicted_labels

# --- Lakukan Evaluasi dan Hitung Metrik ---
true_labels, predicted_labels = evaluate_model(folder_uji)

if len(true_labels) > 0:
    print("\n--- Hasil Evaluasi Statistik ---")
    
    # Hitung Metrik 
    accuracy = accuracy_score(true_labels, predicted_labels)
    # 'weighted' digunakan untuk menghitung metrik untuk setiap label dan mencari rata-ratanya berdasarkan jumlah sampel
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    print(f"Akurasi   : {accuracy:.2%}")
    print(f"Presisi   : {precision:.2%}")
    print(f"Recall    : {recall:.2%}")
    print(f"F1-Score  : {f1:.2%}")
    
    # Menampilkan laporan klasifikasi yang lebih detail per kelas 
    print("\nLaporan Klasifikasi Rinci:")
    print(classification_report(true_labels, predicted_labels, zero_division=0))
    
    # Buat dan tampilkan Confusion Matrix 
    print("Membuat visualisasi Confusion Matrix...")
    cm = confusion_matrix(true_labels, predicted_labels)
    labels = sorted(list(set(true_labels))) # Dapatkan label unik untuk sumbu plot
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Label Prediksi')
    plt.ylabel('Label Sebenarnya')
    plt.title('Confusion Matrix Hasil Evaluasi Model')
    plt.show()

else:
    print("[ERROR] Tidak ada data yang berhasil dievaluasi. Periksa folder 'data_wajah/uji'.")