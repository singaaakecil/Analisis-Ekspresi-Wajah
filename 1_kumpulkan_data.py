import cv2
import os

# --- Konfigurasi Awal ---
# Pastikan folder untuk menyimpan dataset sudah ada
nama_folder_dataset = "data_wajah"
if not os.path.exists(nama_folder_dataset):
    os.makedirs(nama_folder_dataset)
    print(f"Folder '{nama_folder_dataset}' berhasil dibuat.")

# Inisialisasi kamera
nomor_kamera = 0
kamera = cv2.VideoCapture(nomor_kamera)
if not kamera.isOpened():
    print("Error: Kamera tidak dapat diakses.")
    exit()

# Muat model Haar Cascade untuk deteksi wajah
# Pastikan file haarcascade_frontalface_default.xml ada di folder yang sama
file_cascade = 'haarcascade_frontalface_default.xml'
detektor_wajah = cv2.CascadeClassifier(file_cascade)

# --- Pengambilan Data ---
# Minta input dari pengguna untuk ID dan nama
id_pengguna = input('Masukkan ID Pengguna (angka, contoh: 1): ')
nama_pengguna = input('Masukkan Nama Pengguna (contoh: Budi): ')
print("\n[INFO] Inisialisasi pengambilan wajah. Lihat ke kamera dan tunggu...")

jumlah_sampel = 0
batas_sampel = 500 # Ambil 500 sampel gambar

while True:
    # Baca frame dari kamera
    ret, frame = kamera.read()
    if not ret:
        print("[ERROR] Gagal mengambil frame dari kamera.")
        break

    # Ubah frame ke grayscale untuk deteksi wajah
    abu_abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah dalam frame
    daftar_wajah = detektor_wajah.detectMultiScale(abu_abu, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in daftar_wajah:
        # Gambar kotak di sekeliling wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        jumlah_sampel += 1
        
        # Simpan gambar wajah yang sudah dipotong (cropped) dan diubah ke grayscale
        nama_file = f"{nama_folder_dataset}/{nama_pengguna}.{id_pengguna}.{jumlah_sampel}.jpg"
        cv2.imwrite(nama_file, abu_abu[y:y + h, x:x + w])
        
        # Tampilkan informasi jumlah sampel yang sudah diambil
        cv2.putText(frame, f"Sampel: {jumlah_sampel}/{batas_sampel}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Tampilkan frame video
    cv2.imshow("Pengambilan Data Wajah", frame)

    # Kondisi untuk berhenti
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Pengambilan data dihentikan oleh pengguna.")
        break
    elif jumlah_sampel >= batas_sampel:
        print(f"[INFO] Pengambilan {batas_sampel} sampel untuk ID {id_pengguna} ({nama_pengguna}) selesai.")
        break

# --- Selesai ---
print("[INFO] Menutup program.")
kamera.release()
cv2.destroyAllWindows()