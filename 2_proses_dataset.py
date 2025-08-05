import os
import shutil
import random

print("[INFO] Memulai proses pemisahan dataset...")

# --- Konfigurasi Awal ---
sumber_dataset = 'data_wajah'
folder_latih = 'data_wajah/latih'
folder_uji = 'data_wajah/uji'

# Proporsi pembagian dataset (80% latih, 20% uji)
proporsi_latih = 0.8

# --- Buat Folder Baru ---
# Membuat folder untuk data latih dan uji jika belum ada
os.makedirs(folder_latih, exist_ok=True)
os.makedirs(folder_uji, exist_ok=True)
print(f"[INFO] Folder '{folder_latih}' dan '{folder_uji}' telah disiapkan.")

# --- Pengelompokan File Berdasarkan ID Pengguna ---
# Dapatkan semua file gambar dari folder sumber
semua_file = [f for f in os.listdir(sumber_dataset) if f.endswith('.jpg')]

# Buat dictionary untuk mengelompokkan file berdasarkan ID
data_per_id = {}
for nama_file in semua_file:
    try:
        # Ekstrak ID dari nama file (format: Nama.ID.Nomor.jpg)
        id_pengguna = nama_file.split('.')[1]
        
        # Jika ID belum ada di dictionary, buat list baru
        if id_pengguna not in data_per_id:
            data_per_id[id_pengguna] = []
        
        # Tambahkan nama file ke list ID yang sesuai
        data_per_id[id_pengguna].append(nama_file)
    except IndexError:
        # Abaikan file yang tidak sesuai format penamaan
        print(f"[PERINGATAN] File '{nama_file}' tidak sesuai format dan diabaikan.")
        continue

# --- Proses Pembagian dan Penyalinan File ---
total_file_latih = 0
total_file_uji = 0

# Loop untuk setiap ID yang ditemukan
for id_pengguna, daftar_file in data_per_id.items():
    # Acak urutan file agar pembagiannya random
    random.shuffle(daftar_file)
    
    # Tentukan titik pembagian (split point)
    titik_pembagian = int(len(daftar_file) * proporsi_latih)
    
    # Bagi daftar file menjadi dua set: latih dan uji
    file_latih = daftar_file[:titik_pembagian]
    file_uji = daftar_file[titik_pembagian:]
    
    # Salin file latih ke folder 'latih'
    for file in file_latih:
        shutil.copy(os.path.join(sumber_dataset, file), os.path.join(folder_latih, file))
    
    # Salin file uji ke folder 'uji'
    for file in file_uji:
        shutil.copy(os.path.join(sumber_dataset, file), os.path.join(folder_uji, file))
        
    print(f"[INFO] ID {id_pengguna}: {len(file_latih)} gambar latih, {len(file_uji)} gambar uji.")
    total_file_latih += len(file_latih)
    total_file_uji += len(file_uji)

print("\n--- Proses Selesai ---")
print(f"Total data latih: {total_file_latih} file")
print(f"Total data uji: {total_file_uji} file")
print("Dataset telah berhasil dibagi ke dalam subfolder 'latih' dan 'uji'.")