# Proyek UAS: Sistem Pengenalan Wajah dan Analisis Emosi
## Fitur Utama
* [cite_start]**Pengumpulan Data**: Script untuk mengambil sampel gambar wajah melalui webcam untuk membangun dataset primer[cite: 43].
* [cite_start]**Deteksi Wajah**: Menggunakan metode Viola-Jones dengan Haar Cascade untuk mendeteksi lokasi wajah secara cepat dan akurat di setiap frame video[cite: 92].
* [cite_start]**Pengenalan Wajah**: Menggunakan algoritma **Local Binary Patterns Histograms (LBPH)** untuk melatih model dan mengenali identitas wajah yang terdeteksi[cite: 20, 76].
* **Analisis Emosi**: Menggunakan library **DeepFace** dengan backend TensorFlow untuk menganalisis dan mengklasifikasikan ekspresi wajah.
* [cite_start]**Evaluasi Model**: Script untuk mengukur performa model pengenalan wajah menggunakan metrik standar seperti Akurasi, Presisi, Recall, F1-Score, dan Confusion Matrix[cite: 95, 112, 113, 114, 115, 116].
* **Implementasi Real-Time**: Aplikasi utama yang menggabungkan semua fitur di atas dan menampilkannya secara visual.

## Setup dan Instalasi
Proyek ini memerlukan lingkungan Python yang spesifik untuk menjamin kompatibilitas library. Gunakan Conda untuk manajemen lingkungan.
1.  **Buat Lingkungan Conda**: Buat lingkungan baru bernama `uas_ai` dengan Python 3.10.
    ```bash
    conda create --name uas_ai python=3.10
    ```
2.  **Aktifkan Lingkungan**:
    ```bash
    conda activate uas_ai
    ```
3.  **Install Dependensi**: Install semua library yang dibutuhkan dengan `pip`.
    ```bash
    pip install opencv-contrib-python deepface tensorflow-macos tf-keras scikit-learn matplotlib seaborn
    ```
## Cara Penggunaan

Jalankan script berikut secara berurutan dari Terminal Anda. Pastikan Anda sudah mengaktifkan lingkungan `uas_ai` terlebih dahulu (`conda activate uas_ai`).

1.  **Kumpulkan Data Wajah (`1_kumpulkan_data.py`)**
    Jalankan script untuk setiap orang yang ingin Anda daftarkan. Ikuti instruksi di terminal untuk memasukkan ID (angka) dan Nama.
    ```bash
    python 1_kumpulkan_data.py
    ```

2.  **Bagi Dataset (`2_proses_dataset.py`)**
    Setelah data terkumpul di folder `data_wajah`, jalankan script ini untuk membaginya menjadi data latih dan uji.
    ```bash
    python 2_proses_dataset.py
    ```

3.  **Latih Model (`3_latih_model.py`)**
    Jalankan script ini untuk melatih model pengenalan wajah menggunakan data di folder `data_wajah/latih`. Model akan disimpan di `model/model_latih.xml`.
    ```bash
    python 3_latih_model.py
    ```

4.  **Evaluasi Model (`4_evaluasi_model.py`)**
    (Opsional) Jalankan script ini untuk melihat performa model Anda pada data uji.
    ```bash
    python 4_evaluasi_model.py
    ```

5.  **Jalankan Sistem Utama (`5_jalankan_sistem_alternatif.py`)**
    Ini adalah script final. Pastikan Anda telah menyesuaikan kamus `NAMA_PENGGUNA` di dalam script.
    ```bash
    python 5_jalankan_sistem_alternatif.py
    ```
    Tekan **'q'** untuk keluar dari aplikasi.
