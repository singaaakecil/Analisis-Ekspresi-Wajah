import cv2

print(f"Versi OpenCV yang terpasang: {cv2.__version__}")

try:
    # Coba buat instance dari LBPH Recognizer dengan nama fungsi yang baru
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    print(">>> SUKSES! Modul 'cv2.face.LBPHFaceRecognizer' berhasil diakses.")
except AttributeError:
    print(">>> GAGAL! Error 'AttributeError' masih terjadi.")
    print("Ini berarti ada masalah fundamental dengan instalasi OpenCV atau kompatibilitas versi.")
except Exception as e:
    print(f">>> GAGAL dengan error tak terduga: {e}")