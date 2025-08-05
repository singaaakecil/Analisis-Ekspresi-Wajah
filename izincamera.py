import cv2

print("Mencoba mengakses kamera...")
# Coba akses kamera default (index 0)
kamera = cv2.VideoCapture(0) 

if not kamera.isOpened():
    print("Gagal membuka kamera. Pastikan tidak ada aplikasi lain yang menggunakannya.")
else:
    print("Berhasil terhubung ke kamera. Menampilkan satu frame.")
    ret, frame = kamera.read()
    if ret:
        cv2.imshow('Tes Kamera - Tekan tombol apapun untuk keluar', frame)
        cv2.waitKey(0) # Tunggu hingga ada tombol ditekan
    else:
        print("Gagal mengambil frame.")

# Selalu lepaskan kamera setelah selesai
kamera.release()
cv2.destroyAllWindows()
print("Program tes selesai.")