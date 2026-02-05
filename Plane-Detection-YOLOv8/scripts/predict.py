import os
# --- KERNEL HATASI ENGELLEYİCİ (MUTLAKA EN ÜSTTE OLMALI) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. Modeli Yükle (best.pt ile kodun aynı klasörde olmalı)
model_yolu = "best.pt"
model = YOLO(model_yolu)

# 2. Resmi Belirle
resim_yolu = "kanit.jpg"

# 3. Tahmin Yap (Conf=0.5 yaparak net sonuç alalım)
print("🚀 Model uçağı arıyor...")
results = model.predict(source=resim_yolu, conf=0.7, save=True)

# 4. Sonucu Ekrana Bas
res_plot = results[0].plot() # YOLO'nun çizdiği kutulu resim

plt.figure(figsize=(12, 8))
# OpenCV BGR okur, Matplotlib RGB bekler; o yüzden renkleri çeviriyoruz
plt.imshow(cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB))
plt.title(f"Spyder Üzerinde Yerel Test Başarılı! (Skor: %{results[0].boxes.conf[0].item()*100:.0f})")
plt.axis('off')
plt.show()

print("✅ İşlem tamamlandı. Sonuç görseli ekranda!")