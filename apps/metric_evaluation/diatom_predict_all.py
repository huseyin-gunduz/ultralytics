import os
from ultralytics import YOLO
import cv2

# Modeli yükle
model = YOLO("../runs/detect/train8n_ESILU/weights/best.pt")  # Yolov8 modelini burada belirtin. Alternatif olarak, yolov8s.pt, yolov8m.pt vb. kullanılabilir.

# Tahminlerin kaydedileceği klasör
output_dir = "C:/Users/Admin/Desktop/predictions"
os.makedirs(output_dir, exist_ok=True)

# Resimlerin bulunduğu klasör
image_dir = "D:/datasets/diatom/YoloDataset_5_800x600_augmented_test_10_val_10_train_80_raw/images/all"
i = 0
# Tüm resimleri tahmin et
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)

    # Görüntünün tahminini yap
    results = model.predict(source=image_path, save=True, save_txt=True, project=output_dir, name=image_name)

    print(f"{image_name} için tahmin tamamlandı.")
    i = i + 1
    #if i > 10:
    #    break
