import os
from ultralytics import YOLO

# Modeli yükle
model = YOLO("runs/detect/train8n_ESILU/weights/best.pt")  # Yolov8 modelini burada belirtin. Alternatif olarak, yolov8s.pt, yolov8m.pt vb. kullanılabilir.

# Tahminlerin kaydedileceği klasör
output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

# Resimlerin bulunduğu klasör
image_dir = "D:/datasets/diatom/YoloDiatomDataset800x600-28/images/all"

# Her bir resim için işlem yapın
for image_name in os.listdir(image_dir):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, image_name)

        # Görüntünün tahminini yap
        results = model.predict(source=image_path)

        # Tahmin edilen kutuları al
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence score
        pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)  # Tahmin edilen sınıflar

        # Tahmin sonuçlarını txt dosyasına kaydetme
        output_file = os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}.txt')
        with open(output_file, 'w') as f:
            for i in range(len(pred_boxes)):
                box = pred_boxes[i]
                confidence = confidences[i]
                class_id = pred_classes[i]
                # [x1, y1, x2, y2, confidence, class]
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {confidence} {class_id}\n")

        print(f"{image_name} için tahmin sonuçları ve confidence değerleri kaydedildi.")
