import numpy as np


def iou(box1, box2):
    """IoU (Intersection over Union) hesaplama"""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


def load_labels(file_path):
    """ .txt dosyasından etiketleri yükleme """
    boxes = []
    classes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    # YOLO formatında merkezi koordinat ve genişlik/yükseklik olduğundan dönüştür
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    boxes.append([x_min, y_min, x_max, y_max])
                    classes.append(class_id)
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {file_path}")
    return np.array(boxes), np.array(classes)


# Gerçek etiket dosyası ve tahmin edilen etiket dosyası
true_boxes, true_classes = load_labels('D:/datasets/diatom/YoloDiatomDataset800x600-28/labels/all/1303_V.txt')
pred_boxes, pred_classes = load_labels('predictions/1303_V.png/labels/1303_V.txt')

# IoU eşik değeri (örneğin 0.5)
iou_threshold = 0.5

# Gerçek pozitif, yanlış pozitif, yanlış negatif sayıları
true_positives = 0
false_positives = 0
false_negatives = 0

# Tahmin edilen ve gerçek kutular arasında IoU hesaplama
matched_gt = []  # Eşleşen gerçek kutular
for i, pred_box in enumerate(pred_boxes):
    iou_scores = np.array([iou(pred_box, true_box) for true_box in true_boxes])

    # En yüksek IoU değerini yazdır
    if iou_scores.size > 0:
        max_iou = np.max(iou_scores)
        max_iou_index = np.argmax(iou_scores)
        print(f"Tahmin kutusu {i + 1} için en yüksek IoU değeri: {max_iou:.2f}")
        print(f"Tahmin edilen sınıf ID: {pred_classes[i]}, Gerçek sınıf ID: {true_classes[max_iou_index]}")
    else:
        print(f"Tahmin kutusu {i + 1} için IoU hesaplanamadı.")

    # Doğru ve yanlış pozitif hesaplama
    if iou_scores.size > 0 and max_iou >= iou_threshold and max_iou_index not in matched_gt:
        # IoU yeterince yüksekse ve sınıflar aynıysa doğru pozitif
        if pred_classes[i] == true_classes[max_iou_index]:
            true_positives += 1
            matched_gt.append(max_iou_index)
        else:
            false_positives += 1
    else:
        false_positives += 1

# Yanlış negatifler, tahmin edilmeyen doğru kutular
false_negatives = len(true_boxes) - len(matched_gt)

# Precision, Recall ve F1-Score hesaplama
precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1_score:.2f}')
