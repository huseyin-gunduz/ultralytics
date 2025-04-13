from ultralytics import YOLO
import time
import os
from constants import Constants
from collections import defaultdict

def count_class_instances_and_images(label_dir, class_names):
    """
    Veri setindeki her bir sınıf için Instances ve Images bilgisini hesaplar.

    Args:
        label_dir (str): Etiket dosyalarının bulunduğu dizin.
        class_names (list): Sınıf isimlerinin listesi.

    Returns:
        dict: Her bir sınıf için Instances sayısı.
        dict: Her bir sınıf için Images sayısı.
    """
    class_instances = defaultdict(int)  # Her bir sınıfın toplam örnek sayısı
    class_images = defaultdict(set)     # Her bir sınıfın bulunduğu resimlerin kümesi

    # Etiket dosyalarını oku
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):  # Sadece .txt dosyalarını işle
            continue

        file_path = os.path.join(label_dir, label_file)
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Her bir resimdeki sınıfları say
        image_classes = set()  # Bu resimdeki sınıfları tutar
        for line in lines:
            line = line.strip()  # Satırın başındaki ve sonundaki boşlukları temizle
            if not line:  # Boş satırları atla
                continue

            parts = line.split()
            if len(parts) < 5:  # Geçersiz formatlı satırları atla (en az 5 değer olmalı)
                print(f"Uyarı: Geçersiz satır formatı '{line}' dosyası '{label_file}'")
                continue

            try:
                class_id = int(parts[0])  # Sınıf ID'sini al
                if class_id >= len(class_names) or class_id < 0:  # Geçersiz sınıf ID'si
                    print(f"Uyarı: Geçersiz sınıf ID'si {class_id} dosyası '{label_file}'")
                    continue

                class_name = class_names[class_id]
                class_instances[class_name] += 1  # Instances sayısını artır
                image_classes.add(class_name)     # Resimdeki sınıfları kaydet
            except (ValueError, IndexError) as e:
                print(f"Hata: Satır işlenirken hata oluştu '{line}' dosyası '{label_file}': {e}")
                continue

        # Her bir sınıfın bulunduğu resimleri güncelle
        for class_name in image_classes:
            class_images[class_name].add(label_file)

    # Images sayısını hesapla (kümenin boyutu)
    class_images_count = {class_name: len(images) for class_name, images in class_images.items()}

    return class_instances, class_images_count


if __name__ == '__main__':

    root = Constants.ROOT_DIRECTORY

    activation_function_name = 'SILU'

    class_instances, class_images = count_class_instances_and_images(
        "D:/datasets/diatom/YoloDataset_5_800x600_augmented_test_10_val_10_train_80_modified/labels/test",
        Constants.DIATOM_SPECIES)

    total_images = sum(class_images.values())
    total_instances = sum(class_instances.values())

    experiment_name_List = ["yolo8", "yolo9", "yolo10", "yolo11", "yolo12"]

    for experiment_name in experiment_name_List:

        print(experiment_name)

        start_time = time.perf_counter()

        model = YOLO(os.path.join(root, "train", activation_function_name, experiment_name, "weights/best.pt"))  # load the model
        metrics = model.val(data="diatom_test_10_10_80_modified.yaml",
                            project=os.path.join(root, "predict", activation_function_name),
                            name=experiment_name)

        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        # write elapsed time
        file_path_elapsed = os.path.join(root, "predict", activation_function_name, experiment_name, "test_time.txt")

        with open(file_path_elapsed, "w") as f:
            f.write(f'{elapsed_time}')

        # write metrics
        file_path_metrics = os.path.join(root, "predict", activation_function_name, experiment_name, "evaluation_results.txt")

        with open(file_path_metrics, 'w') as f:
            f.write("Class                                    Images     Instances  Box(P    R        mAP50    "
                    "mAP50-95)\n")

            # Genel metrikler
            f.write(f"{'all':<40} {total_images:<10} {total_instances:<10} {metrics.box.mp:<8.3} {metrics.box.mr:<8.3} {metrics.box.map:<8.3f} {metrics.box.map:<8.3f}\n")

            # Sınıf bazında metrikler
            for i, class_index in enumerate(metrics.box.ap_class_index):
                class_name = Constants.DIATOM_SPECIES[class_index]
                precision = metrics.box.p[i]
                recall = metrics.box.r[i]
                map50 = metrics.box.ap50[i]
                map = metrics.box.ap[i]

                f.write(f"{class_name:<40} {class_images[class_name]:<10} {class_instances[class_name]:<10} {precision:<8.3f} {recall:<8.3f} {map50:<8.3f} {map:<8.3f}\n")
