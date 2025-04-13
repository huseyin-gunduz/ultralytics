import os

import pandas as pd

from apps.constants import Constants

output_file = "all_results_summary.csv"
root = Constants.ROOT_DIRECTORY

activation_function_name_list = ["ELU", "GELU", "Mish", "ReLU", "SELU", "SiLU", "SwishRelu"]
experiment_name_list = ["yolo8", "yolo9", "yolo10", "yolo11", "yolo12"]

with open(output_file, "w", encoding="utf-8") as out_f:
    # Başlık satırı
    out_f.write("Activations, Yolo Versions, Precision, Recall,mAp50,mAp50-95\n")

    for activation_function_name in activation_function_name_list:
        for experiment_name in experiment_name_list:
            input_file = os.path.join(
                root, "predict", activation_function_name, experiment_name, "evaluation_results.txt"
            )

            if not os.path.exists(input_file):
                print(f"❌ Dosya bulunamadı: {input_file}")
                continue

            with open(input_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            #all_line = [line.strip() for line in lines if line.strip().startswith("all")]

            # Sütun genişlikleri
            colspecs = [(0, 40), (40, 51), (51, 63), (63, 71), (71, 80), (80, 89), (89, 100)]
            column_names = ['Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95']

            # Dosyayı oku, başlığı atla
            df = pd.read_fwf(input_file, colspecs=colspecs, names=column_names, skiprows=1)

            # 'all' satırını çıkar
            df = df[df['Class'].str.strip() != 'all']

            # Sayısal dönüşüm
            df[['Instances', 'P', 'R', 'mAP50', 'mAP50-95']] = df[['Instances', 'P', 'R', 'mAP50', 'mAP50-95']].apply(
                pd.to_numeric)

            # Toplam instance
            total = df['Instances'].sum()

            # Weighted hesaplama
            wp = (df['P'] * df['Instances']).sum() / total
            wr = (df['R'] * df['Instances']).sum() / total
            wm50 = (df['mAP50'] * df['Instances']).sum() / total
            wm95 = (df['mAP50-95'] * df['Instances']).sum() / total

            # Sonucu başlıkla birlikte yaz
            #print("Activations Yolo Version Precision    Recall    mAP50    mAP50-95")
            #print(f"{wp:.4f}     {wr:.4f}     {wm50:.4f}     {wm95:.4f}")

            row = f"{activation_function_name},{experiment_name},{wp},{wr},{wm50},{wm95}\n"
            out_f.write(row)
            print(f"✅ Eklendi: {activation_function_name}/{experiment_name}")

