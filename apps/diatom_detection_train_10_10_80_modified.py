import time
import os
from ultralytics import YOLO
from constants import Constants
import wandb

if __name__ == '__main__':

    root = Constants.ROOT_DIRECTORY

    activation_function_name = 'LearnableSwishRelu'

    experiment_lookup = {
        #"yolo8": ["yolov8.yaml", "D:/runs/pt/yolov8n.pt"],
        "yolo9": ["yolov9s.yaml", "D:/runs/pt/yolov9s.pt"],
        "yolo10": ["yolov10s.yaml", "D:/runs/pt/yolov10s.pt"],
        "yolo11": ["yolo11.yaml", "D:/runs/pt/yolo11n.pt"],
        "yolo12": ["yolo12.yaml", "D:/runs/pt/yolo12n.pt"],
    }

    for yolo_version, files in experiment_lookup.items():
        start_time = time.perf_counter()

        wandb.init(project="diatom-detection", name=f"{activation_function_name}_{yolo_version}")


        print(f"✅ YAML file: {files[0]} PT file: {files[1]}")

        model = YOLO(files[0]).load(files[1])  # load the model

        experiment_path = os.path.join(activation_function_name, yolo_version)

        results = model.train(data="diatom_train_10_10_80_modified.yaml",
                              epochs=50,
                              project=os.path.join(root, "train"),
                              name=experiment_path,
                              val=True,
                              #imgsz=640,
                              plots=True,
                              save=True
                              )

        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        path = os.path.join(root, "train", experiment_path, "training_time.txt")

        with open(path, "w") as f:
            f.write(f'{elapsed_time}')

        wandb.finish()
        #gc.collect()
        #torch.cuda.empty_cache()
