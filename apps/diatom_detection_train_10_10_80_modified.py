import time
import os

from ultralytics import YOLO

if __name__ == '__main__':

    experiment_name = "train8s"

    start_time = time.perf_counter()

    model = YOLO("yolov8s.yaml").load("yolov8s.pt")  # load the model

    results = model.train(data="diatom_train_10_10_80_modified.yaml",
                          epochs=25,
                          project="D:/runs/detect",
                          name=experiment_name
                          )

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    path = os.path.join("D:/runs/detect", experiment_name, "training_time.txt")

    with open(path, "w") as f:
        f.write(f'{elapsed_time}')







