from ultralytics import YOLO
import time
import os

if __name__ == '__main__':

    experiment_name = "train8n_SILU"

    start_time = time.perf_counter()

    model = YOLO(os.path.join("runs/detect", experiment_name, "weights/best.pt"))  # load the model
    metrics = model.val(data="diatom_test_10_10_80_modified.yaml",
                        project="runs/predict",
                        name=experiment_name)

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    path = os.path.join("runs", "predict", experiment_name, "test_time.txt")

    with open(path, "w") as f:
        f.write(f'{elapsed_time}')

