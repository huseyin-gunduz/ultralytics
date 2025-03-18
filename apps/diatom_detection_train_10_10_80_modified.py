import time
import os
from ultralytics import YOLO
from constants import Constants
import gc
import torch

if __name__ == '__main__':

    root = Constants.ROOT_DIRECTORY

    activation_function_name = 'swish_relu'

    experiment_name_List = ["yolov8s", "yolov9s", "yolov10s", "yolo11s", "yolov12s"]

    for experiment_name in experiment_name_List:

        start_time = time.perf_counter()

        filename_yaml = experiment_name + ".yaml"
        filename_pt = experiment_name + ".pt"

        model = YOLO(filename_yaml).load(filename_pt)  # load the model

        results = model.train(data="diatom_train_10_10_80_modified.yaml",
                              epochs=25,
                              project=os.path.join(root, "detect"),
                              name=experiment_name + '_' + activation_function_name
                              )

        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        path = os.path.join(root, "detect", experiment_name + "_" + activation_function_name, "training_time.txt")

        with open(path, "w") as f:
            f.write(f'{elapsed_time}')

        gc.collect()
        torch.cuda.empty_cache()
