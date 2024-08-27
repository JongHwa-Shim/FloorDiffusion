import os
from PIL import Image
import numpy as np
import torch

from modules import *
from variables import *
from script.make_floor_plan_mask.main import *
def main(dataset_dir, processed_dir):
    plan_name_list = os.listdir(dataset_dir)
    for plan_name in plan_name_list:
        plan_path = os.path.join(dataset_dir, plan_name)
        np_rgb_plan = make_rgb_plan(plan_path)

        pil_rgb_plan = Image.fromarray(np_rgb_plan)
        processed_plan_path = os.path.join(processed_dir, plan_name)
        pil_rgb_plan.save(processed_plan_path)

if __name__ == "__main__":
    main("./original_dataset/floorplan_dataset", "processed_dataset")