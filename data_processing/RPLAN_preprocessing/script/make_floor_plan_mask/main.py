import sys
import os
sys.path.append(os.getcwd())

from PIL import Image
import numpy as np
import sys

from variables import *

def main(original_plan_dir, mask_data_dir):
    original_plan_name_list = os.listdir(original_plan_dir)
    for original_plan_name in original_plan_name_list:
        original_plan_path = os.path.join(original_plan_dir, original_plan_name)
        np_mask = make_mask(original_plan_path)
        mask_pil = Image.fromarray(np_mask.astype(np.uint8))
        mask_plan_path = os.path.join(mask_data_dir, original_plan_name)
        mask_pil.save(mask_plan_path)

import random
def make_mask(original_plan_path,room_rate: list = [0.2, 0.4], make_door: bool = True):
    np_original_plan = np.array(Image.open(original_plan_path))
    np_mask = np.zeros(shape=[np_original_plan.shape[0], np_original_plan.shape[1], 3])
    
    # Make silhouette of floor plan.
    bool_mask = np.where(np_original_plan[:,:,3]==0, True, False)
    np_mask[bool_mask] = [255, 255, 255]
    bool_mask = np.where(np_original_plan[:,:,0]==127, True, False)
    np_mask[bool_mask] = [0, 0, 0]
    bool_mask = np.where(np_original_plan[:,:,1]==15, True, False)
    np_mask[bool_mask] = [0, 0, 0]

    # Choose random rooms to be provided as conditions.
    np_room_num_plan = np_original_plan[:,:,2]
    room_num_list = list(np.unique(np_room_num_plan))
    room_num_list.remove(0)
    room_prob = random.uniform(room_rate[0], room_rate[1])
    room_num = int(len(room_num_list) * room_prob)
    choosed_room_list = random.sample(room_num_list, room_num)

    # Choosed rooms are provided as unmasked area.
    for choosed_room in choosed_room_list:
        bool_mask = np.where(np_original_plan[:, :, 1] == choosed_room, True, False)
        np_mask[bool_mask] = [255,255,255]
    
    # Choose door for unmask area
    if make_door:
        door_nums = {"entrance": 10, "front door": 15, "interior door": 17}
        # if 0.5 < random.uniform(0,1):
        #     bool_mask = np.where(np_original_plan[:, :, 1] == door_nums["entrance"], True, False)
        #     np_mask[bool_mask] = [255, 255, 255]
        if 0 < random.uniform(0,1):
            bool_mask = np.where(np_original_plan[:, :, 1] == door_nums["front door"], True, False)
            np_mask[bool_mask] = [255, 255, 255]
        if 0 < random.uniform(0,1):
            bool_mask = np.where(np_original_plan[:, :, 1] == door_nums["interior door"], True, False)
            np_mask[bool_mask] = [255, 255, 255]
    return np_mask

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    main(os.path.join(current_dir, "original_data"), (os.path.join(current_dir, "mask_data")))