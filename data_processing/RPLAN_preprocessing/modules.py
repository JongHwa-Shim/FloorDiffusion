from PIL import Image
import numpy as np
import torch
from typing import Union, List, Dict

def visualize_channel(channel: Union[torch.tensor, np.array], filter_value: List[int]=None):
    channel = np.squeeze(channel)
    if not isinstance(filter_value, List):
        filter_value = [filter_value]
    
    if len(channel.shape) == 3:

        if filter_value is not None:
            if not len(filter_value) == 3:
                raise ValueError("Length of filter_value in RGB image is 3.")
            channel_bool = np.where(channel==filter_value, 255, 0).astype(np.uint8)
            img = Image.fromarray(channel_bool)
            img.show()
        else:
            channel.show()

    elif len(channel.shape) == 2:
        if not len(filter_value) == 1:
            raise ValueError("Length of filter_value in gray image is 1.")
        
        channel_bool = np.where(channel==filter_value, 255, 0).astype(np.uint8)
        img = Image.fromarray(channel_bool)
        img.show()

    else:
        raise ValueError

from variables import *
def make_rgb_plan(plan_path):
    plan_pil = Image.open(plan_path)
    plan_np = np.array(plan_pil)[:,:,1]

    h = plan_np.shape[0]
    w = plan_np.shape[1]
    rgb_plan_np = np.zeros(shape=[h,w,3])

    for room_num in ROOM_ORDER:
        room_rgb = ROOM_TYPE_LABEL[room_num]
        plan_bool = np.where(plan_np==room_num, True, False)
        rgb_plan_np[plan_bool] = ROOM_TYPE_COLOR[room_rgb]

    return rgb_plan_np.astype(np.uint8)