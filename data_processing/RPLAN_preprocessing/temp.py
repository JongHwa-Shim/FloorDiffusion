from PIL import Image
import numpy as np
import torch
from modules import *

if __name__ == "__main__":
    sample_data_path = "./sample_floorplan.png"

    sample_image = Image.open(sample_data_path)
    sample_np = np.array(sample_image)
    x = torch.from_numpy(sample_np) # x.shape: [256,256,3]

    a = 1