import sys
import os
sys.path.append(os.getcwd())

from PIL import Image
import numpy as np
import sys

plan_dir = "./color_plan"
mask_dir = "./mask"
result_dir = "./masked_plan"
def main():
    for plan_name in os.listdir(plan_dir):
        plan_path = os.path.join(plan_dir, plan_name)
        plan_np = np.array(Image.open(plan_path).convert("RGB")) # load plan

        mask_path = os.path.join(mask_dir, plan_name) # make mask_path
        mask_pil = Image.open(mask_path).convert("L").resize(size=[plan_np.shape[1], plan_np.shape[0]],resample=Image.NEAREST)
        
        mask_np = np.expand_dims(np.array(mask_pil),axis=2)/255 # load mask # ! check gray image compatibility
        masked_plan_np = (plan_np * mask_np).astype(np.uint8) # elementwise mult with plan & mask
        masked_plan_pil = Image.fromarray(masked_plan_np)# save to result_dir
        masked_plan_pil.save(os.path.join(result_dir, plan_name))

if __name__ == "__main__":
    main()