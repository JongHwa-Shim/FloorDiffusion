import os
import torch
from diffusers import StableDiffusionPipeline

def dummy(images, **kwargs):
    return images, [False]

@ torch.no_grad()
def main(save_path, lora_path, weight_name, prompt, guidance_scale=7.5, seed=None, iter=12):
    os.makedirs(save_path, exist_ok=True) if not os.path.exists(save_path) else None

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5" ,torch_dtype=torch.float16)
    # pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base" ,torch_dtype=torch.float16) # * stable diffusion 2
    # pipe.unet.load_attn_procs(lora_path, weight_name=weight_name, cross_attention_kwargs={"scale": 1.0}) # load lora weight to unet only 이거 제대로 작동 안함
    pipe.load_lora_weights(lora_path, weight_name=weight_name) # TODO: 다양한 로라 로드방법 익히기 lora_scale 조정이라던가...
    pipe.safety_checker = dummy
    pipe.to("cuda")

    # torch.manual_seed(100) # TODO: 시드 고정 하는 방법 익히기
    if seed is not None:
        torch.manual_seed(seed)
    else:
        pass
    
    num_iter = iter
    for i in range(num_iter):


        image = pipe(prompt=prompt, num_inference_steps=999, guidance_scale=guidance_scale).images[0] # TODO: 이미지 보간 어떻게 하는지 체크하기

        save_img_name = f"{prompt}_{i}.png"
        image.save(os.path.join(save_path, save_img_name))

if __name__ == "__main__":
    for i in [7.5]:
        checkpoint = 500000
        dataset = "rplan_original"
        seed = None
        guidance_scale = i
        iter = 1000

        lora_path = f"./finetune/{dataset}/checkpoint-{checkpoint}"
        weight_name = "pytorch_lora_weights.safetensors"
        prompt="A floor plan of residential buildings. SJH_STYLE_FLOOR_PLAN"

        # save_path=f"./result/{dataset}/checkpoint={checkpoint},guidance={guidance_scale},seed={seed}"
        save_path = f"./result/{dataset}/final_result"

        main(save_path=save_path, lora_path=lora_path, weight_name =weight_name, prompt=prompt, guidance_scale=guidance_scale, seed=seed, iter=iter)
    