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

    from diffusers import AutoencoderKL
    model_path = "runwayml/stable-diffusion-v1-5"
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to("cuda")
    from PIL import Image
    import PIL
    import numpy as np
    import torchvision.transforms.functional as f
    import torchvision.transforms as transforms
    def preprocess_image_repaintpipeline(image):
        deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"

        if isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, PIL.Image.Image):
            image = image

        if isinstance(image, PIL.Image.Image):
            w, h = image.size
            w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

            # image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
            # image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            # image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = 2.0 * image - 1.0
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)
        return image

    def preprocess_image(image: PIL.Image.Image):
        # reuse the function: 1. pil to tensor, 2. permute channel for pytorch
        image = preprocess_image_repaintpipeline(image)
        # resize the image resolution
        resized_image = f.resize(image, [512,512], interpolation=transforms.InterpolationMode.NEAREST)
        return resized_image
    for file_id in ["106", "235", "303", "361"]:
        file_path = f"./{file_id}.png"
        pil_original_img = Image.open(file_path)
        tensor_original_img = preprocess_image(pil_original_img).to("cuda")
        # latents = vae.encode(tensor_original_img).latent_dist.sample() * pipe.vae.config.scaling_factor
        # inpainted_image = vae.decode(latents / vae.config.scaling_factor).sample

        latents = vae.encode(tensor_original_img).latent_dist.sample() * pipe.vae.config.scaling_factor
        reconstructed_image = ((vae.decode(latents/ vae.config.scaling_factor).sample)+1)/2
        reconstructed_image = torch.clamp(reconstructed_image, min=0, max=1)
        import torchvision
        torchvision.utils.save_image(reconstructed_image, f"./{file_id}_vae_finetuned.png")

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
    