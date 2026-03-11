from diffusers import DiffusionPipeline
import torch

# CPU 环境：使用 float32，不使用 cuda
pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float32)
pipeline.to("cpu")
image = pipeline("An image of a squirrel in Picasso style").images[0]
image.save("output.png")