import torch
from diffusers import AutoPipelineForText2Image
from safetensors.torch import load_file

# 加载基础模型管道
flux_pipe = AutoPipelineForText2Image.from_pretrained("/mnt/d/studying-code/modelscope/FLUX.1-dev", torch_dtype=torch.bfloat16)
flux_pipe.enable_sequential_cpu_offload()

prompt = ["A photo of a bunny"]

image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    height=1024,
    width=1024,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]