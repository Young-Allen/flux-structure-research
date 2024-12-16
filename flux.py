import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("/mnt/d/studying-code/modelscope/FLUX.1-dev", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()

prompt1 = ["A photo of a tiger"]
prompt2 = ["A photo of a bunny"]

res = pipe(
    prompt1,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    height=512,
    width=512,
    generator=torch.Generator("cpu").manual_seed(0),
)

image = res.images[0]

# embedding_changes = res.embedding_changes
# image.save("image.png")