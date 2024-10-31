
1. pip uninstall diffusers (if exist)
2.  clone this repo
3. cd diffusers and pip install -e ".[torch]"
4. quickstart with this python code, try changing num_inference_steps and strength to get better results

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from diffusers import CogView3PlusTransformer2DModel, CogView3PlusImg2ImgPipeline
from diffusers.utils import load_image
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize

bfl_repo = "./"
dtype = torch.float32

transformer = CogView3PlusTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype)
quantize(transformer, weights=qfloat8)
freeze(transformer)

text_encoder = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder", torch_dtype=dtype)
quantize(text_encoder, weights=qfloat8)
freeze(text_encoder)

pipe = CogView3PlusImg2ImgPipeline.from_pretrained(bfl_repo, transformer=None, text_encoder=None, torch_dtype=dtype)
pipe.transformer = transformer
pipe.text_encoder = text_encoder
pipe.to("cuda")
#pipe.enable_model_cpu_offload()
init_image = load_image("R.jpg").resize((1024,1024))
prompt = "real, 8k, high-quality"
image = pipe(
    prompt=prompt,
    guidance_scale=0.0,
    image=init_image,
    output_type="pil",
    num_inference_steps=500,
     =0.1,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]

image.save("CogView3-i2i.png")
```
