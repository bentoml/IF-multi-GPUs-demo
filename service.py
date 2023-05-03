from diffusers import DiffusionPipeline

import bentoml
from bentoml.io import JSON
from bentoml.io import Image

stage1_model = bentoml.diffusers.get("IF-stage1:v1.0")
stage1_runner = stage1_model.with_options(
    enable_model_cpu_offload=True,
).to_runner(name="stage1_runner")

stage2_model = bentoml.diffusers.get("IF-stage2:v1.0")
stage2_runner = stage2_model.with_options(
    enable_model_cpu_offload=True,
    load_pretrained_extra_kwargs={"text_encoder": None},
).to_runner(name="stage2_runner")

stage3_model = bentoml.diffusers.get("sd-upscaler:latest")
stage3_runner = stage3_model.with_options(
    pipeline_class=DiffusionPipeline,
    enable_model_cpu_offload=True,
).to_runner(name="stage3_runner")


svc = bentoml.Service(
    "DeepFloyd-IF", runners=[stage1_runner, stage2_runner, stage3_runner]
)


@svc.api(input=JSON(), output=Image())
def txt2img(input_data):
    prompt = input_data.get("prompt")
    negative_prompt = input_data.get("negative_prompt")
    prompt_embeds, negative_embeds = stage1_runner.encode_prompt.run(
        prompt=prompt, negative_prompt=negative_prompt
    )
    res_t = stage1_runner.run(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        output_type="pt",
    )
    images = res_t[0]
    res_t = stage2_runner.run(
        image=images,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        output_type="pt",
    )
    images = res_t[0]
    res_t = stage3_runner.run(
        prompt=prompt, negative_prompt=negative_prompt, image=images, noise_level=100
    )
    images = res_t[0]
    return images[0]
