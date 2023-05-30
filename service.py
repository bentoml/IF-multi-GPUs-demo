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


# The following codes are for gradio web UI

import gradio as gr, re
from PIL import Image

def inference(prompt, negative_prompt=""):
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
    

css = """
.finetuned-diffusion-div div{
    display:inline-flex;
    align-items:center;
    gap:.8rem;
    font-size:1.75rem
}
.finetuned-diffusion-div div h1{
    font-weight:900;
    margin-bottom:7px
}
.finetuned-diffusion-div p{
    margin-bottom:10px;
    font-size:94%
}
a{
    text-decoration:underline
}
.tabs{
    margin-top:0;
    margin-bottom:0
}
#gallery{
    min-height:20rem
}
"""

with gr.Blocks(css) as demo:

    with gr.Row():

        with gr.Column(scale=55):
            # gallery = gr.Gallery(
            #     label="Generated images", show_label=False,
            # ).style(grid=[2], height="auto", container=True)
            gallery = gr.Image(
                label="Generated images", show_label=False, elem_id="gallery"
            ).style(height="auto", container=True,)

        with gr.Column(scale=45):
            generate = gr.Button(value="Generate", variant="secondary").style(container=False)
            with gr.Group():
              prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=3,placeholder="Enter prompt", lines=3).style(container=False)

              neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")

    inputs = [prompt, neg_prompt]
    outputs = [gallery]
    prompt.submit(inference, inputs=inputs, outputs=outputs, show_progress=True, queue=True)
    generate.click(inference, inputs=inputs, outputs=outputs, show_progress=True, queue=True)


svc.mount_asgi_app(demo.app, path="/ui/")
