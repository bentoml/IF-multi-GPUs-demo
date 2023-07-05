import bentoml
from bentoml.exceptions import NotFound
import diffusers

if __name__ == "__main__":
    stage1_signatures = {
        "__call__": {"batchable": False},
        "encode_prompt": {"batchable": False},
    }

    stage1_model_tag = "IF-stage1:v1.0"
    try:
        bentoml.diffusers.get(stage1_model_tag)
    except NotFound:
        bentoml.diffusers.import_model(
            stage1_model_tag, "DeepFloyd/IF-I-XL-v1.0",
            signatures=stage1_signatures,
            variant="fp16",
            pipeline_class=diffusers.DiffusionPipeline,)

    stage2_model_tag = "IF-stage2:v1.0"
    try:
        bentoml.diffusers.get(stage2_model_tag)
    except NotFound:
        bentoml.diffusers.import_model(
            stage2_model_tag, "DeepFloyd/IF-II-L-v1.0",
            variant="fp16",
            pipeline_class=diffusers.DiffusionPipeline
        )

    upscaler_model_name = "sd-upscaler"
    try:
        bentoml.diffusers.get(upscaler_model_name)
    except NotFound:
        bentoml.diffusers.import_model(
            upscaler_model_name, "stabilityai/stable-diffusion-x4-upscaler"
        )
