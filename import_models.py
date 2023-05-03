import bentoml
import diffusers

if __name__ == "__main__":
    stage1_signatures = {
        "__call__": {"batchable": False},
        "encode_prompt": {"batchable": False},
    }

    bentoml.diffusers.import_model(
        "IF-stage1:v1.0", "DeepFloyd/IF-I-XL-v1.0",
        signatures=stage1_signatures,
        variant="fp16",
        pipeline_class=diffusers.DiffusionPipeline,)
    #bentoml.diffusers.import_model("IF-stage2:v1.0", "DeepFloyd/IF-II-L-v1.0", variant="fp16", pipeline_class=diffusers.DiffusionPipeline)
    #bentoml.diffusers.import_model("sd-upscaler", "stabilityai/stable-diffusion-x4-upscaler")
