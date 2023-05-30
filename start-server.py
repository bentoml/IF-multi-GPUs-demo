import argparse
import bentoml
from service import svc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-gpu", type=int, dest="gpu1", default=0,
                        help="stage1 gpu index (default to 0)")
    parser.add_argument("--stage2-gpu", type=int, dest="gpu2", default=0,
                        help="stage2 gpu index (default to 0)")
    parser.add_argument("--stage3-gpu", type=int, dest="gpu3", default=0,
                        help="stage3 gpu index (default to 0)")
    parser.add_argument("--bentoml-api-workers", type=int, dest="api_workers", default=1,
                        help="BentoML api server worker number (default to 1)")
    parser.add_argument("--bentoml-server-port", type=int, dest="bentoml_port", default=3000,
                        help="BentoML server port (default to 3000)")
    parser.add_argument("--bentoml-server-host", type=str, dest="bentoml_host", default="0.0.0.0",
                        help="BentoML server host (default to 0.0.0.0)")
    parser.add_argument("--gradio-server-port", type=int, dest="gradio_port", default=7860,
                        help="gradio web UI server port (default to 7860)")
    parser.add_argument("--gradio-server-name", type=str, dest="gradio_host", default="0.0.0.0",
                        help="gradio web UI server host (default to 0.0.0.0)")
    parser.add_argument("--gradio-share", type=bool, default=False,
                        help="share your gradio app (default to False)")

    args = parser.parse_args()

    env_vars = {
        "BENTOML_CONFIG_OPTIONS": f"""
            runners.stage1_runner.resources."nvidia.com/gpu"[0]={args.gpu1}
            runners.stage2_runner.resources."nvidia.com/gpu"[0]={args.gpu2}
            runners.stage3_runner.resources."nvidia.com/gpu"[0]={args.gpu3}
            api_server.workers={args.api_workers}
            api_server.timeout=1800
            runners.timeout=1500
        """
    }

    server = bentoml.HTTPServer(svc, port=args.bentoml_port, host=args.bentoml_host)
    server.timeout = 300
    import sys
    with server.start(env=env_vars, blocking=False,
                      stdout=sys.stdout, stderr=sys.stderr,):

        client = server.get_client()

        # gradio interface

        def inference(prompt, negative_prompt=""):
            img = client.txt2img(dict(prompt=prompt, negative_prompt=negative_prompt))
            return img

        import gradio as gr
        with gr.Blocks() as demo:

            with gr.Row():

                with gr.Column(scale=55):
                    gallery = gr.Image(
                        label="Generated images", show_label=False,
                    ).style(grid=[2], height="auto", container=True)

                with gr.Column(scale=45):
                    generate = gr.Button(
                        value="Generate",
                        variant="secondary"
                    ).style(container=True)

                    with gr.Group():
                      prompt = gr.Textbox(
                          label="Prompt",
                          show_label=False,
                          max_lines=3,
                          placeholder="Enter prompt",
                          lines=3
                      ).style(container=False)

                      neg_prompt = gr.Textbox(
                          label="Negative prompt",
                          placeholder="What to exclude from the image"
                      )

            inputs = [prompt, neg_prompt]
            outputs = [gallery]
            prompt.submit(inference, inputs=inputs, outputs=outputs, show_progress=True)
            generate.click(inference, inputs=inputs, outputs=outputs, show_progress=True)

        demo.queue().launch(
            server_name=args.gradio_host,
            server_port=args.gradio_port, 
            share=args.gradio_share
        )
