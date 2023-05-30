Run [IF by DeepFloyd Lab](https://github.com/deep-floyd/IF) across multiple GPUs

Preparations:

```
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
python import_models.py
```

Run the server with web UI powered by gradio:

```
# if you have a GPU with more than 40GB VRAM, you can run all model on same GPU
python start-server.py

# if you have 2 Tesla T4 with 15GB VRAM, you can assign stage1 model to the first GPU, and assign stage2 and stage3 models to the second GPU
python start-server.py --stage1-gpu=0 --stage2-gpu=1 --stage3-gpu=1

# if you have 1 Tesla T4 with 15GB VRAM and another 2 GPUs with smaller VRAM size, you can assign stage1 model to T4, and assign stage2 and stage3 models to the second and third GPU
python start-server.py --stage1-gpu=0 --stage2-gpu=1 --stage3-gpu=2
```

Then you can visit the web UI at <http://localhost:7860>. BentoML's api endpoint is also accessible at <http://localhost:3000>. To show all options that you can change (like server's port), just run `python start-server --help`
