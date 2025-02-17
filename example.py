import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import gradio as gr

from huggingface_hub import CommitScheduler


JSON_DATASET_DIR = Path("json_dataset")
JSON_DATASET_DIR.mkdir(parents=True, exist_ok=True)

JSON_DATASET_PATH = JSON_DATASET_DIR / f"train-{uuid4()}.json"

scheduler = CommitScheduler(
    repo_id="example-space-to-dataset-json",
    repo_type="dataset",
    folder_path=JSON_DATASET_DIR,
    path_in_repo="data",
)


def greet(name: str) -> str:
    return "Hello " + name + "!"


def save_json(name: str, greetings: str) -> None:
    with scheduler.lock:
        with JSON_DATASET_PATH.open("a") as f:
            json.dump({"name": name, "greetings": greetings, "datetime": datetime.now().isoformat()}, f)
            f.write("\n")


with gr.Blocks() as demo:
    with gr.Row():
        greet_name = gr.Textbox(label="Name")
        greet_output = gr.Textbox(label="Greetings")
    greet_btn = gr.Button("Greet")
    greet_btn.click(fn=greet, inputs=greet_name, outputs=greet_output).success(
        fn=save_json,
        inputs=[greet_name, greet_output],
        outputs=None,
    )


demo.launch()