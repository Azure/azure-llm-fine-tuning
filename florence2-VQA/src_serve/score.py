import os
import re
import json
import torch
import base64
import logging

from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, BitsAndBytesConfig, get_scheduler
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_example_base64(task_prompt, text_input, base64_image, params):

    max_new_tokens = params["max_new_tokens"]
    num_beams = params["num_beams"]
    
    image = Image.open(BytesIO(base64.b64decode(base64_image)))
    prompt = task_prompt + text_input

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        num_beams=num_beams
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global processor
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_name_or_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "outputs"
    )
    
    model_kwargs = dict(
        trust_remote_code=True,
        revision="refs/pr/6",        
        device_map=device
    )
    
    processor_kwargs = dict(
        trust_remote_code=True,
        revision="refs/pr/6"
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name_or_path, **processor_kwargs)    

    logging.info("Loaded model.")

def run(json_data: str):
    logging.info("Request received")
    data = json.loads(json_data)
    task_prompt = data["task_prompt"]
    text_input = data["text_input"]
    base64_image = data["image_input"]
    params = data['params']

    generated_text = run_example_base64(task_prompt, text_input, base64_image, params)
    json_result = {"result": str(generated_text)}
    
    return json_result    
