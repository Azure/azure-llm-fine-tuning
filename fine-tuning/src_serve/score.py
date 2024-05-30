import os
import logging
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import LoraConfig, get_peft_model

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global tokenizer
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "outputs"
    )
    #model_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/poc-a100/code/Users/daekeunkim/artifact_downloads/phi-3-finetune2-2024-05-15/outputs/"

    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":0}, torch_dtype="auto", trust_remote_code=True)

    model.load_adapter(model_path)
    logging.info("Loaded model.")
    
def run(json_data: str):
    logging.info("Request received")
    data = json.loads(json_data)
    input_data= data["input_data"]
    params = data['params']
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = pipe(input_data, **params)
    generated_text = output[0]['generated_text']
    logging.info("Output Response: " + generated_text)
    #json_result = json.dumps({"result":str(generated_text)})
    json_result = {"result": str(generated_text)}
    return json_result    
