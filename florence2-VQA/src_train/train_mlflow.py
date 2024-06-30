import os
import sys
import json
import torch
import shutil
import argparse
import logging
import mlflow
import random
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from datetime import datetime
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, get_scheduler
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import textwrap

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the custom DocVQADataset class
class DocVQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<DocVQA>" + example['question']
        first_answer = example['answers'][0]
        image = example['image']
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, first_answer, image

# Overray image with text
def create_image_with_text(image, text, font_size=20, font_path=None):
    width, height = image.size
    width = (int)(width * 0.5)
    height = (int)(height * 0.5)
    
    image = image.resize((width, height))
    
    # Set the font size and path
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()
    
    # Wrap the text
    wrapper = textwrap.TextWrapper(width=40)
    text_lines = wrapper.wrap(text)
    
    # Create a new blank image with extra space for text
    total_height = image.height + font_size * len(text_lines) + 20
    new_image = Image.new("RGB", (image.width, total_height), "white")
    
    # Paste the original image on the new image
    new_image.paste(image, (0, 0))
    
    # Create a drawing context
    draw = ImageDraw.Draw(new_image)
    
    # Draw the text below the image
    text_y_position = image.height + 10
    for line in text_lines:
        draw.text((10, text_y_position), line, font=font, fill="black")
        text_y_position += font_size + 2
    
    return new_image
    

# Function to run the model on an example
def run_example(task_prompt, text_input, image):
    prompt = task_prompt + text_input

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer


def load_model(model_name_or_path="microsoft/Florence-2-base-ft", freeze_vision_encoder=True):
    global model
    global processor
    
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
    
    if freeze_vision_encoder:
        for param in model.vision_tower.parameters():
            param.is_trainable = False
        

def collate_fn(batch, processor):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers


def parse_args():
    # setup argparse
    parser = argparse.ArgumentParser()
    # curr_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # hyperparameters
    parser.add_argument("--model_name_or_path", default="microsoft/Florence-2-base-ft", type=str, help="Model name or path")
    parser.add_argument("--train_dir", default="../dataset", type=str, help="Input directory for training")
    parser.add_argument("--model_dir", default="./model", type=str, help="output directory for model")
    parser.add_argument("--epochs", default=1, type=int, help="number of epochs")
    parser.add_argument("--output_dir", default="./output_dir", type=str, help="directory to temporarily store when training a model")    
    parser.add_argument("--train_batch_size", default=10, type=int, help="training - mini batch size for each gpu/process")
    parser.add_argument("--eval_batch_size", default=10, type=int, help="evaluation - mini batch size for each gpu/process")
    parser.add_argument("--learning_rate", default=1e-06, type=float, help="learning rate")
    parser.add_argument("--logging_steps", default=5, type=int, help="logging steps")
    parser.add_argument("--save_steps", default=20, type=int, help="save steps")    
    parser.add_argument("--grad_accum_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", default="linear", type=str)
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument("--warmup_ratio", default=0.2, type=float, help="warmup ratio")

    # parse args
    args = parser.parse_args()

    # return args
    return args


def train_model(args, train_dataset, val_dataset):
    epochs = args.epochs
    save_steps = args.save_steps
    grad_accum_steps = args.grad_accum_steps
    
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    num_workers = 0

    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=partial(collate_fn, processor=processor), num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, collate_fn=partial(collate_fn, processor=processor), num_workers=num_workers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    saved_models = []
    model.train() 
    
    with mlflow.start_run() as run: 
    
        mlflow.log_params({
            "epochs": epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "seed": args.seed,
            "lr_scheduler_type": args.lr_scheduler_type,        
            "grad_accum_steps": grad_accum_steps, 
            "num_training_steps": num_training_steps,
            "num_warmup_steps": num_warmup_steps,
        })

        for epoch in range(epochs):     
            train_loss = 0.0
            optimizer.zero_grad()

            for step, (inputs, answers) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")):

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"] 
                labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                train_loss += loss.item()           

                if (step + 1) % grad_accum_steps == 0:
                    train_loss /= grad_accum_steps # compute gradient average  
                    learning_rate = lr_scheduler.get_last_lr()[0]
                    progress = (step+1)/len(train_loader)
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], Learning Rate: {learning_rate}, Loss: {train_loss}')
                    mlflow.log_metric("train_loss", train_loss)
                    mlflow.log_metric("learning_rate", learning_rate)
                    mlflow.log_metric("progress", progress)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    train_loss = 0.0

                if (step + 1) % save_steps == 0:
                    output_dir = f"./{args.output_dir}/steps_{step+1}"
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    processor.save_pretrained(output_dir)                
                    print(f'Model saved at step {step+1} of epoch {epoch+1}')
                    saved_models.append(output_dir)

                    # Log image
                    idx = random.randrange(len(val_dataset))
                    val_img = val_dataset[idx][-1]
                    result = run_example("DocVQA", 'What do you see in this image?', val_dataset[idx][-1])
                    val_img_result = create_image_with_text(val_img, json.dumps(result))
                    mlflow.log_image(val_img_result, key="DocVQA", step=step)

                    # Manage to save only the most recent 3 checkpoints
                    if len(saved_models) > 2:
                        old_model = saved_models.pop(0)
                        if os.path.exists(old_model):
                            shutil.rmtree(old_model)
                            print(f'Removed old model: {old_model}')

            # Validation phase
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for (inputs, answers) in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):

                    input_ids = inputs["input_ids"]
                    pixel_values = inputs["pixel_values"]
                    labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)

                    outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            
            mlflow.log_metric("avg_val_loss", avg_val_loss)
            print(f"Average Validation Loss: {avg_val_loss}")
            
        # Save model checkpoint
        model_dir = args.model_dir
        #os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        processor.save_pretrained(model_dir)
        
        dependencies_dir = "dependencies"
        shutil.copytree(dependencies_dir, model_dir, dirs_exist_ok=True)

        
def main(args):
    
    # Load model
    load_model(args.model_name_or_path)

    # Load datasets
    dataset = load_from_disk(args.train_dir)
    train_dataset = DocVQADataset(dataset['train'])
    val_dataset = DocVQADataset(dataset['validation'])

    # Train model
    train_model(args, train_dataset, val_dataset)
    
    
if __name__ == "__main__":
    #sys.argv = ['']
    args = parse_args()
    #args.model_name_or_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/poc-phi3/code/Users/model/Florence-2-base-ft"
    print(args)
    main(args)    