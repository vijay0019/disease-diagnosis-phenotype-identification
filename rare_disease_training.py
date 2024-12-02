#!/usr/bin/env python
# coding: utf-8

# # Libraries
# Cite: https://www.datacamp.com/tutorial/fine-tuning-llama-3-2

import os, torch, wandb, requests
#import bitsandbytes as bnb

from peft import PeftModel
from PIL import Image
from sklearn.metrics import classification_report
from transformers import ( 
    AutoModelForCausalLM,
    AutoTokenizer,
#     BitsAndBytesConfig,
    EarlyStoppingCallback,
    HfArgumentParser,
    IntervalStrategy,
    Trainer,   
    TrainingArguments, 
    pipeline,
    logging,
)

from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
    
from datasets import (
    Dataset,
    load_dataset,
)

from trl import (
#     SFTConfig,
    SFTTrainer,
    setup_chat_format,
)

from config import config

from huggingface_hub import login

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#os.environ["CUDA_LAUNCH_BLOCKING"] = 1

#user_secrets = UserSecretsClient()

hf_token = config["hf_token"]
login(token = hf_token)

wb_token = config["wandb"]
wandb.login(key = wb_token)
run = wandb.init(
    project="Fine-tune Llama 3.2 on RareDis-v1 Dataset",
    job_type="training",
    anonymous="allow"
)

# Meta Llama 3.2 3B Instruct
base_model = "/data/user/bsindala/PhD/CS762-NaturalLanguageProcessing/disease-classification-generation/models/Meta-Llama-3.2-3B-Instruct"
new_model = "/data/user/bsindala/PhD/CS762-NaturalLanguageProcessing/disease-classification-generation/fine_tuned_models/Llama-3.2-3B-Instruct-Rare-Disease_v3"

def read_files(data_dir):
    texts = []
    annotations = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), 'r') as f:
                texts.append(f.read())
        elif filename.endswith(".ann"):
            with open(os.path.join(data_dir, filename), 'r') as f:
                annotations.append(f.read())
    return texts, annotations

train_texts, train_annotations = read_files("/data/user/bsindala/PhD/CS762-NaturalLanguageProcessing/disease-classification-generation/dataset/RareDis-v1/train")
dev_texts, dev_annotations = read_files("/data/user/bsindala/PhD/CS762-NaturalLanguageProcessing/disease-classification-generation/dataset/RareDis-v1/dev")
test_texts, test_annotations = read_files("/data/user/bsindala/PhD/CS762-NaturalLanguageProcessing/disease-classification-generation/dataset/RareDis-v1/test")

# Debugging: Print the contents of train_texts and train_annotations
print("Train Texts:", train_texts)
print("Train Annotations:", train_annotations)

# Debugging: Print the contents of train_texts and train_annotations
print("Dev Texts:", dev_texts)
print("Dev Annotations:", dev_annotations)

# Debugging: Print the contents of train_texts and train_annotations
print("Test Texts:", test_texts)
print("Test Annotations:", test_annotations)

# PreProcess
def preprocessing(texts, annotations):
    processed_data = []
    for text, ann in zip(texts, annotations):
        processed_data.append({"text": text, "annotations": ann})
    return processed_data

train_data = preprocessing(train_texts, train_annotations)
dev_data = preprocessing(dev_texts, dev_annotations)
test_data = preprocessing(test_texts, test_annotations)

print("Processed Training Data:", train_data)
print("Processed Dev Data:", dev_data)
print("Processed Test Data:", test_data)

# Convert to dictionary
def dictionary_converter(data):
    dict_data = {"text": [], "annotations": []}
    for item in data:
        dict_data["text"].append(item["text"])
        dict_data["annotations"].append(item["annotations"])
    return dict_data

# Convert the processed data to the required format
train_dictionary = dictionary_converter(train_data)
dev_dictionary = dictionary_converter(dev_data)
test_dictionary = dictionary_converter(test_data)

# Dataset Convertion
train_dataset = Dataset.from_dict(train_dictionary)
dev_dataset = Dataset.from_dict(dev_dictionary)
test_dataset = Dataset.from_dict(test_dictionary)

print(train_dataset)
print(dev_dataset)
print(test_dataset)

# Set torch dtype and attention implementation
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
    attn_implementation = "eager" #"flash_attention_2"
else:
    torch_dtype = torch.float16
    attn_implementation = "eager"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
#     quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation,
)

# Enable grandient checkpoints to reduce memory usage
model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True
)

instruction = """
    You are a highly rated rare disease classification agent name Chimz. 
    Provide users all the answers regarding their question.
    """

def format_chat_template(row):
    row_json = [{"role": "system", "content": instruction},
                {"role": "user", "content": row["text"]},
                {"role": "assistant", "content": row["annotations"]}]
    
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

train_dataset = train_dataset.map(
    format_chat_template,
    num_proc=4,
)

test_dataset = test_dataset.map(
    format_chat_template,
    num_proc=4,
)

dev_dataset = dev_dataset.map(
    format_chat_template,
    num_proc=4,
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","v_proj"] #modules
)

# Set chat template to None before calling setup_chat_format
tokenizer.chat_template = None

model, tokenizer = setup_chat_format(model, tokenizer)
model = get_peft_model(model, peft_config)

# Tokenization
# Use the End-of-Sequence Token as the Padding Token
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_dev = dev_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Hyperparameter
training_args = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    optim="adamw_hf", #"paged_adamw_32bit",
    num_train_epochs=3,
    eval_strategy="steps", #IntervalStrategy.EPOCH,
    eval_steps=0.2,
    logging_steps=1, #500,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=1e-5, #2e-4,
    fp16=True,
    bf16=False,
    group_by_length=True,
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset, #tokenized_train,#
    eval_dataset=dev_dataset, #tokenized_dev,#
    peft_config=peft_config,
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

# Train
trainer.train()

# Merge adapters
# Merge LoRA adapters into the base model
merged_model = trainer.model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained(new_model)

# Save the fine-tuned model and tokenizer
#trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

wandb.finish()

# Test Fine Tuned Model
# messages = [{"role": "system", "content": instruction},
#     {"role": "user", "content": "I feel my leg is falling off, my heart won't stop racing, I am bleeding through my nose, and my back is killing me"}]

# prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
# inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

# outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)

# text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(text.split("assistant")[1])

# Evaluate
# eval_results = trainer.evaluate()
# print(eval_results)

# Test
# test_results = trainer.predict(tokenized_test)
# print(test_results)

# Validation
# predictions = test_results.predictions.argmax(-1)
# labels = test_results.label_ids
# report = classification_report(labels, predictions)
# print(report)