import os, torch, requests
import evaluate

from datasets import (
    Dataset,
    load_dataset,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from trl import (
    SFTTrainer,
    setup_chat_format,
)

model_id = "/data/user/bsindala/PhD/CS762-NaturalLanguageProcessing/disease-diagnosis-phenotype-identification/fine_tuned_models/Llama-3.2-3B-Instruct-Rare-Disease_v3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

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

test_texts, test_annotations = read_files("/data/user/bsindala/PhD/CS762-NaturalLanguageProcessing/disease-diagnosis-phenotype-identification/dataset/RareDis-v1/test")

# PreProcess
def preprocessing(texts, annotations):
    processed_data = []
    for text, ann in zip(texts, annotations):
        processed_data.append({"text": text, "annotations": ann})
    return processed_data

test_data = preprocessing(test_texts, test_annotations)

# Convert to dictionary
def dictionary_converter(data):
    dict_data = {"text": [], "annotations": []}
    for item in data:
        dict_data["text"].append(item["text"])
        dict_data["annotations"].append(item["annotations"])
    return dict_data

test_dictionary = dictionary_converter(test_data)

# Dataset Convertion
test_dataset = Dataset.from_dict(test_dictionary)

print(test_dataset)

# Tokenization
# Use the End-of-Sequence Token as the Padding Token
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_test = tokenized_test.remove_columns(["text"])

# Accuracy
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

# Define the compute_metrics function
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    precision = precision_metric.compute(predictions=preds, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=preds, references=labels, average="weighted")
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

# Hyperparameter
training_args = TrainingArguments(
    output_dir="./results",
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
    remove_unused_columns=True,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    eval_dataset=test_dataset,
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    data_collator=data_collator,
    args=training_args,
    packing=False,
    compute_metrics=compute_metrics,
)

# Evaluate the model
results = trainer.evaluate()

# Print the results
print(f"Results: {results}")