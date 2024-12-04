import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = './saved_llama_3.2_3B_ins'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
disclamer = "Disclamer:\nThe information provided is for educational purposes and should not replace professional medical advice. Individuals should consult healthcare professionals or local health authorities for personalized guidance."


def get_diagnosis(symptoms):

    messages = [
        {"role": "system", "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Diagnose the user based on the symptoms with 1 line explanation and 3 preventive measures."},
        {"role": "user", "content": f"Symptoms : {symptoms}"},
    ]

    tokenized_message = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", return_dict=True)
    response_token_ids = model.generate(tokenized_message['input_ids'].cuda(),
                                        attention_mask=tokenized_message['attention_mask'].cuda(),
                                        max_new_tokens=128, 
                                        pad_token_id = tokenizer.eos_token_id
                                        )
    generated_tokens =response_token_ids[:, len(tokenized_message['input_ids'][0]):]
    diagnosis = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return diagnosis+f"\n\n{disclamer}"