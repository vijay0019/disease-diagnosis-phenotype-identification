from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import gc

save_dir = '/home/jainv/Fall_2024/Projects/prevention_generation/saved_llama_3.2_3B_ins'

model_name = "meta-llama/Llama-3.2-3B-Instruct"
hf_token = json.load(open('hconfig.json', 'r'))['hf_token']

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
#     )

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', token=hf_token)
print(f"Model download complete. Saving to {save_dir}")
model.save_pretrained(f'{save_dir}/')
del model
gc.collect()

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.save_pretrained(f'{save_dir}/')
print(f"Tokenizer download complete. Saving to {save_dir}")
del tokenizer
gc.collect()