from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

model_path = './saved_hpo_bert'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()
labels = {0: "O", 1: "B-HPO", 2:"I-HPO"}

def get_hpo_terms(text):
    """
    Extract HPO terms from text.
    
    Arg:
    - text (str): Input text for NER.
    
    Returns:
    - List of recognized HPO terms.
    """

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    predictions = torch.argmax(logits, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    pred_labels = predictions[0].tolist()
    
    recognized_entities = []
    current_entity = []
    
    for token, label_id in zip(tokens, pred_labels):
        label = labels.get(label_id, "O")

        if label == "B-HPO":
            if current_entity:
                recognized_entities.append(" ".join(current_entity))
            current_entity = [token]
        elif label == "I-HPO":
            if current_entity:
                current_entity.append(token)
        else:
            if current_entity:
                recognized_entities.append(" ".join(current_entity))
                current_entity = []

    if current_entity:
        recognized_entities.append(" ".join(current_entity))
    
    hpo_terms = [" ".join(e.replace(" ##", "").replace("##", "") for e in entity.split()) for entity in recognized_entities]
    
    return hpo_terms