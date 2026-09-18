import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "I went to the bank to deposit money."

inputs = tokenizer(
    text,
    return_tensors="pt"
)

print("input_ids:")
print(inputs["input_ids"])

print("\ntokens:")
print(
    tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )
)

with torch.no_grad():
    outputs = model(**inputs)

print("\nBERT output shape:")
print(outputs.last_hidden_state.shape)