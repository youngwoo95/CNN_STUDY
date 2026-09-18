import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

texts = [
    "I went to the bank to deposit money.",
    "I sat on the bank of the river."
]

for text in texts:

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    bank_index = tokens.index("bank")

    bank_embedding = outputs.last_hidden_state[
        0,
        bank_index
    ]

    print()
    print(text)
    print(bank_embedding[:10])