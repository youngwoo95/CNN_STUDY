from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="bert-base-uncased"
)

text = "I went to the [MASK] to deposit money."

results = fill_mask(text)

for result in results:
    print(
        result["token_str"],
        result["score"]
    )