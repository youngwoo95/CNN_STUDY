'''
레이어별 변화 - 12개 층을 지나면서 언제부터 갈라지나

층을 지날수록 두 bank 가 점점 멀어지는 게 보이는데, 그게 바로 self-attention이 주변 단어(money, river)를 읽어들이는 과정입니다.
'''
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "bert-base-uncased"
TARGET_WORD = "bank"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

texts = [
    "I went to the bank to deposit money.",
    "The bank approved my loan application.",
    "I sat on the bank of the river.",
    "Trees grow along the muddy bank.",
]

labels = ["금융1", "금융2", "강둑1", "강둑2"]

def analyze(text, word):
    '''
    한 문장에서 target 단어의 정적 임베딩 / 문맥 임베딩 / 레이어별 임베딩을 뽑는다.
    '''
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    if word not in tokens:
        raise ValueError(f"'{word}' 토큰 없음. 토큰: {tokens}")
    idx = tokens.index(word)

    with torch.no_grad():
        # 순수 정적 임베딩 : 위치/문맥 전혀 안 섞인 사전 조회 결과
        static = model.embeddings.word_embeddings(input_ids)[0, idx]

        # output_hidden_states=True -> 13개 텐서 (임베딩층 + 12개 블록)
        out = model(**inputs, output_hidden_states=True)
        per_layer = torch.stack([h[0, idx] for h in out.hidden_states]) # (13, 768)
        contextual = out.last_hidden_state[0, idx]

    return {
        "token" : tokens,
        "idx" : idx,
        "static" : static,
        "contextual" : contextual,
        "per_layer" : per_layer
    }

def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def print_matrix(title, vectors, labels):
    print(f"\n{title}")
    print("         " + "".join(f"{1:>8}" for l in labels))
    for i, li in enumerate(labels):
        row = "".join(f"{cos(vectors[i], vectors[j]):>8.3f}" for j in range(len(labels)))
        print(f"{li:>8}{row}")

if __name__ == "__main__":
    results = [analyze(t, TARGET_WORD) for t in texts]

    print("==== 토큰화 결과 ====")
    for label, text, r in zip(labels, texts, results):
        print(f"[{label}] {text}")
        print(f"        {r['token']}")
        print(f"        '{TARGET_WORD}' 위치: {r['idx']}")

    statics = [r["static"] for r in results]
    contextuals = [r["contextual"] for r in results]

    print_matrix("=== 정적 임베딩 유사도 (사전 조회 결과) ===", statics, labels)
    print("  -> 전부 1.000. 문맥과 무관하게 같은 벡터라는 뜻.")

    print_matrix("=== 문맥 임베딩 유사도 (12층 통과 후) ===", contextuals, labels)
    print("  -> 같은 뜻끼리는 높고, 다른 뜻끼리는 낮아야 함.")

    print("\n=== 레이어별 유사도 변화 ===")
    print("  층  | 금융1-금융2 | 금융1-강둑1 | 격차")
    print("  ----+-------------+-------------+------")
    same_pair = (0, 1)  # 금융1 vs 금융2
    diff_pair = (0, 2)  # 금융1 vs 강둑1
    for layer in range(results[0]["per_layer"].shape[0]):
        s = cos(results[same_pair[0]]["per_layer"][layer],
                results[same_pair[1]]["per_layer"][layer])
        d = cos(results[diff_pair[0]]["per_layer"][layer],
                results[diff_pair[1]]["per_layer"][layer])
        name = "emb" if layer == 0 else f"{layer:>3}"
        print(f"  {name:>3} |    {s:.3f}    |    {d:.3f}    | {s - d:+.3f}")

    print("\n  emb층에서는 격차가 거의 0 (아직 문맥이 안 섞임).")
    print("  층이 올라갈수록 격차가 벌어지면, attention이 주변 단어를 읽은 것.")