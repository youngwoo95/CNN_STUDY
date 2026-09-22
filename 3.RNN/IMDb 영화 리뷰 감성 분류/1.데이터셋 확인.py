'''
데이터 셋 불러오기.
'''
from datasets import load_dataset
dataset = load_dataset("stanfordnlp/imdb")

print(dataset)
print(dataset["train"][0])


'''
{
    "text": "...영화 리뷰...",
    "label": 0
}

0 = 부정 리뷰
1 = 긍정 리뷰
'''
