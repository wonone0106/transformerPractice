import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import spacy


from glob import glob
import pandas as pd

class Data:
    def __init__(self, mode="train"):
        data_paths = glob(f"data/{mode}/*.xlsx")
        self.data = []
        for path in data_paths:
            self.data.extend(self.get_sample(path))
        self.idx = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def get_sample(path):
        df = pd.read_excel(path)
        txt = df['원문']
        label = df['번역문']
        samples = [(src, tgt) for src, tgt in zip(txt, label)]
        return samples
    

data = Data("train")
kr_tokenizer = get_tokenizer("spacy", language="ko_core_news_sm")
en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
def yield_tokens(data_iter):
    for src, tgt in data_iter:
        yield kr_tokenizer(src), en_tokenizer(tgt)

for i in range(3):
    src, tgt = yield_tokens(data)
    breakpoint()

