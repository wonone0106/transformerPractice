import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import spacy

import pickle

from glob import glob
import pandas as pd

class Data:
    def __init__(self, mode="train"):
        data_paths = glob(f"data/{mode}/*.xlsx")
        self.data = []
        for path in data_paths:
            self.data.extend(self.get_sample(path))
            
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
    

train_data = Data("train")
valid_data = Data("valid")

ko_tokenizer = get_tokenizer("spacy", language="ko_core_news_sm")
en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
def ko_yield_tokens(data_iter):
    for src, _ in data_iter:
        token = ko_tokenizer(src)
        yield token
        
def en_yield_tokens(data_iter):
    for _, tgt in data_iter:
        token = en_tokenizer(tgt)
        yield token


ko_vocab = build_vocab_from_iterator(ko_yield_tokens(train_data), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
ko_vocab.set_default_index(ko_vocab["<unk>"])

en_vocab = build_vocab_from_iterator(en_yield_tokens(train_data), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
en_vocab.set_default_index(en_vocab["<unk>"])

def collate_fn(batch):
    srcs, tgts = [], []
    for src, tgt in batch:
        srcs.append([ko_vocab[token] for token in ko_tokenizer(src)])
        tgts.append([en_vocab["<bos>"]]+[en_vocab[token] for token in en_tokenizer(tgt)]+[en_vocab["<eos>"]])
    max_length_question = max(len(text) for text in srcs)
    max_length_answer = max(len(text) for text in tgts)
    padded_srcs = [text + [ko_vocab["<pad>"]] * (max_length_question - len(text)) for text in srcs]
    padded_tgts = [text + [en_vocab["<pad>"]] * (max_length_answer - len(text)) for text in tgts]
    return {"src":torch.tensor(padded_srcs),
            "tgt":torch.tensor(padded_tgts)}
    
train_dl = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
valid_dl = DataLoader(valid_data, batch_size=16, shuffle=False, collate_fn=collate_fn)

save_obj = {
    "train_dl": train_dl,
    "valid_dl": valid_dl,
    "en_vocab": en_vocab,
    "ko_vocab": ko_vocab
}

with open("preprocessed.pkl", "wb") as f:
    pickle.dump(save_obj, f)