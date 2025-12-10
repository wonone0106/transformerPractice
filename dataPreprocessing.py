import torch
import sentencepiece as spm
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

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/spm.model")

train_data = Data("train")
valid_data = Data("valid")

def collate_fn(batch):
    srcs, tgts, labels = [], [], []
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()
    for src, tgt in batch:
        srcs.append(sp.encode(src))
        tgts.append([bos_id] + sp.encode(tgt))
        labels.append(sp.encode(tgt) + [eos_id])
    max_src_len = max(len(s) for s in srcs)
    max_tgt_len = max(len(t) for t in tgts)
    for i in range(len(srcs)):
        srcs[i] = srcs[i] + [pad_id] * (max_src_len - len(srcs[i]))
        tgts[i] = tgts[i] + [pad_id] * (max_tgt_len - len(tgts[i]))
        labels[i] = labels[i] + [pad_id] * (max_tgt_len - len(labels[i]))
    return {"src": torch.tensor(srcs), 
            "tgt": torch.tensor(tgts), 
            "label": torch.tensor(labels)}