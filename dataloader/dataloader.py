import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import gluonnlp as nlp

DATA_PATH = "../data/malicious_comment_dataset.csv"
mcc_df = pd.read_csv(DATA_PATH, encoding="utf-8")

from KoBERT.kobert.utils import get_tokenizer
from KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model


class MaliciousCommentDataset(Dataset):
    def __init__(self, dataset, max_length, pad, pair, label_cols):
        tokenizer = get_tokenizer()
        _ , vocab = get_pytorch_kobert_model()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

        transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=max_length, pad=pad, pair=pair)
        self.sentence = [transform([txt]) for txt in dataset.댓글]
        self.labels = dataset[label_cols].values

    def __getitem__(self, index):
        return (self.sentence[index] + (self.labels[index]))
    
    def __len__(self):
        return (len(self.labels))


if __name__ == "__main__":
    dataset = MaliciousCommentDataset()
    print(dataset)