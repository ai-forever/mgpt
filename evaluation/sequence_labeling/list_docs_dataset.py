import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ListDocsDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, padding=True, strip=True):
        if strip:
            textstexts = [t.strip() for t in texts]

        tokenizer.pad_token_id = tokenizer.encoder["<pad>"]
        samples = []
        for text in texts:
            sample = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            samples.append(sample)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return ("unk", self.samples[item])


def get_dataloader(texts, tokenizer, batch_size, max_length):
    dataset = ListDocsDataset(texts, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False)
    return loader
