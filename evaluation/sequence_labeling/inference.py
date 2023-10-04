import os
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from list_docs_dataset import get_dataloader


class InferenceModel:
    def __init__(self, model_path, tokenizer_path=None, device=0):
        self.device = device
        if tokenizer_path is None:
            tokenizer_path = model_path

        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(device).eval()
        print("Model loaded")

    def forward(self, texts, batch_size=8, seq_length=512, loss_per_pos=False):
        dataset = get_dataloader(
            texts, self.tokenizer, batch_size, max_length=seq_length
        )
        return self.forward_pretokenized(dataset, loss_per_pos)

    def forward_pretokenized(self, dataset, loss_per_pos=False, limit=0):
        losses = []
        langs = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataset, total=limit or len(dataset))):
                if isinstance(batch, list) and len(batch) == 2:
                    batch_langs, batch = batch
                else:
                    batch_langs = []
                if limit and i >= limit:
                    break
                if self.device is not None:
                    for k, v in batch.items():
                        batch[k] = v.view(-1, v.size(-1)).cuda(self.device)
                sample_losses = self.forward_single_batch(batch, loss_per_pos)
                sample_losses = sample_losses.detach().tolist()
                if isinstance(sample_losses, float):
                    losses.append(sample_losses)
                else:
                    losses.extend(sample_losses)
                langs += batch_langs
        return losses, langs if len(langs) else losses

    def forward_single_batch(self, inputs, loss_per_pos=False):
        output = self.model(**inputs).logits

        input_ids = inputs["input_ids"]
        labels = inputs["input_ids"]

        labels = labels[:, 1:].contiguous()
        output = output[:, :-1].contiguous().float()

        losses = F.cross_entropy(output.transpose(1, 2), labels, reduction="none")

        pad_token = self.tokenizer.encoder["<pad>"]
        loss_mask = torch.ones(
            input_ids.size(), dtype=torch.float, device=input_ids.device
        )
        loss_mask[input_ids == pad_token] = 0.0
        loss_mask = loss_mask[:, 1:].contiguous()

        if loss_per_pos:
            return losses * loss_mask

        loss_mask = loss_mask.view(-1)
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        return loss


def load_mgpt(path, tokenizer_path=None, device=0):
    model = InferenceModel(path, tokenizer_path, device=device)
    return model
