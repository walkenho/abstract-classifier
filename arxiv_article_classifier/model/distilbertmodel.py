"""Create a DistilBert model."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DistilBertModel


def seed_everything(seed):
    """Seed environment for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiLabelDataset(Dataset):
    """A dataset for multilabel classification."""

    def __init__(self, tokenizer, max_len, X, y=None):
        """Initialize MuliLabelDataset instance."""
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.X = X
        self.targets = y

    def __len__(self):
        """Return length of feature data."""
        return len(self.X)

    def __getitem__(self, index):
        """Return data."""
        text = str(self.X[index])

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=False,
        )
        out = {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
        }

        if self.targets is not None:
            out["targets"] = torch.tensor(self.targets[index], dtype=torch.float)

        return out


class DistilBertClass(torch.nn.Module):
    """A DistilBert model."""

    def __init__(self, n_classes):
        """Create DistilBert class instance."""
        super(DistilBertClass, self).__init__()

        self.n_classes = n_classes
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, n_classes),
        )

    def forward(self, input_ids, attention_mask):
        """Propagate model."""
        # Since we use DistilBert, no token_type_ids are needed
        hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return self.classifier(hidden_state[:, 0])
