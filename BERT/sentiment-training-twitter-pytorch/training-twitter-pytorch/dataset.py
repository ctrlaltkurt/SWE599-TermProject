import os

os.environ["HF_HOME"] = os.path.abspath("../hf-data-models")

import pandas as pd
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset

np.random.seed(0)

def load_twitter_dataset(file_path, validation_ratio=0.2):
    # Load Twitter dataset
    data = pd.read_csv(file_path, header=None, encoding='latin1')
    data.columns = ["label", "id", "date", "query", "user", "text"]

    data["label"] = data["label"].map({0: 0, 4: 1})

    # Shuffle dataset
    data = data.sample(frac=1).reset_index(drop=True)

    # Train/val split
    val_n_samples = int(len(data) * validation_ratio)
    val_data = data[:val_n_samples]
    train_data = data[val_n_samples:]

    return train_data, val_data

class TwitterDataset(Dataset):
    def __init__(self, data, tokenizer_name='bert-base-uncased', max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.texts = data["text"]
        self.labels = data["label"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])

        # Tokenize the text
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': label
        }

def vis_dataset(train_data, val_data):
    print("N Samples [train]:", len(train_data))
    print("N Samples [val]:", len(val_data))
    print("Label Distribution [train]:", train_data["label"].value_counts().to_dict())
    print("Label Distribution [val]:", val_data["label"].value_counts().to_dict())

if __name__ == "__main__":
    # dataset_path = "/Users/onuraltinkurt/repos/SWE599/twitter_dataset.csv"
    dataset_path = "../datasets/twitter/training.1600000.processed.noemoticon.csv"
    train_data, val_data = load_twitter_dataset(dataset_path)

    vis_dataset(train_data, val_data)

    # Example usage
    train_dataset = TwitterDataset(train_data)
    sample = train_dataset[0]
    print("Input IDs:", sample['input_ids'])
    print("Attention Mask:", sample['attention_mask'])
    print("Label:", sample['label'])

