from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import joblib
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
def create_vectorizer():
    vectorizer=joblib.load('vectorizer.pkl')
    return vectorizer
def mlb_transform():
    mlb=joblib.load('mlb.pkl')
    return mlb
class GenreDataset(Dataset):
    def __init__(self, text, tokenizer, vectorizer, max_len=128):
        self.texts = text if isinstance(text, list) else [text]
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.max_len = max_len

        # 🔹 Use transform (NOT fit)
        bow_matrix = self.vectorizer.transform(self.texts)
        self.bow = torch.tensor(bow_matrix.toarray(), dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            "input_ids": encoding['input_ids'].squeeze(0),
            "attention_mask": encoding['attention_mask'].squeeze(0),
            "bow": self.bow[idx],
        }

def create_dataset(texts, max_len=128):
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vectorizer = create_vectorizer()
    dataset = GenreDataset(texts,tokenizer, vectorizer, max_len)
    load_data=DataLoader(dataset, batch_size=1, shuffle=False)
    return load_data

