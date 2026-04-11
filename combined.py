from model import get_model
from dataset import create_dataset
import torch
from inference import inspect_sample
from dataset import create_vectorizer
from dataset import mlb_transform
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=get_model(device)

def  do_inference(texts,threshold=0.2,true_labels=None):
    loaded_data=create_dataset(texts)
    vectorizer=create_vectorizer()
    vocab=vectorizer.get_feature_names_out()
    mlb=mlb_transform()
    label_names=mlb.classes_
    true_labels=mlb.transform(true_labels) if true_labels is not None else None
    num_topics=32
    bow_dim=len(vocab)
    results=inspect_sample(texts,model, loaded_data, vocab,label_names,device,threshold,vectorizer=vectorizer)
    return results

    
