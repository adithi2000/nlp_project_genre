import torch
import torch.nn as nn
from transformers import AutoModel
import joblib
from sklearn.feature_extraction.text import CountVectorizer

def create_vectorizer():
    vectorizer=joblib.load('vectorizer.pkl')
    return vectorizer

class topicGenreModel(nn.Module):
    def __init__(self, num_topics, num_genres, bow_dim, vocab_size, model_name="answerdotai/ModernBERT-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        bert_dim = self.backbone.config.hidden_size
        self.bow_weights = nn.Parameter(torch.tensor(0.1))
        
        self.context_encoder = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.bow_encoder = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 🔹 Latent space (topics)
        self.fc_mu = nn.Linear(512, num_topics)
        self.fc_logvar = nn.Linear(512, num_topics)

        # 🔹 Topic → word matrix
        self.beta = nn.Parameter(torch.randn(num_topics, vocab_size))

        # 🔹 Topic → genre
        self.classifier = nn.Sequential(
            nn.Linear(num_topics + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_genres)
        )

    # ---------- Encode ----------
    def encode(self, bow, embedding):
        h_ctx = torch.relu(self.context_encoder(embedding))
        alpha = torch.sigmoid(self.bow_weights)
        
        if bow is not None:
            h_bow = torch.relu(self.bow_encoder(bow))
            if self.training:
                # Dropout-like mask for BoW features
                mask = (torch.rand(h_bow.size(0), 1).to(h_bow.device) > 0.3).float()
                h_bow = h_bow * mask
            h = torch.cat([h_ctx, alpha * h_bow], dim=1)
        else:
            h_bow = torch.zeros_like(h_ctx)
            h = torch.cat([h_ctx, h_bow], dim=1)
            
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar, h_ctx

    # ---------- Reparameterization ----------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---------- Decode ----------
    def decode(self, topic_dist):
        logits = torch.matmul(topic_dist, self.beta)
        return torch.softmax(logits, dim=1)

    # ---------- Forward ----------
    def forward(self, input_ids, attention_mask, bow):
        # 🔹 BERT embedding (using CLS token)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        embedding = outputs.last_hidden_state[:, 0, :]

        # 🔹 Encode
        mu, logvar, h_ctx = self.encode(bow, embedding)

        # 🔹 Sample topics
        z = self.reparameterize(mu, logvar)
        topic_dist = torch.softmax(z, dim=1)

        # 🔹 Genre prediction
        # Detaching topic distribution to prevent it from dominating the backbone gradients
        topic_dist_detached = topic_dist.detach()
        combined = torch.cat([h_ctx, 0.5 * topic_dist_detached], dim=1)
        genre_logits = self.classifier(combined)

        # 🔹 Word reconstruction
        word_dist = self.decode(topic_dist)

        return topic_dist, genre_logits, word_dist, mu, logvar

def get_model(device):
    num_topics = 32
    num_genres = 13
    vectorizer = create_vectorizer()
    vocab_size = len(vectorizer.get_feature_names_out())
    bow_dim = vocab_size
    model = topicGenreModel(num_topics, num_genres, bow_dim=bow_dim, vocab_size=vocab_size)
    state_dict=torch.load('./checkpoint_v8_on_v7.pt',map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
