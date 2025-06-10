# features.py
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.eval()

def extract_bert_single(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        outputs = bert(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)
    return pooled

def extract_features_single(text):
    term_density = 0.33
    word_count = 21
    author_trust = 0.3
    citation = 5.0
    extra = torch.tensor([[term_density, word_count, author_trust, citation]])
    return extra
