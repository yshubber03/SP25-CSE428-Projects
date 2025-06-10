# main.py
from features import extract_features_single, extract_bert_single
from model import BinaryClassifier
import torch

# Define one malicious abstract sample
abstract = "DrugX has shown promising results in treating DiseaseY, supported by multiple clinical trials."
extra_feats = torch.tensor([[0.33, 21, 0.3, 5.0]])  # term_density, word_count, trust, citation

# 1. BERT-only setup
bert_embed = extract_bert_single(abstract)
model_bert_only = BinaryClassifier(bert_embed.shape[1])
output_baseline = model_bert_only(bert_embed)

# 2. BERT + features setup
input_full = torch.cat([bert_embed, extra_feats], dim=1)
model_bert_plus = BinaryClassifier(input_full.shape[1])
output_enhanced = model_bert_plus(input_full)

print("BERT-only model output (confidence):", output_baseline.item())
print("→ Prediction:", "Not Malicious" if output_baseline.item() > 0.5 else "Malicious")

print("BERT + features model output (confidence):", output_enhanced.item())
print("→ Prediction:", "Not Malicious" if output_enhanced.item() > 0.5 else "Malicious")

print("\nNote: We define 1 = not malicious, 0 = malicious. So lower values indicate more suspicious abstracts.")
