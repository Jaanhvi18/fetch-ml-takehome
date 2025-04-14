from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch
from tasks import MultiTaskModel

print(" Task 1: Sentence Embeddings")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["The cat sits on the mat.", "The dog runs in the park."]
embeddings = sentence_model.encode(sentences)

for i, embedding in enumerate(embeddings):
    print(f"Sentence: {sentences[i]}")
    print(f"Embedding shape: {embedding.shape}\n")


print("Task 2: Multi-Task Model Output ")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = MultiTaskModel()

# Tokenizing inputs for MultiTaskModel
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# Forward pass
with torch.no_grad():
    out_a, out_b = model(inputs["input_ids"], inputs["attention_mask"])

print("Task A logits (e.g., category classification):")
print(out_a)

print("\nTask B logits (e.g., sentiment classification):")
print(out_b)
