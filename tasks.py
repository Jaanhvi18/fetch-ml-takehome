import torch.nn as nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(
        self,
        transformer_name="distilbert-base-uncased",
        hidden_dim=768,
        num_classes_a=3,
        num_classes_b=2,
    ):
        super(MultiTaskModel, self).__init__()

        # Load a pretrained transformer as the shared encoder
        self.encoder = AutoModel.from_pretrained(transformer_name)

        # Task A: classifier head
        self.classifier_a = nn.Linear(hidden_dim, num_classes_a)

        # Task B: classifier head (e.g., 2-class sentiment analysis)
        self.classifier_b = nn.Linear(hidden_dim, num_classes_b)

    def forward(self, input_ids, attention_mask):
        # Get transformer outputs --> (last hidden states)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Using the [CLS] embedding (first token) as pooled output
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Predicting outputs for each task using their respective heads
        out_a = self.classifier_a(pooled_output)
        out_b = self.classifier_b(pooled_output)

        return out_a, out_b
