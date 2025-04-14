import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tasks import MultiTaskModel

# Fake dataset for testing the training loop
class DummyMultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, size=10):
        # Repeating a few example sentences
        self.sentences = [
            "I love this!", 
            "Bad experience.", 
            "Very useful.", 
            "Would not recommend."
        ] * (size // 4)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Tokenize the sentence
        inputs = self.tokenizer(
            self.sentences[idx],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=16
        )
        # Return tokenized inputs + random labels for both tasks
        return (
            inputs["input_ids"].squeeze(0),
            inputs["attention_mask"].squeeze(0),
            torch.randint(0, 3, (1,)).item(),  # 3-class classi for Task A
            torch.randint(0, 2, (1,)).item(),  # 2-class classi for Task B
        )

# Simulated training loop
def train_loop(model, dataloader, optimizer, loss_fn_a, loss_fn_b):
    model.train()

    for input_ids, attention_mask, labels_a, labels_b in dataloader:
        # Forward pass
        out_a, out_b = model(input_ids, attention_mask)

        # Compute losses for each task
        loss_a = loss_fn_a(out_a, labels_a)
        loss_b = loss_fn_b(out_b, labels_b)

        # Combine the losses (equal weight here)
        loss = loss_a + loss_b

        # Backpropagation step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss for monitoring
        print(f"Loss A: {loss_a.item():.4f}, Loss B: {loss_b.item():.4f}")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = MultiTaskModel()

    # Loading fake data
    dataset = DummyMultiTaskDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=2)

    # Set up optimizer and loss functions for both tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn_a = nn.CrossEntropyLoss()
    loss_fn_b = nn.CrossEntropyLoss()

    #  training begins
    train_loop(model, dataloader, optimizer, loss_fn_a, loss_fn_b)
