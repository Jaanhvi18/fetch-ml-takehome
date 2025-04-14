# Fetch ML Apprentice Take-Home

Hi! This is my take-home project for the Machine Learning Engineer Apprentice role at Fetch. The assignment focused on working with sentence transformers, setting up a multi-task learning model, and explaining how I’d approach training and transfer learning. I’ve also included a simple training loop using fake data to show how everything fits together.

---

##  What’s in here

| Task | Description |
|------|-------------|
| **Task 1** | Load a pretrained sentence transformer and get embeddings from sample sentences |
| **Task 2** | Build a multi-task learning model that does two NLP tasks at once |
| **Task 3** | Explain how I’d train the model — when to freeze layers, and how to use transfer learning |
| **Task 4** | Write a basic training loop using dummy inputs (just to simulate what training would look like) |

---

## File Overview

```
fetch_takehome/
├── main.py          # Runs Task 1 + a forward pass of Task 2
├── tasks.py         # MultiTaskModel definition (transformer + two heads)
├── train.py         # Simulated training loop with fake data
├── requirements.txt # Python packages I used
├── README.md        # This file!
```

---

##  How to Run

First, install the required packages (I used Python 3.10):

```bash
pip install -r requirements.txt
```

Then, to test things:

**Task 1 + Task 2 forward pass:**
```bash
python main.py
```

**Training loop with fake data:**
```bash
python train.py
```

---

## My Notes

### Task 1: Sentence Embeddings
I used the `sentence-transformers` library and the `all-MiniLM-L6-v2` model. It's a good balance of size and performance. I ran two test sentences and printed out their embedding shapes.

### Task 2: Multi-Task Learning
I built a simple model with one shared transformer (DistilBERT) and two separate heads:
- **Task A**: 3-class sentence classification
- **Task B**: 2-class sentiment analysis

### Task 3: Training & Transfer Learning
I wrote about different training setups:
- When to freeze the full model
- When to just freeze the transformer
- When it might make sense to freeze one task head
Also explained how I’d use a pretrained model like DistilBERT and fine-tune only certain layers depending on the data size and task.

### Task 4: Training Loop
For Task 4, I wrote a basic training loop in `train.py` to simulate multi-task training using fake data.

- I created a dummy dataset with random labels
- Each batch goes through the model to get predictions for both tasks
- I used `CrossEntropyLoss` for both tasks and added the two losses together
- The model updates its weights just like in real training

This shows how I’d structure training with real data, and keeps the code simple and easy to follow.

---

## What I Focused On

- Keeping the code clean and easy to follow
- Explaining my decisions as clearly as I could
- Making sure the project runs without needing extra setup

---

Let me know if you have any questions, and thanks again for the opportunity!
