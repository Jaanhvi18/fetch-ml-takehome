##  Task 3: Training Considerations + Transfer Learning

### Freezing Parts of the Model

Here are a few ways to train the model, and what I think about each:

#### 1. Freeze the entire model
- This means I’m not updating any weights — I’m just using the transformer to get sentence embeddings and training nothing new.
- It’s fast and works if I already know the transformer does a good job for my task.
- But it doesn’t let the model learn anything new from my data, which might hurt performance.

#### 2. Freeze just the transformer (and train the task-specific heads)
- This is helpful when I don’t have a lot of data.
- The transformer still gives good general sentence representations, and I only train the classification layers on top.
- This can prevent overfitting and save compute time.

#### 3. Freeze one task head and fine-tune the rest
- Let’s say Task A is working well and I don’t want to mess with it — I can freeze its head.
- Then I can focus on improving Task B without losing what I’ve learned for Task A.
- This is useful if I’ adding a new task later on or trying to boost one task more than the other.

---

### Transfer Learning Plan

Here’s how I’d approach transfer learning in this setup:

- **Pretrained Model**: I used `distilbert-base-uncased` because it’s lightweight and still powerful. Good for quick experiments and not too heavy on compute.
  
- **What I’d freeze**:
  - I’d start by freezing the first few transformer layers.
  - The early layers usually learn more general stuff (like grammar and structure), while the later ones learn more task-specific patterns.
  - I’d keep the top layers + heads trainable so the model can adapt to my tasks.

- **Why**:
  - Freezing everything is fast but too rigid.
  - Fine-tuning everything might overfit or take too long.
  - So freezing just part of the model is a good middle ground — it keeps general language understanding but still adapts to my task.

---

### Summary

- Freezing layers is a tradeoff between speed and flexibility.
- Transfer learning works best when I reuse a strong base (like DistilBERT) and fine-tune the parts that matter most.
- I’d pick my setup based on how much data I have and how different my tasks are from what the model was originally trained on.
