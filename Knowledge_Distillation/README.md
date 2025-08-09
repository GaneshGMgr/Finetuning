# 📘 Knowledge Distillation — Teacher to Student Overview

> **Goal:** Train a smaller **student** model to learn from a larger **teacher** model by combining *soft* predictions (teacher outputs) with *hard* ground-truth labels. This allows the student to efficiently mimic the teacher’s behavior and generalize better with fewer parameters.

---

## 📚 References
- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/pdf/1503.02531)  
- [Knowledge Distillation: A Survey](https://arxiv.org/pdf/2006.05525)  
- [DistilBERT: smaller, faster, cheaper and lighter](https://arxiv.org/pdf/1910.01108)  
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/pdf/2305.07759)  

---

## What is Knowledge Distillation?

Knowledge distillation is a training technique where a smaller, compact student model learns to replicate the behavior of a larger, well-trained teacher model. Instead of training solely on hard labels (ground-truth), the student also learns from the teacher’s *soft* output probabilities, which contain richer information about class similarities (sometimes called “dark knowledge”).

This combination improves the student’s performance and generalization, making it smaller, faster, and more efficient while retaining much of the teacher’s reasoning ability.

---

## Key Concepts

- **Soft targets:** The teacher model’s output probabilities softened by a temperature parameter to reveal nuanced relationships between classes.  
- **Hard targets:** The true dataset labels used in traditional supervised learning.  
- **Loss function:** A weighted sum of soft target loss (typically KL divergence) and hard target loss (cross-entropy).

---

## Differences Between BERT and LLM Logits Handling

- **BERT-based models:** Provide logits for every token in the input sequence, predicting tokens simultaneously.  
- **Large Language Models (LLMs):** Use causal modeling to predict the next token based on previous tokens, so the last token's logits are omitted to align predictions correctly.

This difference impacts how logits are sliced and aligned with labels during training and distillation.

---

## Why Distillation Works

Soft targets carry more information than hard labels alone because they encode the teacher’s learned distribution over classes. By learning from these soft targets, the student can capture the teacher’s subtle decision boundaries and improve generalization on smaller architectures.

---

## Practical Notes

- Distillation can scale from small experiments with a few prompts to massive datasets with millions of examples.  
- For large-scale training, frameworks like Hugging Face’s `DistillationTrainer` or `accelerate` help manage multi-GPU/TPU setups.  
- Proper tuning of temperature, loss weighting, and optimizer settings is key to success.

---

This overview covers the foundational concepts and distinctions essential to understanding and implementing knowledge distillation effectively.