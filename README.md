# Spam-Not-Spam-Classification-RNN-
Spam /Ham message or email detected by DL RNN


## 🧠 Model Architecture & Deep Learning Pipeline

The system is constructed as a **Stacked Simple Recurrent Neural Network (RNN)** optimized for sequence modeling and binary text classification. The architecture systematically scales temporal text representations, injects regularized dropout layers to prevent overfitting, and collapses sequence dimensions down to a single binary probability mapping.

### 📊 Layer-by-Layer Breakdown

| Layer (Type) | Output Shape | Parameter Formula | Structural Functional Description |
| :--- | :--- | :--- | :--- |
| **1. Embedding** | `(batch_size, MAX_LEN, EMBEDDING_DIM)` | `(VOCAB_SIZE + 1) × EMBEDDING_DIM` | Maps integer-encoded tokens to dense, continuous vector spaces, dynamically learning geometric word meanings. |
| **2. SimpleRNN 1** | `(batch_size, MAX_LEN, 64)` | `64 × (EMBEDDING_DIM + 64 + 1)` | Standard recurrent layer running sequentially over time steps. By setting `return_sequences=True`, it outputs the full hidden state sequence for every downstream token slot. |
| **3. Dropout (0.3)** | `(batch_size, MAX_LEN, 64)` | `0` | Regularization layer that randomly zeroes out 30% of temporal activations to mitigate layer-to-layer node co-dependency. |
| **4. SimpleRNN 2** | `(batch_size, 32)` | `32 × (64 + 32 + 1)` | Deeper abstraction recurrent layer extracting abstract patterns. By setting `return_sequences=False`, it drops sequence tracking to output exclusively the final single architectural state vector. |
| **5. Dropout (0.3)** | `(batch_size, 32)` | `0` | Final safety regularizer that drops 30% of the aggregated summary features right before classification to combat network overfitting. |
| **6. Dense (Sigmoid)** | `(batch_size, 1)` | `(32 × 1) + 1 = 33` | Fully connected projection layer squeezing the latent features into a strict final continuous probability scale ($0.0 \dots 1.0$). |

---

![RNN Architecture Flow](https://github.com/amitpoa/Spam-Not-Spam-Classification-RNN-/releases/download/v1.0/model.flow.png)

---
