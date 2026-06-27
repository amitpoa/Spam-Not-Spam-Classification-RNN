# Spam-Not-Spam-Classification-RNN-
Spam /Ham message or email detected by DL RNN

Run the model in locally or colab use [Test Inferance](https://github.com/amitpoa/Spam-Not-Spam-Classification-RNN-/blob/main/Interactive_Inferance_(spam_rnn_classifier).ipynb)

Download [model](https://github.com/amitpoa/Spam-Not-Spam-Classification-RNN/releases/download/v1.0/simplernn_model_1.keras) and [tokenizer](https://github.com/amitpoa/Spam-Not-Spam-Classification-RNN/releases/download/v1.0/tokenizer.pickle) from Releases (Intial Model Asstes)


---

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
Basic Understanding of RNN Model Architecture

<p align="center">
  <img src="https://github.com/amitpoa/Spam-Not-Spam-Classification-RNN-/releases/download/v1.0/basic.rnn.flow.png" style="width: 350px; height: 600px; object-fit: contain;" alt=" Basic RNN Architecture ">
</p>

---

MY RNN Architecture (AI generated)

`EMBEDDING_DIM` = 128 

<p align="center">
  <img src="https://github.com/amitpoa/Spam-Not-Spam-Classification-RNN-/releases/download/v1.0/model.architecture.png" style="width: 550px; height: 600px; object-fit: contain;" alt=" RNN Architecture Diagram">
</p>

---

## Explainable AI (XAI) - Post Training

Deep Neural Networks: `LIME` can be used to interpret predictions from convolutional neural networks (CNNs) and recurrent neural networks (RNNs).

The purpose of using `LIME` for the sentimental analysis because it's concentrate into a specific area and fits a linear model predection. `LIME` works by taking a specific text message, creating slight variations of it (by randomly hiding words), observing how the model's prediction changes, and fitting a simple, interpretable model (like a linear regression) to determine exactly which words contributed most to a "Spam" or "Ham" prediction.

Why not using `SHAP`: the number of possible word combinations scales exponentially as $2^N$ which computing time huge, where `LIME`  doesn't care about every mathematical combination. It takes a quick, random sample of perturbed sentences directly around into text instance.

Ref: https://arxiv.org/pdf/2412.00800#page=67.16

