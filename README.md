# Spam-Not-Spam-Classification-RNN-
Spam /Ham message or detected by DL RNN


### 📊 Model Architecture Summary

| Layer Number | Layer Type | Output Shape | Parameters | Description |
| :--- | :--- | :--- | :--- | | :--- | | :--- |
| 1 | **Embedding** | `((batch_size, MAX_LEN, EMBEDDING_DIM)` | `((VOCAB_SIZE + 1) × EMBEDDING_DIM))' |
| 2 | **SimpleRNN 1** | `(None, 100, 64)` | `return_sequences=True`, Dropout: 0.2 |
| 3 | **Dropout 1** | `(None, 100, 64)` | Structural Regularization (30%) |
| 4 | **SimpleRNN 2** | `(None, 32)` | `return_sequences=False`, Dropout: 0.2 |
| 5 | **Dropout 2** | `(None, 32)` | Final feature regularization (30%) |
| 6 | **Dense (Output)**| `(None, 1)` | `Sigmoid` activation (Binary Probability) |
