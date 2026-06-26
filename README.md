# Spam-Not-Spam-Classification-RNN-
Spam /Ham message or detected by DL RNN


### 📊 Model Architecture Summary

| Layer Number | Layer Type | Output Shape | Activation / Details |
| :--- | :--- | :--- | :--- |
| 1 | **Embedding** | `(None, 100, 50)` | Maps tokens to dense vectors |
| 2 | **SimpleRNN 1** | `(None, 100, 64)` | `return_sequences=True`, Dropout: 0.2 |
| 3 | **Dropout 1** | `(None, 100, 64)` | Structural Regularization (30%) |
| 4 | **SimpleRNN 2** | `(None, 32)` | `return_sequences=False`, Dropout: 0.2 |
| 5 | **Dropout 2** | `(None, 32)` | Final feature regularization (30%) |
| 6 | **Dense (Output)**| `(None, 1)` | `Sigmoid` activation (Binary Probability) |
