# Credit Card Fraud Detection using Autoencoder

---

## Project Overview

This project implements an **unsupervised credit card fraud detection system** using a **deep learning Autoencoder** trained exclusively on normal transactions.  
The model identifies fraudulent transactions by measuring **reconstruction error**, flagging transactions that deviate significantly from learned normal behavior.

Fraud detection is a highly imbalanced and evolving problem where labeled fraud data is often scarce. This project demonstrates a scalable and realistic approach that does not rely on fraud labels during training.

---

## Business Problem

Credit card fraud causes significant financial losses and requires early detection. Traditional supervised models often struggle due to:

- Extreme class imbalance
- Limited labeled fraud examples
- Rapidly changing fraud patterns

This project addresses these challenges using **unsupervised anomaly detection**, making it suitable for real-world financial systems.

---

## Dataset

- **Source**: Kaggle – Credit Card Fraud Detection (ULB)
- **Transactions**: 284,807
- **Fraud Ratio**: ~0.17%
- **Features**:
  - PCA-transformed features (`V1`–`V28`)
  - `Time`
  - `Amount`
- **Target Label**:
  - `Class` (used **only for evaluation**, not for training)

---

## Technical Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

---

## Methodology

### Data Preparation
- Removed duplicate transactions
- Separated features and labels
- Trained the model using **normal transactions only**
- Applied feature scaling using `StandardScaler`
- Split data into training and test sets to prevent data leakage

### Model Architecture
- Fully Connected Autoencoder
- Encoder compresses input features into a latent representation
- Decoder reconstructs the original input
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam

---

## Anomaly Detection Strategy

- **Reconstruction Error** is used as the anomaly score
- A **percentile-based threshold** (99th percentile of normal reconstruction error) is selected
- Transactions exceeding the threshold are flagged as fraudulent

---

## Model Evaluation

Evaluation focuses on metrics appropriate for highly imbalanced data:

- Confusion Matrix
- Precision, Recall, and F1-score
- ROC-AUC using reconstruction error as a continuous anomaly score

**Fraud Recall** is prioritized to minimize false negatives, as missing fraudulent transactions is more costly than false positives.

---

## Results

- Fraud Recall: ~77%
- Clear separation between normal and fraudulent reconstruction errors
- Strong ROC-AUC indicating effective anomaly discrimination
- Acceptable trade-off between detection rate and false positives

---

## Key Insights

- Autoencoders are highly effective for fraud detection in imbalanced datasets
- Training exclusively on normal data enables detection of unseen fraud patterns
- Reconstruction error provides a meaningful and interpretable anomaly signal
- Recall-oriented evaluation is critical in financial risk detection systems

---

## Conclusion

This project demonstrates a practical, scalable, and realistic fraud detection solution using unsupervised deep learning.  
By avoiding reliance on labeled fraud data during training, the model remains adaptable to evolving fraud patterns while maintaining strong detection performance.

---

## Future Work

- Experiment with different latent space sizes
- Apply LSTM Autoencoders to capture temporal patterns
- Compare with other anomaly detection methods (Isolation Forest, LOF)
- Implement adaptive thresholding strategies
- Deploy the model for real-time fraud monitoring
