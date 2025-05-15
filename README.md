# 📰 Fake News Detection using BERT (Hugging Face Transformers)

This project focuses on detecting fake news articles using a state-of-the-art deep learning model—**BERT**—from the Hugging Face Transformers library. It involves text preprocessing, tokenization, fine-tuning a pretrained language model, and evaluating its performance.

---

## 📁 Project Structure

- `Fake News Detection.ipynb`: Jupyter notebook containing data loading, preprocessing, model training, and evaluation steps.
- `README.md`: Project documentation.

---

## 📊 Dataset

The dataset consists of labeled news articles or headlines with:
- `Label = 0`: Real news
- `Label = 1`: Fake news

The dataset may include features like:
- `title`, `text`, `subject`, and `date` depending on the source.

Popular sources include Kaggle datasets like:
- [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

---

## 🤖 Model Used

- **BERT (Bidirectional Encoder Representations from Transformers)**
- Fine-tuned for binary classification
- Implemented via **Hugging Face Transformers**

---

## 🔧 Technologies and Libraries

- Python 3.x
- Hugging Face Transformers
- Tokenizers
- PyTorch / TensorFlow (based on the backend used)
- scikit-learn
- pandas, numpy
- matplotlib / seaborn

---

## 🛠️ Features

- Text preprocessing (cleaning, stopword removal, etc.)
- BERT tokenizer and input formatting
- Fine-tuning BERT for fake news classification
- Performance evaluation (accuracy, precision, recall, F1-score, confusion matrix)

---

