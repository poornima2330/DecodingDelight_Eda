# Sentiment Analysis of Amazon Fine Food Reviews

## Introduction

This project aims to perform sentiment analysis on the Amazon Fine Food Reviews dataset using various machine learning and deep learning techniques.

## Machine Learning and Deep Learning Techniques

The project utilizes the following machine learning and deep learning techniques:

### Machine Learning Techniques:

- **Logistic Regression**: A linear classifier used for binary and multi-class classification tasks.
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem, often used for text classification tasks.

### Deep Learning Techniques:

- **Recurrent Neural Network (RNN)**: Specifically, Long Short-Term Memory (LSTM) networks are employed for sequence processing and sentiment analysis.
- **Embedding Layers**: Transform text data into dense vector representations suitable for deep learning models.
- **RoBERTa Pre-Trained Model**: A variant of the BERT model that is pre-trained on a large corpus of text data. It provides state-of-the-art performance in various NLP tasks, including sentiment analysis.

## Data Preprocessing

Before applying the machine learning and deep learning techniques, the data undergoes preprocessing steps, including:

- Text cleaning (removing punctuation, special characters, etc.).
- Tokenization: Splitting text into individual words or tokens.
- Stopword removal: Eliminating common words that do not carry significant meaning.
- Text vectorization: Converting text data into numerical vectors for model training.

## Evaluation Metrics

The performance of the sentiment analysis models is evaluated using the following metrics:

- **Accuracy**: The proportion of correctly classified reviews.
- **Confusion Matrix**: A tabular representation of predicted vs. actual sentiment classes.

## Dependencies

The project relies on the following libraries and packages:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- NLTK (Natural Language Toolkit)
- Keras (with TensorFlow backend)
- WordCloud

## Instructions

To run the sentiment analysis code:

1. Ensure all dependencies are installed in your Python environment.
2. Load the Amazon Fine Food Reviews dataset.
3. Preprocess the text data.
4. Run the sentiment analysis code using the provided machine learning and deep learning techniques.
