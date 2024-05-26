import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# read the dataset
train = pd.read_csv('train.csv')
validation = pd.read_csv('validation.csv')
test = pd.read_csv('test.csv')

train.shape, validation.shape, test.shape

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    pattern = f"[a-zA-Z\s]"
    text = ''.join(re.findall(pattern, text))
    text = text.lower()
    tokens = nltk.word_tokenize(text)
#     tokens = ' '.join(tokens)
    return tokens

#apply the preprocess text to 
train['user_review'] = train['user_review'].apply(preprocess_text)
validation['user_review'] = validation['user_review'].apply(preprocess_text)
test['user_review'] = test['user_review'].apply(preprocess_text)

train.head()

def build_vocabulary(reviews):
    vocab = {}
    index = 1  # Start indexing from 1; reserve 0 for padding
    for review in reviews:
        for word in review:
            if word not in vocab:
                vocab[word] = index
                index += 1
    return vocab

# Concatenate all reviews to build the vocabulary
all_reviews = train['user_review'].tolist() + validation['user_review'].tolist() + test['user_review'].tolist()
vocab = build_vocabulary(all_reviews)
print("Vocabulary Length:", len(vocab))
first_50 = list(vocab.items())[:50]
for key, value in first_50:
    print(f'{key}: {value}')


#Indexing reviews based on the vocabulary
def index_and_pad_reviews(reviews, vocab, max_length=100):
    """Index and pad tokenized reviews to a fixed length."""
    indexed_reviews = []
    for review in reviews:
        indexed_review = [vocab.get(word, 0) for word in review]  # Use vocab.get to handle unknown words
        # Truncate if review length exceeds max_length
        truncated_review = indexed_review[:max_length]
        # Pad review with zeros if it's shorter than max_length
        padded_review = truncated_review + [0] * (max_length - len(truncated_review))
        indexed_reviews.append(padded_review)
    return indexed_reviews


