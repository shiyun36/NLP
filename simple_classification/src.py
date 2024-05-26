import nltk
import pandas as pd
import numpy as np
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('punkt')
nltk.download('stopwords')

train = pd.read_csv('train.csv')
validation = pd.read_csv('validation.csv')
test = pd.read_csv('test.csv')

train.shape, validation.shape, test.shape
train.head(), test.head(), validation.head()
test['user_suggestion'].value_counts()

##Preprocessing function 
def preprocess_text(texts):
    # lemmatize the tokens and store them in a list
    processed_texts = []
    for doc in nlp.pipe(texts, n_process=-1):
        lemmatized_tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.lemma_ not in nlp.Defaults.stop_words]
        processed_text = " ".join(lemmatized_tokens)
        processed_texts.append(processed_text)
    return processed_texts

# apply preprcoess_text function to user_review column
train['user_review'] = preprocess_text(train['user_review'])
validation['user_review'] = preprocess_text(validation['user_review'])
test['user_review'] = preprocess_text(test['user_review'])

train['user_review'].head()

### Vectorization 
## One hot encoding 
count_vectorizer_ohe = CountVectorizer(min_df=0.001, binary=True)
count_vectorizer_ohe_train = count_vectorizer_ohe.fit_transform(train['user_review'])
