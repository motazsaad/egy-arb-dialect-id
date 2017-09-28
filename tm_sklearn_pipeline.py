import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
import re
import nltk
import os
import glob
import numpy as np
from sklearn.datasets import load_files

import logging
logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
    'This is the first document.',
    'This is the second second second document.',
    'And the third one.',
    'Is this the first document?',
]

logging.info('loading data ...')
wiki_train = load_files('train/', encoding='utf-8')
x_train = wiki_train.data
y_train = wiki_train.target
print('x_train dataset size', len(x_train))
print('y_train dataset size', len(y_train))
print('labels: {}'.format(wiki_train.target_names))
logging.info('loading data is done ...')

logging.info('feature extraction ...')
vectorizer = CountVectorizer(ngram_range=(4, 4), min_df=20, analyzer='char_wb')

analyzer = vectorizer.build_analyzer()
test_sentence  = 'my name is motaz'
# print('test analyzer ({}): {}'.format(test_sentence, analyzer(test_sentence)))

# x_train = corpus
x_train_counts = vectorizer.fit_transform(x_train)
# print(vectorizer.get_feature_names())
# print('word counts: \n{}'.format(x_train_counts.toarray()))
print('counts sum: {}'.format(np.sum(x_train_counts.toarray())))

tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
# print('tf-idf matrix: \n{}'.format(x_train_tfidf.toarray()))
print('tf-idf sum: {}'.format(np.sum(x_train_tfidf.toarray())))
print('training dims:', x_train_counts.shape)
logging.info('feature extraction is done')


classifier = MultinomialNB()
# classifier = GaussianNB()
# classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
# classifier = LinearSVC()


logging.info('training ...')
classifier = classifier.fit(x_train_tfidf.toarray(), y_train)
logging.info('training is done')


###############################
print('testing')
test_dir = 'test/'
wiki_test = load_files('test/', encoding='utf-8')
x_test = wiki_test.data
y_test = wiki_test.target
print('x_test dataset size', len(x_test))
print('y_test dataset size', len(y_test))

x_test_counts = vectorizer.transform(x_test)
x_test_tfidf = tfidf_transformer.transform(x_test_counts)

# predicted = classifier.predict(x_test_counts.toarray())
predicted = classifier.predict(x_test_tfidf.toarray())

print('classifier:', classifier.__class__)
print('accuracy:', metrics.accuracy_score(y_test, predicted));
print(metrics.classification_report(y_test, predicted))

logging.info('testing is done')

