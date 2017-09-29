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
from sklearn.model_selection import GridSearchCV
import re
import nltk
import os
import glob
import numpy as np
from sklearn.datasets import load_files
from pprint import pprint
from time import time
import logging
logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s', level=logging.INFO)


def experiment(text_clf_pipeline):
    logging.info(' loading training data ...')
    wiki_train = load_files('ar_arz_wiki_corpus/train/', encoding='utf-8')
    print('dataset size', len(wiki_train.data))
    print('labels: {}'.format(wiki_train.target_names))

    print('pipline info:')
    for i, step in enumerate(text_clf_pipeline.get_params(deep=False)['steps']):
        print('step {}: {}'.format(i, step))

    logging.info(' train the classifier ...')
    text_clf_pipeline.fit(wiki_train.data, wiki_train.target)
    logging.info(' training is done ...')

    logging.info(' evaluation ...')
    logging.info(' loading test data ...')
    wiki_test = load_files('ar_arz_wiki_corpus/test/', encoding='utf-8')
    print('dataset size', len(wiki_test.data))
    print('labels: {}'.format(wiki_test.target_names))

    predicted = text_clf_pipeline.predict(wiki_test.data)
    print('Accuracy = {}'.format(np.mean(predicted == wiki_test.target)))
    print('classification report \n{}'.format(metrics.classification_report(y_pred=predicted,
                                                                            y_true=wiki_test.target)))
    print('confusion matrix \n{}'.format(metrics.confusion_matrix(y_pred=predicted,
                                                                            y_true=wiki_test.target)))
    f1 = 'f1-score: {0:.3f}'.format(metrics.f1_score(y_pred=predicted, y_true=wiki_test.target))
    print(f1)
    logging.info(' done !')
    return f1


text_clf_pipeline1 = Pipeline([('vect', CountVectorizer(ngram_range=(4, 4),
                                                        min_df=25,
                                                        analyzer='char_wb')),
                              ('tfidf', TfidfTransformer()),
                              ('clf', MultinomialNB()), ])

text_clf_pipeline2 = Pipeline([('vect', CountVectorizer(ngram_range=(5, 5),
                                                        min_df=25,
                                                        analyzer='char_wb')),
                              ('tfidf', TfidfTransformer()),
                              ('clf', MultinomialNB()), ])

text_clf_pipeline3 = Pipeline([('vect', CountVectorizer(ngram_range=(6, 6),
                                                        min_df=25,
                                                        analyzer='char_wb')),
                              ('tfidf', TfidfTransformer()),
                              ('clf', MultinomialNB()), ])

text_clf_pipeline4 = Pipeline([('vect', CountVectorizer(ngram_range=(7, 7),
                                                        min_df=25,
                                                        analyzer='char_wb')),
                              ('tfidf', TfidfTransformer()),
                              ('clf', MultinomialNB()), ])

if __name__ == '__main__':
    result = {}
    for i in range(4, 7):
        min_gram = i
        max_gram = i
        text_clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(min_gram, max_gram),
                                                            min_df=25,
                                                            analyzer='char_wb')),
                                   ('tfidf', TfidfTransformer()),
                                   ('clf', MultinomialNB()), ])
        f1 = experiment(text_clf_pipeline)
        result[(min_gram, max_gram)] = f1
    for i in range(4, 7):
        min_gram = i
        max_gram = i+1
        text_clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(min_gram, max_gram),
                                                            min_df=25,
                                                            analyzer='char_wb')),
                                   ('tfidf', TfidfTransformer()),
                                   ('clf', MultinomialNB()), ])
        f1 = experiment(text_clf_pipeline)
        result[(min_gram, max_gram)] = f1
    print(result)





