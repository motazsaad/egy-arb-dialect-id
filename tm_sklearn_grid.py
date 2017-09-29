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

import psutil

logging.info(' loading training data ...')
wiki_train = load_files('ar_arz_wiki_corpus/train/', encoding='utf-8')
print('dataset size', len(wiki_train.data))
print('labels: {}'.format(wiki_train.target_names))

text_clf_pipeline = Pipeline([('vect', CountVectorizer(min_df=25, analyzer='char_wb')),
                              ('tfidf', TfidfTransformer()),
                              ('clf', MultinomialNB()), ])
print('pipline info:')
for i, step in enumerate(text_clf_pipeline.get_params(deep=False)['steps']):
    print('step {}: {}'.format(i, step))

parameters = {
    'vect__ngram_range': ((4, 4), (5, 5), (4, 5)),
}

grid_search = GridSearchCV(text_clf_pipeline, parameters, n_jobs=-1, verbose=1)
print("Performing grid search...")
print("pipeline:", [name for name, _ in text_clf_pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
grid_search.fit(wiki_train.data, wiki_train.target)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

grid_search.estimator