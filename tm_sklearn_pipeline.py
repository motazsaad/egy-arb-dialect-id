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





logging.info(' loading training data ...')
wiki_train = load_files('train/', encoding='utf-8')
print('dataset size', len(wiki_train.data))
print('labels: {}'.format(wiki_train.target_names))

text_clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(4, 4), min_df=25, analyzer='char_wb')),
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
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()

logging.info(' train the classifier ...')
text_clf_pipeline.fit(wiki_train.data, wiki_train.target)
logging.info(' training is done ...')

logging.info(' evaluation ...')
logging.info(' loading test data ...')
wiki_test = load_files('test/', encoding='utf-8')
print('dataset size', len(wiki_test.data))
print('labels: {}'.format(wiki_test.target_names))

predicted = text_clf_pipeline.predict(wiki_test.data)
print('Accuracy = {}'.format(np.mean(predicted == wiki_test.target)))
print('classification report \n{}'.format(metrics.classification_report(y_pred=predicted,
                                                                        y_true=wiki_test.target)))
print('confusion matrix \n{}'.format(metrics.confusion_matrix(y_pred=predicted,
                                                                        y_true=wiki_test.target)))
print('f1-score: {0:.2f}'.format(metrics.f1_score(y_pred=predicted, y_true=wiki_test.target)))
logging.info(' done !')

