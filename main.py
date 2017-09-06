from data import getTrainingSet
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import re
from sklearn.externals import joblib
from time import time

print "Loading the data..."
N = 1000000
data, targets =  getTrainingSet(n=N)
# Fit vectorizer and trasform text data to matrix
print "Transforming..."
vectorizer = TfidfVectorizer()
# Use character ngram vectorizer to hopefully handle giberish.
# Careful: This makes for really slow processing and huge vectorizer and model.
# vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(4,8))
X = vectorizer.fit_transform(data)
y = np.asarray(targets)
# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Benchmark classifiers
def benchmark(clf):
    print '_' * 80
    print "Training: "
    print clf
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print "train time: %0.3fs" % train_time

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print "test time:  %0.3fs" % test_time

    score = metrics.accuracy_score(y_test, pred)
    print "accuracy:   %0.3f" % score

    print "confusion matrix:"
    print metrics.confusion_matrix(y_test, pred)

    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

results = []
model = MultinomialNB()
results.append(benchmark(model))
print results
joblib.dump(model, 'model/model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
