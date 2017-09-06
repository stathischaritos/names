from data import getTrainingSet
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import re
from sklearn.externals import joblib
from time import time

print "Loading the data..."
N = 1000000
data, targets = getTrainingSet(n=N, shuffle_data=True)
# Fit vectorizer and trasform text data to matrix.
vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1,3))
# Trasform data
print "Transforming..."
X = vectorizer.fit_transform(data)
y = np.asarray(targets)
# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Benchmark classifiers
def benchmark(clf, name):
    print '_' * 80
    print "Training: "
    print name
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

models = {
    "multi_nb": MultinomialNB(),
    # "ridge_classifier" : RidgeClassifier(),
    # "knn": KNeighborsClassifier(n_neighbors=10),
    # "random_forest": RandomForestClassifier(n_estimators=10)
}
results = []
# Train all models and save in file
for model in models:
    clf = models[model]
    results.append(benchmark(clf, model))
    joblib.dump(clf, 'model/' + model + '.pkl')

# Save vectorizer to file
joblib.dump(vectorizer, 'model/vectorizer.pkl')
