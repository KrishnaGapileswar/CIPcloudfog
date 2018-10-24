# from flask import Flask,render_template, url_for, request, jsonify
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib

def train(Cparam):
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    numSamples = len(X)
    # print(numSamples)
    # fit model
    clf = svm.SVC(C=float(Cparam),
                  probability=True,
                  random_state=1)
    clf.fit(X, y)

    # persist model
    joblib.dump(clf, 'model.pkl')

    return ({'accuracy': round(clf.score(X, y) * 100, 2)}, numSamples)
    # /usr/local/lib/python3.4/dist-packages/sklearn/datasets/data

# res = train(1.0)
# print(res[0]['accuracy'])