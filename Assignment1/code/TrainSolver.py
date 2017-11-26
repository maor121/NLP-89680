import collections
import sys

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2:
        print "Wrong number of arguments. Use:\n" + \
                "python TrainSolver.py feature_vecs_file model_file"
        exit()

    feature_vec_filename = args[0]
    model_filename = args[1]

    print "Loading file"
    X_train, y_train = load_svmlight_file(feature_vec_filename)

    print "Initializing"
    logreg = LogisticRegression(solver='sag',multi_class='multinomial', verbose=True, n_jobs=-1, tol=1e-4, max_iter=10)

    print('training model...')
    logreg.fit(X_train, y_train)

    print('eval model (on the training set):')
    predictions = logreg.predict(X_train)

    """
    logreg = joblib.load(model_filename)
    first_xtrain = X_train[0]
    first_pred = predictions[0:first_xtrain.shape[1]]
    first_ytrain = y_train[0:first_xtrain.shape[1]]
    print(first_xtrain)
    print(first_pred)
    print(first_ytrain)
    """

    predictions_count = collections.Counter(np.equal(predictions, y_train))
    success_rate = float(predictions_count[True]) / len(predictions) * 100
    print('success rate: ' + str(success_rate) + '%')

    print "Saving model"
    joblib.dump(logreg, model_filename)

    print "Done"