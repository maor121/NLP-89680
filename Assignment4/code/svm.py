"""
Example file, downloaded from the internet.
Will be deleted later
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import precision_score, accuracy_score, recall_score


def run_svm_print_result(trainX, trainY, devX, devY, classes_dict):
    svc = svm.SVC(kernel='linear', C=1, gamma='auto', class_weight='balanced').fit(trainX, trainY)
    devPrediction = svc.predict(devX)

    assert len(devPrediction) == len(devY)

    precision = precision_score(devY, devPrediction, average=None)
    recall = recall_score(devY, devPrediction, average=None)
    accuracy = accuracy_score(devY, devPrediction)
    f1 = 2 * precision*recall / (precision + recall)

    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format({classes_dict[i]:("%.3f" % p) for i, p in enumerate(precision)}))
    print("Recall: {}".format({classes_dict[i]:("%.3f" % r) for i, r in enumerate(recall)}))
    print("F1: {}".format({classes_dict[i]: ("%.3f" % f) for i, f in enumerate(f1)}))

def run_svm_show_result(X,y):
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=1, gamma='auto').fit(X, y)

    if len(X[0]) > 2:
        from sklearn.manifold import TSNE
        print("To visualize SVM, running TSNE to transform X dim to 2.")
        print("Running TSNE on {} dots...".format(len(X))),
        X = TSNE(n_components=2).fit_transform(X)
        print("Done")

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = abs(x_max / x_min) / 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    plt.subplot(1, 1, 1)
    #Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    #Z = Z.reshape(xx.shape)
    #plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with linear kernel')
    plt.show()


if __name__ == '__main__':
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2] # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    y = iris.target

    run_svm_show_result(X,y)