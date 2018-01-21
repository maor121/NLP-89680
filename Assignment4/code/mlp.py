from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import  OneHotEncoder
from sklearn.metrics import precision_score, accuracy_score, recall_score

def run_mlp_print_result(trainX, trainY, devX, devY, classes_dict, features_per_dim):
    onehot_encoder = OneHotEncoder(sparse=False, n_values=features_per_dim)
    onehot_trainX = onehot_encoder.fit_transform(trainX)
    onehot_devX = onehot_encoder.fit_transform(devX)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    hidden_layer_sizes = (onehot_trainX.shape[1], len(classes_dict)), random_state = 1,
                        verbose=False)
    clf.fit(onehot_trainX, trainY)

    devPrediction = clf.predict(onehot_devX)

    assert len(devPrediction) == len(devY)

    precision = precision_score(devY, devPrediction, average=None)
    recall = recall_score(devY, devPrediction, average=None)
    accuracy = accuracy_score(devY, devPrediction)

    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format({classes_dict[i]: ("%.3f" % p) for i, p in enumerate(precision)}))
    print("Recall: {}".format({classes_dict[i]: ("%.3f" % r) for i, r in enumerate(recall)}))

