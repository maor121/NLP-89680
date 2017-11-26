import sys
from sklearn.datasets import load_svmlight_file


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2:
        print "Wrong number of arguments. Use:\n" + \
                "python TrainSolver.py feature_vecs_file model_file"
        exit()

    feature_vec_filename = args[0]
    model_filename = args[1]

    X_train, y_train = load_svmlight_file(feature_vec_filename)