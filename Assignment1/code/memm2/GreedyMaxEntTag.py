import sys

from sklearn.externals import joblib

import code.memm1.memm_utils
from code.common import utils


class GreedyTag:
    def __init__(self, model, feature_map_dict_vect, T2I, common_words):
        self.__model = model
        self.__feature_map_dict_vect = feature_map_dict_vect
        self.__common_words = common_words
        self.__I2T = utils.inverse_dict(T2I)
    def getPrediction(self, sentence_words):
        words_fivlets = code.memm1.memm_utils.fivelets([None, None] + sentence_words + [None, None])

        predictions = []
        y_prev = utils.START_TAG
        y_prev_prev = utils.START_TAG
        for w_prev_prev, w_prev, wi, w_next, w_next_next in words_fivlets:
            wi_features = code.memm1.memm_utils.create_feature_vec(w_prev_prev, w_prev, wi, w_next, w_next_next, y_prev_prev, y_prev, wi in self.__common_words)
            wi_mapped_vec = self.__feature_map_dict_vect.transform(wi_features)

            y_id = self.__model.predict(wi_mapped_vec)[0]
            y = self.__I2T[y_id]
            predictions.append(y)

            y_prev_prev = y_prev
            y_prev = y
        return predictions


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 4:
        print "Wrong number of arguments. Use:\n" + \
                "python GreedyMaxEntTag.py input_file model_file output_file feature_map_file"
        exit()

    input_filename = args[0]
    model_filename = args[1]
    output_filename = args[2]
    feature_map_filename = args[3]

    logreg = joblib.load(model_filename)

    sentences = utils.read_input_file(input_filename, is_tagged=False, replace_numbers=False)
    feature_map_dict = code.memm1.memm_utils.feature_map_file_to_dict(feature_map_filename)
    T2I, feature_map_dict_vect = code.memm1.memm_utils.feature_dict_to_dict_vectorizer(feature_map_dict)
    common_words, tags = code.memm1.memm_utils.words_and_tags_from_map_dict(feature_map_dict)
    model = joblib.load(model_filename)

    tagger = GreedyTag(model, feature_map_dict_vect, T2I, common_words)

    # Run tagger and write prediction to file
    utils.predict_and_write_to_file(sentences, output_filename, tagger.getPrediction)
