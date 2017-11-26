import sys

from sklearn.externals import joblib

from code.memm1 import memm_utils
from code.common import utils, viterbi


class MEMMTag:
    def __init__(self, model, feature_map_dict_vect, T2I, common_words):
        self.__model = model
        self.__feature_map_dict_vect = feature_map_dict_vect
        self.__common_words = common_words
        self.__I2T = utils.inverse_dict(T2I)
        self.__prob_cache = {}
    def getPrediction(self, sentence_words):
        words_count = len(sentence_words)
        tags_count = len(self.__I2T)

        start_tag_id = T2I[utils.START_TAG]

        words_fivlets = memm_utils.fivelets([None, None] + sentence_words + [None, None])

        getLogScore = self.__getLogScore

        prediction_ids = viterbi.run_viterbi_2nd_order_log_with_beam_search(
            words_fivlets, words_count, tags_count, start_tag_id, getLogScore
        )
        self.__evictCache()

        return [self.__I2T[p_id] for p_id in prediction_ids]
    def __getLogScore(self, w_window, t_id, t_prev_id, t_prev_prev_id):
        key = (w_window, t_prev_id, t_prev_prev_id)
        if key not in self.__prob_cache:
            self.__prob_cache[key] = self.__calcLogProb(w_window, t_prev_id, t_prev_prev_id)
        return self.__prob_cache[key][0][t_id]
    def __calcLogProb(self, w_window, t_prev_id, t_prev_prev_id):
        return \
            self.__model.predict_proba(
            self.__feature_map_dict_vect.transform(
                memm_utils.create_feature_vec(w_window[0], w_window[1], w_window[2], w_window[3], w_window[4],
                                              self.__I2T[t_prev_id], self.__I2T[t_prev_prev_id],
                                              w_window[2] in self.__common_words)
            )
        )
    def __evictCache(self):
        self.__prob_cache.clear()

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 4:
        print "Wrong number of arguments. Use:\n" + \
                "python MEMMTag.py input_file model_file output_file feature_map_file"
        exit()

    input_filename = args[0]
    model_filename = args[1]
    output_filename = args[2]
    feature_map_filename = args[3]

    logreg = joblib.load(model_filename)

    sentences = utils.read_input_file(input_filename, is_tagged=False, replace_numbers=False)
    feature_map_dict = memm_utils.feature_map_file_to_dict(feature_map_filename)
    T2I, feature_map_dict_vect = memm_utils.feature_dict_to_dict_vectorizer(feature_map_dict)
    common_words, tags = memm_utils.words_and_tags_from_map_dict(feature_map_dict)
    model = joblib.load(model_filename)

    # Add Start tag to dict
    T2I[utils.START_TAG] = len(T2I)

    tagger = MEMMTag(model, feature_map_dict_vect, T2I, common_words)

    # Run tagger and write prediction to file
    utils.predict_and_write_to_file(sentences, output_filename, tagger.getPrediction)