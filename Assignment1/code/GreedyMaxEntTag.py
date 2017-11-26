import sys
import utils
from sklearn.externals import joblib
import memm_utils

class GreedyTag:
    __model = None
    __feature_map_dict = None
    __tags = None
    def __init__(self, model, feature_map_dict, tags, W2I):
        self.__model = model
        self.__feature_map_dict = feature_map_dict
        self.__tags = tags
        self.__W2I = W2I
    def getPrediction(self, sentence_words):
        words_fivlets = memm_utils.fivelets([None, None] + sentence_words + [None, None])

        predictions = []
        y_prev = utils.START_TAG
        y_prev_prev = utils.START_TAG
        for w_prev_prev, w_prev, wi, w_next, w_next_next in words_fivlets:
            wi_features = memm_utils.create_feature_vec(w_prev_prev, w_prev, wi, w_next, w_next_next, y_prev_prev, y_prev, self.__W2I)
            wi_mapped_vec = memm_utils.feature_string_vec_to_sparse_dict(wi_features)

            y = self.__model.predict(wi_mapped_vec)
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

    words, tags = utils.read_input_file(input_filename, is_tagged=True, replace_numbers=False)

    miss_total = 0
    total = 0
    sentences_processed = 0
    sentences_count = len(sentences)
    progress = None
    for (words, tags) in sentences:
        prediction = tagger.getPrediction(words)
        #Numbers are more easily compared then strings
        prediction_ids = model.getTagsIds(prediction)
        tags_ids = model.getTagsIds(tags[2:]) #Skip START
        miss_total += sum(1 for i, j in zip(prediction_ids, tags_ids) if i != j)
        total += len(prediction_ids)
        sentences_processed += 1

        progress = utils.progress_hook(sentences_processed, sentences_count, progress)
    hit_total = total - miss_total
    accuracy = hit_total * 1.0 / total
    print("accuracy: {} in {} words".format(str(accuracy), str(total)))

    #TODO: Write predictions to file