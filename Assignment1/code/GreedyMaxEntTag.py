import sys
import utils
from sklearn.externals import joblib
import memm_utils

class GreedyTag:
    def __init__(self, model, feature_map_dict, tags, common_words):
        self.__model = model
        self.__feature_map_dict = feature_map_dict
        self.__tags = tags
        self.__common_words = common_words
        self.__I2T = utils.inverse_dict(feature_map_dict.vocabulary_)
    def getPrediction(self, sentence_words):
        words_fivlets = memm_utils.fivelets([None, None] + sentence_words + [None, None])

        predictions = []
        y_prev = utils.START_TAG
        y_prev_prev = utils.START_TAG
        for w_prev_prev, w_prev, wi, w_next, w_next_next in words_fivlets:
            wi_features = memm_utils.create_feature_vec(w_prev_prev, w_prev, wi, w_next, w_next_next, y_prev_prev, y_prev, self.__common_words)
            wi_mapped_vec = self.__feature_map_dict.transform(wi_features)

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

    sentences = utils.read_input_file(input_filename, is_tagged=True, replace_numbers=False)
    feature_map_dict = memm_utils.feature_map_file_to_dict(feature_map_filename)
    common_words, tags = memm_utils.words_and_tags_from_map_dict(feature_map_dict.vocabulary_)
    model = joblib.load(model_filename)

    tagger = GreedyTag(model, feature_map_dict, tags, common_words)

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