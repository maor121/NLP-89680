import sys
import utils
from sklearn.externals import joblib
import memm_utils


class MEMMTag:
    def __init__(self, model, feature_map_dict_vect, T2I, common_words):
        self.__model = model
        self.__feature_map_dict_vect = feature_map_dict_vect
        self.__common_words = common_words
        self.__I2T = utils.inverse_dict(T2I)
    def getPrediction(self, sentence_words):


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

    sentences = utils.read_input_file(input_filename, is_tagged=True, replace_numbers=False)
    feature_map_dict = memm_utils.feature_map_file_to_dict(feature_map_filename)
    T2I, feature_map_dict_vect = memm_utils.feature_dict_to_dict_vectorizer(feature_map_dict)
    common_words, tags = memm_utils.words_and_tags_from_map_dict(feature_map_dict)
    model = joblib.load(model_filename)

    tagger = MEMMTag(model, feature_map_dict_vect, T2I, common_words)

    miss_total = 0
    total = 0
    sentences_processed = 0
    sentences_count = len(sentences)
    progress = None
    for (words, tags) in sentences:
        prediction = tagger.getPrediction(words)
        #Numbers are more easily compared then strings
        prediction_ids = [T2I[t] for t in prediction]
        tags_ids = [T2I[t] for t in tags[2:]] #Skip START
        miss_total += sum(1 for i, j in zip(prediction_ids, tags_ids) if i != j)
        total += len(prediction_ids)
        sentences_processed += 1

        progress = utils.progress_hook(sentences_processed, sentences_count, progress)
    hit_total = total - miss_total
    accuracy = hit_total * 1.0 / total
    print("accuracy: {} in {} words".format(str(accuracy), str(total)))

    #TODO: Write predictions to file