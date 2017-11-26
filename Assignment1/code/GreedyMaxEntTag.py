import sys
import utils
from sklearn.externals import joblib
import memm_utils

class GreedyTag:
    __model = None
    def __init__(self, model):
        self.__model = model
    def getPrediction(self, sentence_words):
        tags = self.__model.getTags()

        predictions = []
        y_prev = utils.START_TAG
        y_prev_prev = utils.START_TAG
        for word in sentence_words:
            y = max(tags, key=lambda t: self.__mletrain.getE(word, t)*self.__mletrain.getQ(t, y_prev, y_prev_prev))
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