from MLETrain import MLETrain
import sys
import utils

class GreedyTag:
    __mletrain = None
    def __init__(self, mletrain):
        self.__mletrain = mletrain
    def getPrediction(self, sentence_words):
        tags = self.__mletrain.getTags()

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
    if len(args) not in (4, 5):
        print "Wrong number of arguments. Use:\n" + \
                "python GreedyTag.py input_file q_mle_file e_mle_file, out_file, [extra_file]"
        exit()

    input_filename = args[0]
    q_mle_filename = args[1]
    e_mle_filename = args[2]
    out_filename = args[3]
    extra_filename = args[4] if len(args) >= 5 else None

    model = MLETrain(q_mle_filename, e_mle_filename)
    tagger = GreedyTag(model)

    sentences = utils.read_input_file(input_filename, is_tagged=True)

    miss_total = 0
    total = 0
    sentences_processed = 0
    sentences_count = len(sentences)
    progress = None
    for (words, tags) in sentences:
        prediction = tagger.getPrediction(words)
        #Numbers are more easily compared then strings
        prediction_ids = model.getTagsIds(prediction)
        tags_ids = model.getTagsIds(tags[:2]) #Skip START
        miss_total += sum(1 for i, j in zip(prediction_ids, tags_ids) if i != j)
        total += len(prediction_ids)
        sentences_processed += 1

        progress = utils.progress_hook(sentences_processed, sentences_count, progress)
        if (progress == 10):
            break
    hit_total = total - miss_total
    accuracy = hit_total * 1.0 / total
    print("accuracy: {} in {} words".format(str(accuracy), str(total)))
