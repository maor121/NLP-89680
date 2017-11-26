import sys

from code.common import utils
from code.hmm1.MLETrain import MLETrain


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

    sentences = utils.read_input_file(input_filename, is_tagged=False, replace_numbers=True)

    # Run tagger and write prediction to file
    utils.predict_and_write_to_file(sentences, out_filename, tagger.getPrediction)
