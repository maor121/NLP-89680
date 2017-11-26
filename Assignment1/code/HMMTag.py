from collections import Counter
from MLETrain import MLETrain
import sys
import utils
import numpy as np
import viterbi

class HMMTag:
    __mletrain = None
    def __init__(self, mletrain):
        self.__mletrain = mletrain
    def getPrediction(self, sentence_words):
        tags = self.__mletrain.getTags()
        T2I = utils.list_to_ids(tags, ID_SHIFT=0)
        I2T = utils.inverse_dict(T2I)

        words_count = len(sentence_words)
        tags_count = len(tags)
        start_tag_id = T2I[utils.START_TAG]
        getLogScore = lambda wi, t_id, t_prev_id, t_prev_prev_id : \
            np.log(self.__mletrain.getQ(I2T[t_id], I2T[t_prev_id], I2T[t_prev_prev_id])) + \
            np.log(self.__mletrain.getE(wi, I2T[t_id]))

        prediction_ids = viterbi.run_viterbi_2nd_order_log_with_beam_search(
            sentence_words, words_count, tags_count, start_tag_id, getLogScore
        )

        predictions = [I2T[p_id] for p_id in prediction_ids]
        return predictions


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) not in (4, 5):
        print "Wrong number of arguments. Use:\n" + \
                "python HMMTag.py input_file q_mle_file e_mle_file, out_file, [extra_file]"
        exit()

    input_filename = args[0]
    q_mle_filename = args[1]
    e_mle_filename = args[2]
    out_filename = args[3]
    extra_filename = args[4] if len(args) >= 5 else None

    model = MLETrain(q_mle_filename, e_mle_filename)
    tagger = HMMTag(model)

    sentences = utils.read_input_file(input_filename, is_tagged=True, replace_numbers=True)

    # Run tagger and write prediction to file
    utils.predict_and_write_to_file(sentences, out_filename, tagger.getPrediction)
