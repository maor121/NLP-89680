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

    miss_total = 0
    total = 0
    sentences_processed = 0
    sentences_count = len(sentences)
    progress = None
    for (words, tags) in sentences:
        prediction = tagger.getPrediction(words)
        # Numbers are more easily compared then strings
        prediction_ids = model.getTagsIds(prediction)
        tags_ids = model.getTagsIds(tags[2:])  # Skip START
        miss_total += sum(1 for i, j in zip(prediction_ids, tags_ids) if i != j)
        total += len(prediction_ids)
        sentences_processed += 1

        #Temp for debugging
        is_debug = False
        if (is_debug):
            misses = np.argwhere(np.array(prediction_ids) - np.array(tags_ids))
            if (len(misses) > 0):
                print(words)
                print(prediction)
                print(tags[2:])
                for miss in misses:
                    print("{} {} [{}/{}] E-[{},{}]".format(
                        miss[0],
                        words[miss[0]],
                        prediction[miss[0]],
                        tags[2:][miss[0]],
                        model.getE(miss[0], prediction[miss[0]]),
                        model.getE(miss[0], tags[2:][miss[0]]))
                    )

        progress = utils.progress_hook(sentences_processed, sentences_count, progress)
        #break
    hit_total = total - miss_total
    accuracy = hit_total * 1.0 / total
    print("accuracy: {} in {} words".format(str(accuracy), str(total)))

    #TODO: Write predictions to file