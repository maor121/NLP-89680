from collections import Counter
from MLETrain import MLETrain
import sys
import utils
import numpy as np


class HMMTag:
    __mletrain = None
    def __init__(self, mletrain):
        self.__mletrain = mletrain
    def getPrediction(self, sentence_words):
        tags = self.__mletrain.getTags()
        T2I = {t: i for i, t in enumerate(Counter(tags).keys())}
        I2T = utils.inverse_dict(T2I)

        V = np.zeros([len(sentence_words)+1, len(tags), len(tags)], dtype=np.float32)
        bp = np.ndarray([len(sentence_words)+1, len(tags), len(tags)], dtype=object)

        V[0, T2I[utils.START_TAG], T2I[utils.START_TAG]] = 1
        words_count = len(sentence_words)
        words_itr = iter(sentence_words)
        for i in range(1, words_count+1):
            wi = words_itr.next()
            for t in tags:
                t_id = T2I[t]
                E = self.__mletrain.getE(wi, t)
                for t_prev in tags:
                    t_prev_id = T2I[t_prev]

                    prev_row_calc = [V[i - 1, T2I[t_prev_prev], t_prev_id] * \
                        self.__mletrain.getQ(t, t_prev, t_prev_prev) * \
                        E
                        for t_prev_prev in tags]

                    max_id = np.argmax(prev_row_calc)
                    bp[i, t_prev_id, t_id] = max_id
                    V[i, t_prev_id, t_id] = prev_row_calc[max_id]

        pred_prev_last_id, pred_last_id  = np.argmax(V[words_count,:,:])

        predictions_ids = np.array(words_count, dtype=np.int32)
        predictions_ids[words_count-1] = pred_last_id
        predictions_ids[words_count-2] = pred_prev_last_id
        for i in xrange(words_count-2, 1, -1):
            prediction_ids[i] = bp[i+2, prediction_ids[i+1], pred_last_id[i+2]]
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

    sentences = utils.read_input_file(input_filename, is_tagged=True)

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

        progress = utils.progress_hook(sentences_processed, sentences_count, progress)
    hit_total = total - miss_total
    accuracy = hit_total * 1.0 / total
    print("accuracy: {} in {} words".format(str(accuracy), str(total)))