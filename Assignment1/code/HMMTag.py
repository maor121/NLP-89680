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

        words_count = len(sentence_words)
        tags_count = len(tags)

        V = np.full([words_count+1, tags_count, tags_count], float('-inf'), dtype=np.float32)
        bp = np.zeros([words_count+1, tags_count, tags_count], dtype=np.int32)

        V[0, T2I[utils.START_TAG], T2I[utils.START_TAG]] = np.log(1)
        words_itr = iter(sentence_words)
        for i in range(1, words_count+1):
            wi = words_itr.next()

            # Beam search: pick from last layer only results that pass threshold Mt*k
            # Mt = highest score at time t
            # k = percentage (parameter)
            Mt = np.max(V[i-1,:,:])
            threshold = Mt + np.log(0.1)
            tag_ids_in_beam = np.argwhere(V[i-1,:,:] > threshold)

            tag_prev_prev_ids_beam = list(set(utils.reduce_tuple_list(tag_ids_in_beam, 0)))
            tag_prev_ids_beam = list(set(utils.reduce_tuple_list(tag_ids_in_beam, 1)))

            for t in tags:
                t_id = T2I[t]
                E = self.__mletrain.getE(wi, t)
                for t_prev_id in tag_prev_ids_beam:
                    t_prev = I2T[t_prev_id]
                    prev_row_calc = [V[i - 1, t_prev_prev_id, t_prev_id] + \
                                     np.log(self.__mletrain.getQ(t, t_prev, I2T[t_prev_prev_id])) + \
                                     np.log(E)
                                     for t_prev_prev_id in tag_prev_prev_ids_beam]

                    max_id_calc = np.argmax(prev_row_calc)
                    max_prev_prev_id = tag_prev_prev_ids_beam[max_id_calc]
                    bp[i, t_prev_id, t_id] = max_prev_prev_id
                    V[i, t_prev_id, t_id] = prev_row_calc[max_id_calc]

        pred_prev_last_id, pred_last_id  = np.unravel_index(np.argmax(V[words_count,:,:]), [tags_count, tags_count])

        prediction_ids = np.zeros(words_count, dtype=np.int32)
        prediction_ids[words_count-1] = pred_last_id
        prediction_ids[words_count-2] = pred_prev_last_id
        for i in xrange(words_count-3, 1, -1):
            prediction_ids[i] = bp[i+2, prediction_ids[i+1], prediction_ids[i+2]]
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

    #TODO: Write predictions to file