import sys
from collections import Counter
import logging
import numpy as np
import utils
from utils import list_to_ids, reduce_tuple_list, flatten


COMMON_WORD_MIN_COUNT = 5 # 5 times is common enough


class MLETrain:
    __q_counts = None
    __e_counts = None
    __T2I = None
    __W2I = None
    __words_trained_count = None

    __Q_cache = None
    __cache_hit_count = None
    __cache_miss_count = None

    def __init__(self, q_mle_file, e_mle_file):
        self.__T2I, self.__W2I, self.__q_counts, self.__e_counts, self.__words_trained_count = \
            utils.read_mle_files(q_mle_file, e_mle_file)
        tags_count = len(self.__T2I)
        self.__Q_cache = np.full([tags_count, tags_count, tags_count], -1, dtype=np.float32)
        self.__cache_hit_count = self.__cache_miss_count = 0

    def getQ(self, c, b, a):
        c_id, b_id, a_id = [self.__T2I[t]-1 for t in (c,b,a)]

        if self.__Q_cache[c_id, b_id, a_id] == -1:
            self.__Q_cache[c_id, b_id, a_id] = self.__calc_Q(c, b, a)
            self.__cache_miss_count += 1
        else:
            self.__cache_hit_count += 1
        return self.__Q_cache[c_id, b_id, a_id]

    def __calc_Q(self, c, b, a):
        """(c,b,a) - (tag, tag_prev, tag_prev_prev)"""
        c_count = self.__get_tag_count([c]) * 1.0
        ba_count = self.__get_tag_count([a, b])
        b_count = self.__get_tag_count([b])

        three = self.__get_tag_count([a, b, c]) * 1.0 / ba_count if ba_count != 0 else 0
        two = self.__get_tag_count([b, c]) * 1.0 / b_count if b_count != 0 else 0
        one = c_count / self.__words_trained_count
        return ( three + two + one ) / 3

    def getE(self, word, tag):
        word_id = self.__get_word_id(word, self.__W2I)
        tag_id = self.__T2I.get(tag)
        word_count = self.__e_counts.get((word_id, tag_id), 0)
        tag_count = self.__get_tag_count([tag])
        return float(word_count) / tag_count

    def getTags(self):
        return self.__T2I.keys()

    def getTagsIds(self, tags):
        return [self.__T2I[t] for t in tags]

    def __get_tag_count(self, tags):
        tags_ids = filter(None, [self.__T2I.get(t) for t in tags])
        if len(tags_ids) != len(tags):
            return 0
        return self.__q_counts.get(tuple(tags_ids), 0)

    @staticmethod
    def createModelFilesFromInput(input_filename, q_mle_filename, e_mle_filename):
        logging.basicConfig()
        log = logging.getLogger()
        log.setLevel(logging.DEBUG)

        log.debug("Reading input file")
        train_data = utils.read_input_file(input_filename, replace_numbers=False)

        log.debug("- Converting words\\tags to ids")
        W2I = list_to_ids(flatten(reduce_tuple_list(train_data, dim=0)), MIN_COUNT=COMMON_WORD_MIN_COUNT)
        #Unknown words
        unk_words = MLETrain.__generate_unk_words()
        i = max(W2I.values())+1
        for w_unk in unk_words:
            W2I[w_unk] = i
            i +=1
        T2I = list_to_ids(flatten(reduce_tuple_list(train_data, dim=1)))
        train_data_ids = MLETrain.__sentences_to_ids(train_data, W2I, T2I)
        # Inverse dictionary
        I2T = utils.inverse_dict(T2I)
        I2W = utils.inverse_dict(W2I)

        log.debug("- Counting:")
        count_tag_triplets = Counter()
        count_tag_pairs = Counter()
        count_tag_single = Counter()
        count_word_tags = Counter()
        count_word_tags.update()
        for sentence in train_data_ids:
            words_ids = sentence[0]
            tags_ids = sentence[1]
            # Q
            count_tag_triplets.update(utils.count_triplets(tags_ids))
            count_tag_pairs.update(utils.count_pairs(tags_ids))
            count_tag_single.update(utils.count_single(tags_ids))
            # E
            count_word_tags.update(utils.count_word_tags(words_ids, tags_ids))

        log.debug("Writing to file {}".format(q_mle_filename))
        utils.write_q_mle_file(count_tag_triplets, count_tag_pairs, count_tag_single, I2T, q_mle_filename)

        log.debug("Writing to file {}".format(e_mle_filename))
        utils.write_e_mle_file(count_word_tags, I2T, I2W, e_mle_filename)

        log.debug("Done")
    @staticmethod
    def __generate_unk_words():
        return [utils.UNK_Word] + [utils.UNK_Word+s for s in utils.COMMON_SUFFIXES]
    @staticmethod
    def __get_word_id(word, W2I):
        w_id = W2I.get(word)
        if w_id is None:
            for suffix in utils.COMMON_SUFFIXES:
                if word.endswith(suffix):
                    return W2I[utils.UNK_Word+suffix]
            w_id = W2I[utils.UNK_Word]
        return w_id
    @staticmethod
    def __sentences_to_ids(sentences, W2I, T2I):
        result = []
        for words, tags in sentences:
            w_ids = [MLETrain.__get_word_id(w,W2I) for w in words]
            t_ids = [T2I[t] for t in tags]
            result.append((w_ids, t_ids))
        return result


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 3:
        print "Wrong number of arguments. Use:\n" + \
              "python MLETrain.py input_filename q_mle_filename e_mle_filename"
        exit()

    input_filename = args[0]
    q_mle_filename = args[1]
    e_mle_filename = args[2]

    MLETrain.createModelFilesFromInput(input_filename, q_mle_filename, e_mle_filename)
