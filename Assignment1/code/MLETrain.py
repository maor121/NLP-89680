import sys
from collections import Counter

import utils
import logging
import numpy as np

class MLETrain:
    __q_counts = None
    __e_counts = None
    __T2I = None
    __W2I = None
    def __init__(self, q_mle_file, e_mle_file):
        self.__T2I, self.__W2I, self.__q_counts, self.__e_counts = \
            utils.read_mle_files(q_mle_filename, e_mle_filename)
    def getQ(self, c, b, a):
        three = self.__get_tag_count([c, b, a]) * 1.0 / self.__get_tag_count([b, a])
        two = self.__get_tag_count([c, b]) * 1.0 / self.__get_tag_count([b])
        one = self.__get_tag_count([c]) * 1.0 / len(self.__T2I)
        return np.average([three, two, one])
    def getE(self, word, tag):
        word_id = self.__W2I.get(word)
        tag_id = self.__T2I.get(tag)
        if (word_id == None or tag_id == None):
            return 0
        word_count = self.__e_counts.get((word_id, tag_id), 0)
        tag_count = self.__get_tag_count([tag])
        return float(word_count) / tag_count
    def __get_tag_count(self, tags):
        tags_ids = sorted(filter(None, [self.__T2I.get(t) for t in tags]))
        if len(tags_ids) != len(tags):
            return 0
        return self.__q_counts.get(tuple(tags_ids), 0)
    @staticmethod
    def createModelFilesFromInput(input_filename, q_mle_filename, e_mle_filename):
        logging.basicConfig()
        log = logging.getLogger()
        log.setLevel(logging.DEBUG)

        log.debug("Reading input file")
        train_data = utils.read_input_file(input_filename, is_tagged=True)

        log.debug("- Converting words\\tags to ids")
        from utils import list_to_ids, reduce_tuple_list, flatten
        W2I = list_to_ids(flatten(reduce_tuple_list(train_data, dim=0)))
        T2I = list_to_ids(flatten(reduce_tuple_list(train_data, dim=1)))
        train_data_ids = utils.sentences_to_ids(train_data, W2I, T2I)
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
            #Q
            count_tag_triplets.update(utils.count_triplets(tags_ids))
            count_tag_pairs.update(utils.count_pairs(tags_ids))
            count_tag_single.update(utils.count_single(tags_ids))
            #E
            count_word_tags.update(utils.count_word_tags(words_ids, tags_ids))

        log.debug("Writing to file {}".format(q_mle_filename))
        utils.write_q_mle_file(count_tag_triplets, count_tag_pairs, count_tag_single, I2T, q_mle_filename)

        log.debug("Writing to file {}".format(e_mle_filename))
        utils.write_e_mle_file(count_word_tags, I2T, I2W, e_mle_filename)

        log.debug("Done")

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

    #Testing
    """
    model = MLETrain(q_mle_filename, e_mle_filename)
    print(model.getQ("DT","JJR",":"))
    print(model.getE("Law", "NN"))
    print(model.getQ("NNP", utils.START_TAG, utils.START_TAG))
    print(model.getQ("DT", "JJR", utils.START_TAG))
    """