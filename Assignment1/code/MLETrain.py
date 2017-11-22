import sys
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
        two = self.__get_tag_count([c, b]) * 1.0 / self.__get_tag_count(b)
        one = self.__get_tag_count([c]) * 1.0 / len(self.__T2I)
        return np.average(three, two, one)
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
        train_data = utils.read_input_file(input_filename)

        log.debug("- Converting words\\tags to ids")
        W2I = utils.list_to_ids(train_data[0])
        T2I = utils.list_to_ids(train_data[1])
        tags_ids = [T2I[t] for t in train_data[1]]
        word_ids = [W2I[w] for w in train_data[0]]
        # Inverse dictionary
        I2T = utils.inverse_dict(T2I)
        I2W = utils.inverse_dict(W2I)

        log.debug("- Counting tags: triplets"),
        count_tag_triplets = utils.count_triplets(tags_ids)
        log.debug("... pairs"),
        count_tag_pairs = utils.count_pairs(tags_ids)
        log.debug("... singles")
        count_tag_single = utils.count_single(tags_ids)

        log.debug("Writing to file {}".format(q_mle_filename))
        utils.write_q_mle_file(count_tag_triplets, count_tag_pairs, count_tag_single, I2T, q_mle_filename)

        log.debug("- Counting tags per word")
        count_word_tags = utils.count_word_tags(word_ids, tags_ids)
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

    #MLETrain.createModelFilesFromInput(input_filename, q_mle_filename, e_mle_filename)

    model = MLETrain(q_mle_filename, e_mle_filename)
    print(model.getQ("DT","JJR",":"))