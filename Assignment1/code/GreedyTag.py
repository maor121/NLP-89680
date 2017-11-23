from MLETrain import MLETrain
import sys
import utils

class GreedyTag:
    __mletrain = None
    def __init__(self, mletrain):
        self.__mletrain = mletrain
    def getP(self, sentence_words, sentence_predictions):

        #[y0,y1,y2,y3,y4] -> [(y0,y1,y2),(y1,y2,y3),(y2,y3,y4)]
        predictions_triplets = utils.triplets(sentence_predictions)
        pass

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


