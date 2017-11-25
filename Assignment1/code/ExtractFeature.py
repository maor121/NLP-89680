import sys
import utils
import re
import numpy as np

VOCAB_SIZE = 15000
PREFIX_SUFFIX_LEN = 4


def extract_features(words, tags, W2I):
    featuresList = []
    w_itr = iter(words)
    t_itr = iter(tags)

    w_prev_prev = w_prev = None
    wi = next(w_itr)
    w_next = next(w_itr, None)
    w_next_next = next(w_itr, None)

    t_prev_prev = next(t_itr)
    t_prev = next(t_itr)
    ti = next(t_itr)
    while True:
        features = {}

        ################################################################################################
        ###Extract features per condition table in paper: http://u.cs.biu.ac.il/~89-680/memm-paper.pdf###
        ################################################################################################

        not_a_rare_word = wi in W2I

        if not_a_rare_word:
            features['w_i'] = wi
        else:
            features['has_number'] = bool(re.search(utils.FLOAT_NUMBER_PATTERN, wi)) or bool(re.search(utils.NUMBER_Word, wi))
            features['has_capital'] = bool(re.search(utils.CAPITAL_PATTERN, wi))
            features['has_hyphen'] = wi.__contains__('-')
            wi_len = len(wi)
            for i in range(np.min([wi_len, PREFIX_SUFFIX_LEN])):
                features['prefix_' + str(i + 1)] = wi[:i + 1]
                features['suffix_' + str(i + 1)] = wi[wi_len - i - 1:]
        # For every wi
        features['w_i+1'] = w_next
        features['w_i+2'] = w_next_next
        features['w_i-1'] = w_prev
        features['w_i-2'] = w_prev_prev
        features['t_i'] = ti
        features['t_i-1'] = t_prev
        features['t_i-1 t_i-2'] = t_prev + " " + t_prev_prev
        features = dict((k, v) for k, v in features.iteritems() if v is not None)

        ################################################################################################

        featuresList.append(features)

        w_prev_prev = w_prev
        w_prev = wi
        wi = w_next
        w_next = w_next_next
        w_next_next = next(w_itr, None)

        t_prev_prev = t_prev
        t_prev = ti
        ti = next(t_itr, None)

        if wi is None or ti is None:
            break

    return featuresList


def writeFeaturesListToFile(featuresList, f):
    for features in featuresList:
        label = features.pop('t_i')
        line = label + ' ' + ' '.join(['{}={}'.format(k,v) for k,v in features.iteritems()]) + '\n'
        f.write(line)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2:
        print "Wrong number of arguments. Use:\n" + \
                "python ExtractFeature.py corpus_file feature_filename"
        exit()

    corpus_filename = args[0]
    feature_filename = args[1]

    train_data = utils.read_input_file(corpus_filename, is_tagged=True, replace_numbers=False)
    from utils import list_to_ids, reduce_tuple_list, flatten

    W2I = list_to_ids(flatten(reduce_tuple_list(train_data, dim=0)), MAX_SIZE=VOCAB_SIZE)

    try:
        with open(feature_filename, "w+") as f:
            sentences_count = len(train_data)
            done_count = 0
            progress = None
            for words, tags in train_data:
                featuresList = extract_features(words, tags, W2I)
                writeFeaturesListToFile(featuresList, f)

                done_count += 1
                progress = utils.progress_hook(done_count, sentences_count, progress)
    except Exception:
        raise

    print "Done"