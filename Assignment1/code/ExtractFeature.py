import sys
import utils
import memm_utils
import re
import numpy as np

VOCAB_SIZE = 15000
PREFIX_SUFFIX_LEN = 4


def extract_features(words, tags, W2I):
    featuresList = []

    words_fivlets = memm_utils.fivelets([None, None] + words + [None, None])
    tags_triplets = utils.triplets(tags)
    for (w_prev_prev, w_prev, wi, w_next, w_next_next), (t_prev_prev, t_prev, ti) in zip(words_fivlets, tags_triplets):
        features = memm_utils.create_feature_vec(w_prev_prev, w_prev, wi, w_next, w_next_next, t_prev, t_prev_prev, W2I)
        features['t_i'] = ti

        featuresList.append(features)

        assert wi is not None and ti is not None

    return featuresList


def writeFeaturesListToFile(featuresList, f):
    for features in featuresList:
        label = features.pop('t_i') # Remove label from feature vec
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