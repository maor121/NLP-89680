import utils
import re
import numpy as np


def create_feature_vec(w_prev_prev, w_prev, wi, w_next, w_next_next, t_prev, t_prev_prev, W2I, PREFIX_SUFFIX_LEN=4):
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
    features['t_i-1'] = t_prev
    features['t_i-1_t_i-2'] = t_prev + "&" + t_prev_prev
    features = dict((k, v) for k, v in features.iteritems() if v is not None)

    ################################################################################################

    return features


def feature_map_file_to_dict(feature_map_filename):
    feature_dict = {}
    try:
        with open(feature_map_filename, "rb") as map_file:
            for line in map_file:
                key, value = line.split()
                feature_dict[key] = int(value)
    except Exception:
        raise
    return feature_dict


def feature_dict_to_dict_vectorizer(feature_dict):
    # Convert to DictVectorizer
    from sklearn.feature_extraction import DictVectorizer

    DV = DictVectorizer(sparse=True)
    T2I = {}


    feature_names = []
    vocab = {}

    for k, v in feature_dict.iteritems():
        if k not in vocab:
            if not k.__contains__('='):
                T2I[k] = v
            vocab[k] = v
            feature_names.append(k)

    if DV.sort:
        feature_names.sort()

    DV.feature_names_ = feature_names
    DV.vocabulary_ = vocab
    return T2I, DV


def words_and_tags_from_map_dict(feature_map_dict):
    #TODO: Make this method more efficient
    common_words = []
    tags = []

    keys = feature_map_dict.keys()
    for k in keys:
        if k.startswith('w_i='):
            common_words.append(k[4:])
        else:
            if not k.__contains__('='):
                tags.append(k)
    return set(common_words), tags


def fivelets(iterable):
    "s -> (s0,s1,s2,s3,s4), (s1,s2,s3,s4,s5), (s2, s3,s4,s5,s6), ..."
    from itertools import tee, izip
    a, b, c, d, e = tee(iterable, 5)
    next(b, None)
    next(c, None)
    next(c, None)
    next(d, None)
    next(d, None)
    next(d, None)
    next(e, None)
    next(e, None)
    next(e, None)
    next(e, None)
    return izip(a, b, c, d, e)