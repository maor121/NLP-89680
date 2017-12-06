import re
from utils import StringCounter, list_to_tuples, inverse_dict
import torch
import torch.nn as nn
import torch.nn.functional as F

RARE_WORDS_MAX_COUNT = 3
RARE_FEATURE_MAX_COUNT = 3

DIGIT_PATTERN = re.compile('\d')

def extract_features(word, F2I, updateF2I):
    """Return a list of features per word. Does not have to be uniform in size.
       Because, features embeddings are SUMMED, so we can sum 1 feature embedding to word vector, or 10.
       In this exercise we were asked to have only two features: prefix, suffix of len 3, but since we filter rare features
       (To save running time), the ending feature list might not be uniform in size"""

    prefix_3 = word[:3] # Will work even for words of size < 3
    suffix_3 = word[-3:]
    return [prefix_3, suffix_3]

def filter_rare_features(F2I, f_ids):
    F2I.filter_rare_words(RARE_FEATURE_MAX_COUNT+1)
    F2I_2 = StringCounter(F2I.S2I.keys())
    I2F = inverse_dict(F2I.S2I)
    f_ids_2 = []
    for f_window in f_ids:
        for f_list_per_word in f_window:
            f_ids_2.append(filter(None, [F2I_2.S2I.get(I2F.get(f_id, None), None) for f_id in f_list_per_word]))
    return F2I_2, f_ids_2

def filter_rare_words(W2I, words_ids, UNK_WORD):
    W2I.filter_rare_words(RARE_WORDS_MAX_COUNT + 1)
    W2I_2 = StringCounter(W2I.S2I.keys(), UNK_WORD)
    I2W = inverse_dict(W2I.S2I)
    words_ids_2 = []
    for window in words_ids:
        words_ids_2.append(tuple([W2I_2.S2I[I2W.get(w_id, UNK_WORD)] for w_id in window]))
    return W2I_2, words_ids_2

def windows_from_sentence(sentence_ids, window_size, w_start_id, w_end_id):
    sentence_ids = [w_start_id]*window_size + sentence_ids + [w_end_id]*window_size
    w_windows = []
    for window in list_to_tuples(sentence_ids, window_size * 2 + 1):
        w_windows.append(window)
    return w_windows

def load_dataset(path, window_size=2, W2I=None, T2I=None, F2I=None,
                 UNK_WORD="*UNK*", START_WORD="*START*", END_WORD="*END*",
                 lower_case=False, replace_numbers=True, calc_sub_word=False):
    calc_W = W2I == None
    calc_T = T2I == None
    calc_F = F2I == None and calc_sub_word
    if calc_W:
        W2I = StringCounter([START_WORD, END_WORD, UNK_WORD], UNK_WORD)
    else:
        #Pretrained
        W2I.get_id_and_update(START_WORD)
        W2I.get_id_and_update(END_WORD)
    if calc_T:
        T2I = StringCounter()
    if calc_F:
        F2I = StringCounter()

    w_start_id = W2I.get_id(START_WORD)
    w_end_id = W2I.get_id(END_WORD)

    sentence_ids = []
    words_ids = []
    tags_ids = []
    is_end_sentence = False
    with open(path) as data_file:
        for line in data_file:
            if replace_numbers:
                line = re.sub(DIGIT_PATTERN,'#', line.strip())
            if lower_case:
                line = line.strip().lower()
            if len(line) > 0:
                if is_end_sentence:
                    words_ids.extend(windows_from_sentence(sentence_ids, window_size, w_start_id, w_end_id))
                    sentence_ids = []
                    is_end_sentence = False
                w, t = line.split()
                w_id = W2I.get_id_and_update(w) if calc_W else W2I.get_id(w)
                t_id = T2I.get_id_and_update(t) if calc_T else T2I.get_id(t)
                sentence_ids.append(w_id)
                tags_ids.append(t_id)
            else:
                is_end_sentence = True
    words_ids.extend(windows_from_sentence(sentence_ids, window_size, w_start_id, w_end_id))

    # Calc sub word features
    if F2I is not None:
        I2W = inverse_dict(W2I.S2I)
        f_ids = []
        for window in words_ids:
            f_windpw = []
            for w in window:
                w_features = extract_features(I2W[w], F2I, updateF2I=calc_F)
                f_windpw.append(w_features)
            f_ids.append(f_windpw)
        # Filter rare features
        F2I, f_ids = filter_rare_features(F2I, f_ids)
    else:
        f_ids = None

    # Filter rare words from dataset
    if calc_W:
        W2I, words_ids = filter_rare_words(W2I, words_ids, UNK_WORD)

    assert len(words_ids)==len(tags_ids)

    if F2I is not None:
        return W2I, T2I, words_ids, tags_ids, f_ids
    else:
        return W2I, T2I, words_ids, tags_ids


class Model(nn.Module):
    def __init__(self, num_words, num_tags, embed_depth, window_size):
        super(Model, self).__init__()
        self.embed_depth=embed_depth
        self.window_size=window_size

        self.embed1 = nn.Embedding(num_words, embed_depth)
        self.norm1 = nn.BatchNorm1d(embed_depth*(window_size*2+1))
        self.fc1 = nn.Linear(embed_depth*(window_size*2+1), num_tags*4)
        self.fc2 = nn.Linear(num_tags*4, num_tags)

    def forward(self, x):
        x = self.embed1(x)
        x = x.view(-1, self.embed_depth*(self.window_size*2+1))
        x = self.norm1(x)
        x = F.tanh(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    @classmethod
    def pretrained(cls, num_tags, window_size, embeddings):
        num_words = embeddings.shape[0]
        embed_depth = embeddings.shape[1]
        model = cls(num_words, num_tags, embed_depth, window_size)
        model.embed1.weight = nn.Parameter(torch.from_numpy(embeddings))
        return model