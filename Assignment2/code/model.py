import re
from utils import StringCounter, list_to_tuples, inverse_dict
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable

UNK_WORD = "*UNK*"
START_WORD = "*START*"
END_WORD = "*END*"

RARE_WORDS_MAX_COUNT = 3

DIGIT_PATTERN = re.compile('\d')

def windows_from_sentence(sentence_ids, window_size, w_start_id, w_end_id):
    sentence_ids = [w_start_id]*window_size + sentence_ids + [w_end_id]*window_size
    w_windows = []
    for window in list_to_tuples(sentence_ids, window_size * 2 + 1):
        w_windows.append(window)
    return w_windows

def load_dataset(path, window_size=2, is_train=True, W2I=None, T2I=None, F2I=None):
    if is_train:
        W2I = StringCounter([START_WORD, END_WORD, UNK_WORD], UNK_WORD)
        T2I = StringCounter([], None)

    w_start_id = W2I.get_id(START_WORD)
    w_end_id = W2I.get_id(END_WORD)

    sentence_ids = []
    words_ids = []
    tags_ids = []
    is_end_sentence = False
    with open(path) as data_file:
        for line in data_file:
            line = re.sub(DIGIT_PATTERN,'#', line.strip())
            if len(line) > 0:
                if is_end_sentence:
                    if is_train:
                        for i in range(window_size): #Count appearences of START, END
                            W2I.get_id_and_update(START_WORD)
                            W2I.get_id_and_update(END_WORD)
                    words_ids.extend(windows_from_sentence(sentence_ids, window_size, w_start_id, w_end_id))
                    sentence_ids = []
                    is_end_sentence = False
                w, t = line.split()
                if is_train:
                    w_id = W2I.get_id_and_update(w)
                    t_id = T2I.get_id_and_update(t)
                else:
                    w_id = W2I.get_id(w)
                    t_id = T2I.get_id(t)
                sentence_ids.append(w_id)
                tags_ids.append(t_id)
            else:
                is_end_sentence = True
    words_ids.extend(windows_from_sentence(sentence_ids, window_size, w_start_id, w_end_id))

    # Filter rare words from dataset
    if is_train:
        W2I.filter_rare_words(RARE_WORDS_MAX_COUNT+1)
        W2I_2 = StringCounter(W2I.S2I.keys(), UNK_WORD)
        I2W = inverse_dict(W2I.S2I)
        words_ids_2 = []
        for window in words_ids:
            words_ids_2.append(tuple([W2I_2.S2I[I2W.get(w_id,UNK_WORD)] for w_id in window]))
        W2I = W2I_2
        words_ids = words_ids_2
        assert START_WORD in W2I.S2I
        assert END_WORD in W2I.S2I
        assert UNK_WORD in W2I.S2I

    assert len(words_ids)==len(tags_ids)
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
