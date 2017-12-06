import re
from utils import StringCounter, list_to_tuples, inverse_dict
import torch
import torch.nn as nn
import torch.nn.functional as F

RARE_WORDS_MAX_COUNT = 3
RARE_FEATURE_MAX_COUNT = 3

FEATURES_PER_WORD = 2

DIGIT_PATTERN = re.compile('\d')

def scan_input_file(path, W2I=None, T2I=None, F2I=None,
                    UNK_WORD="*UNK*", START_WORD="*START*", END_WORD="*END*",
                    lower_case=False, replace_numbers=True, calc_sub_word=False):
    calc_W = W2I == None
    calc_T = T2I == None
    calc_F = F2I == None and calc_sub_word
    if calc_W:
        W2I = StringCounter([START_WORD, END_WORD, UNK_WORD], UNK_WORD)
    else:
        # Pretrained
        W2I.get_id_and_update(START_WORD)
        W2I.get_id_and_update(END_WORD)
    if calc_T:
        T2I = StringCounter()
    if calc_F:
        F2I = StringCounter([UNK_WORD], UNK_WORD=UNK_WORD)

    num_words = 0
    with open(path) as data_file:
        for line in data_file:
            if replace_numbers:
                line = re.sub(DIGIT_PATTERN, '#', line.strip())
            if lower_case:
                line = line.strip().lower()
            if len(line) > 0: # Not end of sentence
                w, t = line.split()
                if calc_W:
                    W2I.get_id_and_update(w)
                if calc_T:
                    T2I.get_id_and_update(t)
                if calc_F:
                    extract_features(w, F2I, updateF2I=True)
                num_words += 1

    if calc_W:
        W2I.filter_rare_words(RARE_WORDS_MAX_COUNT+1)
        W2I = StringCounter(W2I.S2I.keys(), W2I.UNK_WORD)
    if calc_F:
        F2I.filter_rare_words(RARE_FEATURE_MAX_COUNT+1)
        F2I = StringCounter(F2I.S2I.keys(), F2I.UNK_WORD)

    return num_words, W2I, T2I, F2I

def extract_features(word, F2I, updateF2I):
    """Return a torch.LongTensor of features per word."""

    prefix_3 = word[:3]  # Will work even for words of size < 3
    suffix_3 = word[-3:]
    if updateF2I:
        prefix_3_id = F2I.get_id_and_update(prefix_3)
        suffix_3_id = F2I.get_id_and_update(suffix_3)
    else:
        prefix_3_id = F2I.get_id(prefix_3)
        suffix_3_id = F2I.get_id(suffix_3)
    return [prefix_3_id, suffix_3_id]

def windows_from_sentence(sentence, window_size):
    w_windows = []
    for window in list_to_tuples(sentence, window_size * 2 + 1):
        w_windows.append(window)
    return torch.LongTensor(w_windows)


def load_dataset(path, window_size=2, W2I=None, T2I=None, F2I=None,
                 UNK_WORD="*UNK*", START_WORD="*START*", END_WORD="*END*",
                 lower_case=False, replace_numbers=True, calc_sub_word=False):
    num_words, W2I, T2I, F2I = scan_input_file(path, W2I=W2I, T2I=T2I, F2I=F2I,
                                    UNK_WORD=UNK_WORD, START_WORD=START_WORD, END_WORD=END_WORD,
                                    lower_case=lower_case, replace_numbers=replace_numbers,
                                    calc_sub_word=calc_sub_word)

    train_w_depth = FEATURES_PER_WORD+1 if calc_sub_word else 1
    input_tensor = torch.LongTensor(num_words, window_size*2+1, train_w_depth)
    labels_tensor = torch.LongTensor(num_words)

    sentence = []
    word_index = 0
    saw_empty_line = True
    with open(path) as data_file:
        for line in data_file:
            if replace_numbers:
                line = re.sub(DIGIT_PATTERN, '#', line.strip())
            if lower_case:
                line = line.strip().lower()
            if len(line) > 0:
                w, t = line.split()
                sentence.append(w)
                labels_tensor[word_index] = T2I.get_id(t)
                word_index += 1
                saw_empty_line = False
            else:
                if not saw_empty_line: # END of sentence
                    sentence_len = len(sentence)
                    sentence = [START_WORD]*window_size + sentence + [END_WORD]*window_size
                    sentence_ids = [W2I.get_id(w) for w in sentence]

                    input_tensor[word_index-sentence_len:word_index,:,0] = windows_from_sentence(sentence_ids, window_size)

                    if F2I is not None:
                        features_ids = [extract_features(w, F2I, updateF2I=False) for w in sentence]

                        input_tensor[word_index-sentence_len:word_index,:,1:] = windows_from_sentence(features_ids, window_size)
                saw_empty_line = True

    return W2I, T2I, F2I, input_tensor, labels_tensor


class Model(nn.Module):
    def __init__(self, num_words, num_tags, embed_depth, window_size, sub_words, num_features):
        super(Model, self).__init__()
        self.embed_depth = embed_depth
        self.window_size = window_size
        self.sub_words=sub_words

        if self.sub_words:
            self.embed1 = nn.Embedding(num_words+num_features, embed_depth)
        else:
            self.embed1 = nn.Embedding(num_words,embed_depth)
        self.norm1 = nn.BatchNorm1d(embed_depth * (window_size * 2 + 1))
        self.fc1 = nn.Linear(embed_depth * (window_size * 2 + 1), num_tags * 4)
        self.fc2 = nn.Linear(num_tags * 4, num_tags)

    def forward(self, x):
        x = self.embed1(x)
        x = x.view(-1, self.embed_depth * (self.window_size * 2 + 1))
        x = self.norm1(x)
        x = F.tanh(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    @classmethod
    def pretrained(cls, num_tags, window_size, embeddings,sub_words, num_features):
        num_words = embeddings.shape[0]
        embed_depth = embeddings.shape[1]
        model = cls(num_words, num_tags, embed_depth, window_size,sub_words, num_features)
        model.embed1.weight = nn.Parameter(torch.from_numpy(embeddings))
        return model
