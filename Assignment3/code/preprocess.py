import pickle


class Preprocess:
    def __init__(self, W2I, contexts):
        self.W2I = W2I
        self.contexts = contexts

    def save_to_file(self, pickle_filename):
        save_obj((self.W2I, self.contexts), pickle_filename)

    @staticmethod
    def from_input(filename):
        drop_pos_set = set(
            ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
             'VBZ', 'WRB'])

        W2I = corpus_lemmas_to_ids(filename, UNK_WORD="*UNK*")
        contexts = corpus_lemmas_ids_to_context_freq(filename, W2I, drop_pos_set, UNK_WORD="*UNK*", min_count=3)

        return Preprocess(W2I, contexts)

    @staticmethod
    def load_from_file(pickle_filename):
        W2I, contexts = load_obj(pickle_filename)
        return Preprocess(W2I, contexts)

class StringCounter:
    def __init__(self, initialStrList=[], UNK_WORD=None):
        from collections import Counter
        self.S2I = {}
        self.S2C = Counter()
        self.last_id = 0
        self.UNK_WORD = UNK_WORD
        for s in initialStrList:
            self.get_id_and_update(s)

    def get_id_and_update(self, str, count=1):
        if not self.S2I.__contains__(str):
            self.S2I[str] = self.last_id
            self.last_id += 1
        self.S2C[str] = self.S2C.get(str, 0) + count
        return self.S2I[str]

    def get_id(self, str):
        if not self.S2I.__contains__(str):
            str = self.UNK_WORD
        return self.S2I[str]

    def filter_rare_words(self, min_count):
        w_to_filter = [k for k, v in self.S2C.iteritems() if v < min_count]
        for w in w_to_filter:
            self.S2C.pop(w)
            self.S2I.pop(w)
        self.get_id_and_update(self.UNK_WORD)

    def len(self):
        return len(self.S2I)

    def shift_ids_by(self, n):
        S2I = {}
        for k, v in self.S2I.iteritems():
            S2I[k] = v + n
        self.S2I = S2I


def save_obj(obj, filename):
    with open(filename, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def corpus_lemmas_to_ids(filename, UNK_WORD):
    with open("../data/"+filename, 'r') as input_file:
        W2I = StringCounter(UNK_WORD=UNK_WORD)
        for line in input_file:
            line = line.strip()
            if len(line) > 0:
                w_arr = line.split() # ID, FORM, LEMMA, CPOSTAG, POSTAG, FEATS, HEAD, DEPREL, PHEAD, PDEPREL
                # we need lemma
                lemma = w_arr[2]
                W2I.get_id_and_update(lemma)

    W2I.filter_rare_words(100)
    W2I = StringCounter(W2I.S2I.keys(), W2I.UNK_WORD)

    return W2I


def corpus_lemmas_ids_to_context_freq(filename, W2I, drop_pos_set, UNK_WORD, min_count=None):
    def update_contexts(contexts, sentence):
        for lemma_id in sentence:
            for lemma_id_context in sentence:
                if lemma_id != lemma_id_context:
                    context_dict = contexts.get(lemma_id)
                    if not context_dict:
                        context_dict = contexts[lemma_id] = {}
                    context_dict[lemma_id_context] = context_dict.get(lemma_id_context, 0) + 1
    unk_id = W2I.get_id(UNK_WORD)
    contexts = {}

    with open("../data/" + filename, 'r') as input_file:
        sentence = []
        saw_empty_line = True
        for line in input_file:
            line = line.strip()
            if len(line) > 0:
                saw_empty_line = False
                w_arr = line.split()  # ID, FORM, LEMMA, CPOSTAG, POSTAG, FEATS, HEAD, DEPREL, PHEAD, PDEPREL
                # we need lemma
                lemma = w_arr[2]
                pos = w_arr[3]
                lemma_id = W2I.get_id(lemma)
                if lemma_id != unk_id and pos not in drop_pos_set: # Don't count unknown words
                    sentence.append(lemma_id)
            else:
                if not saw_empty_line:
                    update_contexts(contexts, sentence)
                    sentence = []
                saw_empty_line = True
        update_contexts(contexts, sentence)

    # Filter rare pairs,
    if min_count is not None:
        for w_id, w_context in contexts.items():
            to_filter = [w_id_context for w_id_context,count in w_context.items() if count < min_count]
            for w_id_context in to_filter:
                w_context.pop(w_id_context)
    return contexts