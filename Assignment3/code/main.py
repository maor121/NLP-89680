import pickle


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

    def save_to_file(self, filename):
        with open(filename, 'w+') as out_file:
            out_file.write("UNK_WORD={}\n".format(self.UNK_WORD))
            for w, w_id in self.S2I:
                out_file.write("{}\t{}\n".format(w,w_id))
    @staticmethod
    def load_from_file(filename):
        with open(filename, 'r') as inp_file:
            line_iter = iter(inp_file)
            UNK_WORD = line_iter.next().strip().split(sep='=')[1]
            words_ids_tuples = []
            while True:
                try:
                    line = line_iter.next()
                    w, w_id = line.split()
                    words_ids_tuples.append((w,w_id))
                except StopIteration:
                    break
            words_ids_tuples = sorted(words_ids_tuples, key=lambda (w,w_id) : w_id)
            words_sorted = [w_id for w, w_id in words_ids_tuples]
            result = StringCounter(words_sorted, UNK_WORD=UNK_WORD)
            return result


def save_obj(obj, filename):
    with open(filename, 'w+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    with open(filename, 'r') as f:
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

    print(W2I.len())
    W2I.filter_rare_words(100)
    W2I = StringCounter(W2I.S2I.keys(), W2I.UNK_WORD)
    print(W2I.len())

    return W2I


def corpus_lemmas_ids_to_context_freq(filename, W2I, UNK_WORD):
    def update_contexts(contexts, sentence):
        for lemma_id in sentence:
            for lemma_id_context in sentence:
                if lemma_id != lemma_id_context:
                    contexts[lemma_id] = contexts.get(lemma_id, 0) + 1
    unk_id = W2I.get_id(UNK_WORD)
    contexts = {}

    with open("../data/" + filename, 'r') as input_file:
        sentence = []
        saw_empty_line = True
        for line in input_file:
            line = line.strip()
            if len(line) > 0:
                w_arr = line.split()  # ID, FORM, LEMMA, CPOSTAG, POSTAG, FEATS, HEAD, DEPREL, PHEAD, PDEPREL
                # we need lemma
                lemma = w_arr[2]
                lemma_id = W2I.get_id(unk_id)
                if lemma_id != unk_id: # Don't count unknown words
                    sentence.append(lemma_id)
            else:
                if not saw_empty_line:
                    update_contexts(contexts, sentence)
                    sentence = []
                saw_empty_line = True
        update_contexts(contexts, sentence)

    return contexts


# DROP: 'JJ','JJR','JJS','NN','NNS','NNP','NNPS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ','WRB'
if __name__ == '__main__':
    import time
    is_tiny = False
    if is_tiny:
        filename = "wikipedia.tinysample.trees.lemmatized"
    else:
        filename = "wikipedia.sample.trees.lemmatized"

    time_s = time.time()
    W2I = corpus_lemmas_to_ids(filename, UNK_WORD="*UNK*")
    contexts = corpus_lemmas_ids_to_context_freq(filename, W2I, UNK_WORD="*UNK*")
    time_e = time.time()

    print("Done. time: %.2f secs" % (time_e-time_s))