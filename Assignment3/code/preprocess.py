import utils


class Preprocess:
    def __init__(self, W2I, contexts):
        self.W2I = W2I
        self.contexts = contexts

    def save_to_file(self, pickle_filename):
        utils.save_obj((self.W2I, self.contexts), pickle_filename)

    @staticmethod
    def from_input(filename, context_mode):
        keep_pos_set = set(
            ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
             'VBZ', 'WRB'])

        W2I = corpus_lemmas_to_ids(filename, UNK_WORD="*UNK*")
        contexts = corpus_lemmas_ids_to_context_freq(filename, W2I, keep_pos_set, "IN", UNK_WORD="*UNK*", min_count=2, context_mode=context_mode)

        return Preprocess(W2I, contexts)

    @staticmethod
    def load_from_file(pickle_filename):
        W2I, contexts = utils.load_obj(pickle_filename)
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


def corpus_lemmas_to_ids(filename, UNK_WORD):
    with open(filename, 'r') as input_file:
        W2I = StringCounter(UNK_WORD=UNK_WORD)
        for line in input_file:
            line = line.strip()
            if len(line) > 0:
                w_arr = line.split()  # ID, FORM, LEMMA, CPOSTAG, POSTAG, FEATS, HEAD, DEPREL, PHEAD, PDEPREL
                # we need lemma
                lemma = w_arr[2]
                W2I.get_id_and_update(lemma)

    W2I.filter_rare_words(100)
    W2I = StringCounter(W2I.S2I.keys(), W2I.UNK_WORD)

    return W2I


def corpus_lemmas_ids_to_context_freq(filename, W2I, keep_pos_set, prep_pos, UNK_WORD, min_count=None, context_mode="sentence"):
    """Context mode: one of three: 1. sentence, 2. window, 3. tree"""
    unk_id = W2I.get_id(UNK_WORD)

    def update_contexts_pair(contexts, (u, v)):
        context_dict = contexts.get(u)
        if not context_dict:
            context_dict = contexts[u] = {}
        context_dict[v] = context_dict.get(v, 0) + 1

    def update_contexts_sentence(contexts, sentence):
        sentence_lemmas = [sentence[id+1][0] for id in range(len(sentence))
                           if sentence[id+1][1] in keep_pos_set and sentence[id+1][0] != unk_id]
        for lemma_id in sentence_lemmas:
            for lemma_id_context in sentence_lemmas:
                if lemma_id != lemma_id_context:
                    update_contexts_pair(contexts, (lemma_id, lemma_id_context))

    def update_contexts_window(contexts, sentence, window_size):
        sentence_lemmas = [sentence[id+1][0] for id in range(len(sentence))
                           if sentence[id+1][1] in keep_pos_set and sentence[id+1][0] != unk_id]
        sentence_lemmas = [None] * window_size + sentence_lemmas + [None] * window_size
        all_windows = windows_from_sentence(sentence_lemmas, window_size)
        for window in all_windows:
            lemma_id = window[window_size]
            for lemma_id_context in window:
                if lemma_id != lemma_id_context and lemma_id_context is not None:
                    update_contexts_pair(contexts, (lemma_id, lemma_id_context))

    W2I_TREE = StringCounter()
    def update_contexts_tree(contexts, sentence):
        for word_id, word in sentence.items():
            # Apply both directions:
            if word[0] != unk_id and (word[1] in keep_pos_set):  # Content word, known
                # 1) go to parent
                current_id = str(word[0]) # lemma_id
                direct_parent_id = word[3]
                addition = ""
                while (direct_parent_id != 0 and sentence[direct_parent_id][0] == unk_id):
                    direct_parent_id = sentence[direct_parent_id][3] # skip unknown words
                if direct_parent_id == 0: # Root
                    parent_id = "*ROOT*"
                else:
                    parent_node = sentence[direct_parent_id]
                    if parent_node[1]==prep_pos: #parent IN
                        addition = "{} {}".format(parent_node[2], str(parent_node[0])) #IN_deprel, IN_lemma_id
                        grandparent_id = parent_node[3]
                        while (grandparent_id != 0 and sentence[grandparent_id][0] == unk_id):
                            grandparent_id = sentence[grandparent_id][3]  # skip unknown words
                        if grandparent_id == 0: # Parent on IN is Root here
                            parent_id = "*ROOT*"
                        else:
                            grandparent_node = sentence[grandparent_id]
                            parent_id = str(grandparent_node[0])
                    else:
                        parent_id = str(parent_node[0])
                current_deprel = word[2]

                up_content_id = W2I_TREE.get_id_and_update(current_id)
                up_feature_id = W2I_TREE.get_id_and_update(" ".join([addition,parent_id,current_deprel, "up"]))
                down_content_id = W2I_TREE.get_id_and_update(parent_id)
                down_feature_id = W2I_TREE.get_id_and_update(" ".join([addition, current_id, current_deprel, "down"]))

                update_contexts_pair(contexts, (up_content_id, up_feature_id))
                update_contexts_pair(contexts, (up_feature_id, up_content_id))
                update_contexts_pair(contexts, (down_content_id, down_feature_id))
                update_contexts_pair(contexts, (down_feature_id, down_content_id))

    contexts = {}

    if context_mode == "sentence":
        update_contexts = lambda contexts, sentence: update_contexts_sentence(contexts, sentence)
    else:
        if context_mode == "window":
            update_contexts = lambda contexts, sentence: update_contexts_window(contexts, sentence, window_size=2)
        else:
            if context_mode == "tree":
                update_contexts = lambda contexts, sentence: update_contexts_tree(contexts, sentence)
            else:
                raise Exception("Unknown context mode")

    with open("../data/" + filename, 'r') as input_file:
        sentence = {} # key: id, value: (lemma_id, pos, parent_id)
        saw_empty_line = True
        for line in input_file:
            line = line.strip()
            if len(line) > 0:
                saw_empty_line = False
                w_arr = line.split()  # ID, FORM, LEMMA, CPOSTAG, POSTAG, FEATS, HEAD, DEPREL, PHEAD, PDEPREL
                # we need lemma
                id = int(w_arr[0])
                lemma = w_arr[2]
                pos = w_arr[3]
                lemma_id = W2I.get_id(lemma)
                head_id = w_arr[6]
                deprel = w_arr[7]
                sentence[id] = (lemma_id, pos, deprel, int(head_id))
            else:
                if not saw_empty_line:
                    update_contexts(contexts, sentence)
                    sentence = {}
                saw_empty_line = True
        update_contexts(contexts, sentence)

    # Filter rare pairs,
    if min_count is not None:
        for w_id, w_context in contexts.items():
            to_filter = [w_id_context for w_id_context, count in w_context.items() if count < min_count]
            for w_id_context in to_filter:
                w_context.pop(w_id_context)
        # Filter words that have no pairs left
        to_filter = []
        for w_id, w_context in contexts.items():
            if len(w_context) == 0:
                to_filter.append(w_id)
        for w_id in to_filter:
            contexts.pop(w_id)

    return W2I_TREE, contexts


def list_to_tuples(L, tup_size):
    "s -> (s0,s1,s2), (s1,s2,s3), (s2, s3,s4), ..."
    from itertools import tee, izip
    tupItr = tee(L, tup_size)
    for i, itr in enumerate(tupItr):
        for j in range(i):
            next(itr, None)
    return izip(*tupItr)


def windows_from_sentence(sentence, window_size):
    w_windows = []
    for window in list_to_tuples(sentence, window_size * 2 + 1):
        w_windows.append(window)
    return w_windows
