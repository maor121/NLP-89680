class StringCounter:
    def __init__(self, initialStrList=[], UNK_WORD=None):
        from collections import Counter
        self.S2I = {}
        self.S2C = Counter()
        self.last_id = 0
        self.UNK_WORD = UNK_WORD
        for s in initialStrList:
            self.get_id_and_update(s)
        if UNK_WORD is not None:
            self.get_id_and_update(UNK_WORD)

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


def inverse_dict(dict):
    return {v: k for k, v in dict.iteritems()}
