"""Usage: generate.py [FILE] [-n number]

-h --help    show this
-n number    number of sentences to generate [default: 1]

"""
from docopt import docopt
from collections import defaultdict
import random


class PCFG(object):
    def __init__(self):
        self._rules = defaultdict(list)
        self._sums = defaultdict(float)

    def add_rule(self, lhs, rhs, weight):
        assert(isinstance(lhs, str))
        assert(isinstance(rhs, list))
        self._rules[lhs].append((rhs, weight))
        self._sums[lhs] += weight

    @classmethod
    def from_file(cls, filename):
        grammar = PCFG()
        with open(filename) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line: continue
                w,l,r = line.split(None, 2)
                r = r.split()
                w = float(w)
                grammar.add_rule(l,r,w)
        return grammar

    def is_terminal(self, symbol): return symbol not in self._rules

    def gen(self, symbol):
        if self.is_terminal(symbol): return symbol
        else:
            expansion = self.random_expansion(symbol)
            return " ".join(self.gen(s) for s in expansion)

    def random_sent(self):
        return self.gen("ROOT")

    def random_expansion(self, symbol):
        """
        Generates a random RHS for symbol, in proportion to the weights.
        """
        p = random.random() * self._sums[symbol]
        for r,w in self._rules[symbol]:
            p = p - w
            if p < 0: return r
        return r

    def is_legal(self, words):
        if words == ['ROOT']:
            return True
        matching_rules = PCFG.get_matching_rules(words, self._rules)
        for i, ruleSymbol, ruleLen in matching_rules:
            words_next = words[:i] + [ruleSymbol] + words[i+ruleLen:]
            if self.is_legal(words_next):
                return True
        return False

    @staticmethod
    def get_matching_rules(words, rules):
        matched = []
        for i in range(len(words)):
            for r, r_info in rules.iteritems():
                for r_w, __ in r_info:
                    sub_words_list = words[i:i+len(r_w)]
                    if (sub_words_list == r_w):
                        matched.append((i, r, len(r_w)))
        return matched

def test_sentence(sentence, pcfg):
    print("{}:{}".format(pcfg.is_legal(sentence.strip().split()), sentence))


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Naval Fate 2.0')
    pcfg = PCFG.from_file(arguments['FILE'])
    sentence_count = int(arguments['-n'])

    test_sentence("Sally ate a sandwich .", pcfg)
    test_sentence("Sally and the president wanted and ate a sandwich .", pcfg)
    for i in range(sentence_count):
        print pcfg.random_sent() + '\n'