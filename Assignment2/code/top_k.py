import numpy as np
import sys

class DotWithCache:
    def __init__(self):
        self.squareRoot = {}
    def dist(self, uW, vW, uVec, vVec):
        """Return cosine distance between two word vectors"""

        uSquareRoot = self.sqaureRootWithCache(uW, uVec)
        vSquareRoot = self.sqaureRootWithCache(vW, vVec)
        np.dot(uVec,vVec) / (uSquareRoot * vSquareRoot)

    def sqaureRootWithCache(self, uW, uVec):
        if not self.squareRoot.__contains__(uW):
            self.squareRoot[uW] = np.sqrt(np.dot(uVec, uVec))
        return self.squareRoot[uW]

if __name__ == '__main__':
    #sys.argv = sys.argv[1:]
    #if len(sys.argv) != 3:
    #    print("Wrong number of parameters, Usage:\n" + \
    #          "python top_k.py vocab.txt wordVectors.txt fromWords.txt k-top-count")
    vocab_filename = "../data/pretrained/vocab.txt"                 #sys.argv[0]
    word_vectors_filename = "../data/pretrained/wordVectors.txt"    #sys.argv[1]
    from_words_filename = "../data/pretrained/fromWords.txt"        #sys.argv[2]
    k = 5                                                           #int(sys.argv[3])

    words = np.loadtxt(vocab_filename, dtype=object)
    vecs = np.loadtxt(word_vectors_filename)
    from_words = np.loadtxt(from_words_filename, dtype=object)
    assert len(words) == len(vecs)

    #Convert to dictionary
    vecByWord = {}
    for w,v in zip(words, vecs):
        vecByWord[w] = v
    del words, vecs



    print 0
