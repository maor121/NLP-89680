import numpy as np
import sys

class DotWithCache:
    def __init__(self):
        self.squareRoot = {}
        self.dot = {}
    def dist(self, uW, vW, uVec, vVec):
        """Return cosine distance between two word vectors"""

        uSquareRoot = self.sqaureRootWithCache(uW, uVec)
        vSquareRoot = self.sqaureRootWithCache(vW, vVec)
        uvDot = self.dotWithCache(uW, vW, uVec, vVec)
        return uvDot / (uSquareRoot * vSquareRoot)

    def sqaureRootWithCache(self, uW, uVec):
        if not self.squareRoot.__contains__(uW):
            self.squareRoot[uW] = np.sqrt(np.dot(uVec, uVec))
        return self.squareRoot[uW]
    def dotWithCache(self, uW, vW, uVec, vVec):
        if not self.dot.__contains__((uW, vW)):
            self.dot[(uW,vW)] = self.dot[(vW, uW)] = np.dot(uVec, vVec)
        return self.dot[(uW,vW)]

if __name__ == '__main__':
    #sys.argv = sys.argv[1:]
    #if len(sys.argv) != 3:
    #    print("Wrong number of parameters, Usage:\n" + \
    #          "python top_k.py vocab.txt wordVectors.txt fromWords.txt k-top-count")
    vocab_filename = "../data/pretrained/vocab.txt"                 #sys.argv[0]
    word_vectors_filename = "../data/pretrained/wordVectors.txt"    #sys.argv[1]
    from_words_filename = "../data/pretrained/fromWords.txt"        #sys.argv[2]
    k = 5                                                           #int(sys.argv[3])

    words = np.loadtxt(vocab_filename, dtype=object, comments=None)
    vecs = np.loadtxt(word_vectors_filename)
    from_words = np.loadtxt(from_words_filename, dtype=object)
    assert len(words) == len(vecs)

    #Convert to dictionary
    vecByWord = {}
    for w,v in zip(words, vecs):
        vecByWord[w] = v
    del words, vecs

    dwc = DotWithCache()
    for from_w in from_words:
        from_vec = vecByWord[from_w]
        all_dist = {w:dwc.dist(from_w, w, from_vec, v) for w,v in vecByWord.iteritems()}
        all_dist_tup = sorted(all_dist.items(), key=lambda x: x[1], reverse=True)

        print from_w, all_dist_tup[1:k+1]
