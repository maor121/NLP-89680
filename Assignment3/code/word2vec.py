import preprocess
import numpy as np
import utils

def load_from_files(words_filename, contexts_filename):
    W2I = preprocess.StringCounter()
    C2I = preprocess.StringCounter()
    words = load_arr_from_file(W2I, words_filename)

    contexts = load_arr_from_file(C2I, contexts_filename)

    return W2I, C2I, words, contexts


def load_arr_from_file(W2I, filename):
    with open(filename, 'r') as file:
        width = 0
        for line in file:
            width = len(line.split())-1
            break
        height = 1 # Continue where the last for left
        for line in file:
            height += 1

    words_arr = np.ndarray(shape=(height, width), dtype=np.float32)

    row = 0
    with open(filename, 'r') as file:
        words_ids = []
        for line in file:
            line_arr = line.split()
            w_id = W2I.get_id_and_update(line_arr[0])
            words_arr[row] = line_arr[1:]
            words_ids.append(w_id)
            row += 1

    return words_arr


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
    W2I, C2I, words, contexts = load_from_files("../data/word2vec/bow5/bow5.words","../data/word2vec/bow5/bow5.contexts")
    #W2I, C2I, words, contexts = load_from_files("../data/word2vec/deps/deps.words","../data/word2vec/deps/deps.contexts")


    k = 20

    I2W = utils.inverse_dict(W2I.S2I)
    I2C = utils.inverse_dict(C2I.S2I)

    target_words = ["car", "bus", "hospital", "hotel", "gun", "bomb", "horse", "fox", "table", "bowl", "guitar", "piano"]

    # First order
    # Find top k context features for each target word
    # top features is highest dot product (word, context_word)
    print("1st order")
    dwc = DotWithCache()
    for from_w in target_words:
        from_w_id = W2I.get_id(from_w)
        from_vec = words[from_w_id,:]
        all_dist = {I2C[i]:dwc.dotWithCache(from_w, I2C[i], from_vec, contexts[i,:]) for i in range(contexts.shape[0])}
        all_dist_tup = sorted(all_dist.items(), key=lambda x: x[1], reverse=True)

        #all_dist_tup = [(w, "%.3f" % d) for w,d in all_dist_tup] # Round results
        all_dist_tup = [w for w,d in all_dist_tup]                # Remove score
        print from_w, all_dist_tup[1:k+1]

    print("2nd order")
    del dwc
    dwc = DotWithCache()
    for from_w in target_words:
        from_w_id = W2I.get_id(from_w)
        from_vec = words[from_w_id,:]
        all_dist = {I2W[i]:dwc.dist(from_w, I2W[i], from_vec, words[i,:]) for i in range(words.shape[0])}
        all_dist_tup = sorted(all_dist.items(), key=lambda x: x[1], reverse=True)

        #all_dist_tup = [(w, "%.3f" % d) for w,d in all_dist_tup] # Round results
        all_dist_tup = [w for w,d in all_dist_tup]                # Remove score
        print from_w, all_dist_tup[1:k+1]