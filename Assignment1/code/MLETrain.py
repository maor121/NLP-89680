import sys
import numpy as np

def read_input_file(input_filename):
    """Return a list of pairs, (word, tag)"""
    words = []
    tags = []
    try:
        with open(input_filename, 'rb') as f:
            for line in f:
                wordsAndTags = line.split()
                for w_t in wordsAndTags:
                    for word, tag in zip(*w_t.split("\\")):
                        words.append(word)
                        tags.append(tag)
        return [words, tags]
    except Exception as e:
        raise

def list_to_ids(L):
    from collections import Counter
    L2I = {t: i for i, t in enumerate(Counter(L).keys())}
    return L2I

def count_triplets(tags, T2I):
    from collections import Counter
    count_y1_y2_y3 = Counter()
    for t1, t2, t3 in triplets(tags):
        i1, i2, i3 = T2I[t1], T2I[t2], T2I[t3]
        count_y1_y2_y3.update((i1,i2,i3)) #tuple, order does not matter for hashing
    return count_y1_y2_y3

def triplets(iterable):
    "s -> (s0,s1,s2), (s1,s2,s3), (s2, s3,s4), ..."
    from itertools import tee, izip
    a, b, c = tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return izip(a, b, c)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 3:
        print "Wrong number of arguments. Use:\n" + \
                "python MLETrain.py input_filename q_mle_filename e_mle_filename"
        exit()

    input_filename = args[0]
    q_mle_filename = args[1]
    e_mle_filename = args[2]

    train_data = read_input_file(input_filename)

    W2I = list_to_ids(train_data[0])
    T2I = list_to_ids(train_data[1])

    count_triplets = count_triplets(train_data[1], T2I)