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
                    word, tag = w_t.rsplit("/",1)
                    words.append(word)
                    tags.append(tag)
        return [words, tags]
    except Exception as e:
        raise

def write_q_mle_file(count_tags_triplets, count_tags_pairs, count_tags_single, I2T, q_mle_filename):
    try:
        with open(q_mle_filename, 'w+') as f: #Overwrite file if exists
            for triplet, count in count_tags_triplets:
                i1, i2, i3 = triplet
                t1, t2, t3 = I2T[i1], I2T[i2], I2T[i3]
                line = "{} {} {}\t{}\n".format(t1,t2,t3,count)
                f.write(line)

            for pair, count in count_tags_pairs:
                i1, i2 = pair
                t1, t2 = I2T[i1], I2T[i2]
                line = "{} {}\t{}\n".format(t1,t2,count)
                f.write(line)

            for i1, count in count_tags_single:
                t1 = I2T[i1]
                line = "{}\t{}\n".format(t1,count)
                f.write(line)
    except Exception:
        raise


def list_to_ids(L):
    from collections import Counter
    L2I = {t: i for i, t in enumerate(Counter(L).keys())}
    return L2I

def count_triplets(tags_ids):
    from collections import Counter
    count_y1_y2_y3 = Counter()
    for i1, i2, i3 in triplets(tags_ids):
        count_y1_y2_y3.update([frozenset([i1,i2,i3])]) #set, order does not matter for hashing
    return count_y1_y2_y3

def count_pairs(tags_ids):
    from collections import Counter
    count_y1_y2 = Counter()
    for i1, i2, i3 in triplets(tags_ids):
        count_y1_y2.update([frozenset([i1, i2])])  # set, order does not matter for hashing
    return count_y1_y2

def count_single(tags_ids):
    from collections import Counter
    return Counter(tags_ids)

def triplets(iterable):
    "s -> (s0,s1,s2), (s1,s2,s3), (s2, s3,s4), ..."
    from itertools import tee, izip
    a, b, c = tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return izip(a, b, c)

if __name__ == '__main__':
    from collections import Counter
    c = Counter()
    c.update([frozenset([1,2])])
    c.update([frozenset([2,1])])

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
    tags_ids = [T2I[t] for t in train_data[1]]

    count_tag_triplets = count_triplets(tags_ids)
    count_tag_pairs = count_pairs(tags_ids)
    count_tag_single = count_single(tags_ids)

    #Inverse dictionary
    I2T = {v: k for k, v in T2I.iteritems()}

    write_q_mle_file(count_tag_triplets, count_tag_pairs, count_tag_single, I2T, q_mle_filename)
    print("Done")