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
            for triplet, count in count_tags_triplets.iteritems():
                i1, i2, i3 = triplet
                t1, t2, t3 = I2T[i1], I2T[i2], I2T[i3]
                line = "{} {} {}\t{}\n".format(t1,t2,t3,count)
                f.write(line)

            for pair, count in count_tags_pairs.iteritems():
                i1, i2 = pair
                t1, t2 = I2T[i1], I2T[i2]
                line = "{} {}\t{}\n".format(t1,t2,count)
                f.write(line)

            for i1, count in count_tags_single.iteritems():
                t1 = I2T[i1]
                line = "{}\t{}\n".format(t1,count)
                f.write(line)
    except Exception:
        raise

def write_e_mle_file(count_word_tags, I2T, I2W, e_mle_filename):
    try:
        with open(e_mle_filename, 'w+') as f:  # Overwrite file if exists
            for (word_id, tag_id), count in count_word_tags.iteritems():
                line = "{} {}\t{}\n".format(I2W[word_id], I2T[tag_id], count)
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
        count_y1_y2_y3.update([tuple(sorted([i1,i2,i3]))]) #allow repeat, order not important
    return count_y1_y2_y3

def count_pairs(tags_ids):
    from collections import Counter
    count_y1_y2 = Counter()
    for i1, i2, i3 in triplets(tags_ids):
        count_y1_y2.update([tuple(sorted([i1, i2]))])  #allow repeat, order not important
    return count_y1_y2

def count_single(tags_ids):
    from collections import Counter
    return Counter(tags_ids)

def count_word_tags(word_ids, tag_ids):
    from collections import Counter
    return Counter(zip(word_ids, tag_ids))

def triplets(iterable):
    "s -> (s0,s1,s2), (s1,s2,s3), (s2, s3,s4), ..."
    from itertools import tee, izip
    a, b, c = tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return izip(a, b, c)