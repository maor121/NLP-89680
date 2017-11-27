import numpy as np
import re

UNK_Word = "*UNK*"
START_TAG = "Start"
END_TAG = "End"

#Special word types
FLOAT_NUMBER_PATTERN = re.compile(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
NUMER_WORD_PATTERN = re.compile(r'One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Eleven|Twelve', flags=re.IGNORECASE)
NUMBER_Word = "**Number**"

CAPITAL_PATTERN = re.compile(r'[A-Z]')

COMMON_SUFFIXES= ["ed", "ing"] #Create an unknown UNK word for each

def read_input_file(input_filename, replace_numbers, is_tagged=True):
    """Return a list of pairs, [[(words, tags],[(words,tags)], every pair is a sentence"""
    result = []
    try:
        with open(input_filename, 'rb') as f:
            for line in f:
                sentence = line  # In this assignment each line is a sentence
                if replace_numbers:
                    sentence = FLOAT_NUMBER_PATTERN.sub(NUMBER_Word, sentence)
                    sentence = NUMER_WORD_PATTERN.sub(NUMBER_Word, sentence)
                words = []
                tags = []
                if is_tagged:
                    tags.extend([START_TAG, START_TAG])
                    for w_t in sentence.split():
                        word, tag = w_t.rsplit("/", 1)
                        words.append(word)
                        tags.append(tag)
                        # tags.append(END_TAG)
                else:
                    words += sentence.split()
                result.append((words, tags))
        return result
    except ValueError:
        if is_tagged:
            return read_input_file(input_filename, replace_numbers, is_tagged=False)
        raise

def write_q_mle_file(count_tags_triplets, count_tags_pairs, count_tags_single, I2T, q_mle_filename):
    try:
        with open(q_mle_filename, 'w+') as f:  # Overwrite file if exists
            for triplet, count in count_tags_triplets.iteritems():
                i1, i2, i3 = triplet
                t1, t2, t3 = I2T[i1], I2T[i2], I2T[i3]
                line = "{} {} {}\t{}\n".format(t1, t2, t3, count)
                f.write(line)

            for pair, count in count_tags_pairs.iteritems():
                i1, i2 = pair
                t1, t2 = I2T[i1], I2T[i2]
                line = "{} {}\t{}\n".format(t1, t2, count)
                f.write(line)

            for i1, count in count_tags_single.iteritems():
                t1 = I2T[i1]
                line = "{}\t{}\n".format(t1, count)
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


def read_mle_files(q_mle_filename, e_mle_filename):
    try:
        with open(q_mle_filename, "rb") as qFile, \
                open(e_mle_filename, "rb") as eFile:
            tags = []
            counts = []
            for line in qFile:
                tags_line_count = line.rstrip().split("\t")
                tags_line = tags_line_count[0].split()
                count = int(tags_line_count[1])

                tags.append(tuple(tags_line))
                counts.append(count)
            tags_set = set(sum(tags, ()))
            T2I = list_to_ids(tags_set)
            # Convert tags to ids
            tags = [tuple(sorted([T2I[t] for t in tags_tuple])) for tags_tuple in tags]
            q_counts = {}
            for tags_tuple, count in zip(tags, counts):
                q_counts[tags_tuple] = count

            counts = []
            words = []
            tags = []
            for line in eFile:
                w_t, count = line.rstrip().split("\t")
                word, tag = w_t.split()

                words.append(word)
                tags.append(tag)
                counts.append(count)

            W2I = list_to_ids(words)
            e_counts = {}
            for word, tag, count in zip(words, tags, counts):
                e_counts[(W2I[word], T2I[tag])] = count

            tags = T2I.keys()
            total_word_count = np.sum(q_counts[tuple([T2I[t]])] for t in tags)
            total_word_count -= q_counts[tuple([T2I[START_TAG]])]

            return T2I, W2I, q_counts, e_counts, total_word_count
    except Exception:
        raise


def predict_and_write_to_file(sentences, out_filename, getPredictionFunction):
    try:
        with open(out_filename, "w+") as predict_file:
            done_count = 0
            sentences_count = len(sentences)
            progress = None
            for (words, tags) in sentences:
                prediction = getPredictionFunction(words)

                line = ' '.join('{}/{}'.format(w,t) for w,t in zip(words, prediction))+'\n'
                predict_file.write(line)

                done_count += 1
                progress = progress_hook(done_count, sentences_count, progress)
    except Exception:
        raise


def reduce_tuple_list(L, dim):
    return [l[dim] for l in L]


def flatten(L):
    return [item for sublist in L for item in sublist]


def list_to_ids(L, MAX_SIZE=None, ID_SHIFT=1):
    from collections import Counter
    if MAX_SIZE is None:
        return {t: i + ID_SHIFT for i, t in enumerate(Counter(L).keys())}  # +1 not including 0, 0 is like None for python
    else:
        vocab = set([x for x, c in Counter(L).most_common(MAX_SIZE)])
        return {t: i + ID_SHIFT for i, t in enumerate(vocab)}  # +1 not including 0, 0 is like None for python


def inverse_dict(dict):
    return {v: k for k, v in dict.iteritems()}


def count_triplets(tags_ids):
    from collections import Counter
    count_y1_y2_y3 = Counter()
    for i1, i2, i3 in triplets(tags_ids):
        count_y1_y2_y3.update([tuple([i1, i2, i3])])  # allow repeat, order important
    return count_y1_y2_y3


def count_pairs(tags_ids):
    from collections import Counter
    count_y1_y2 = Counter()
    for i1, i2, i3 in triplets(tags_ids):
        count_y1_y2.update([tuple([i1, i2])])  # allow repeat, order important
    return count_y1_y2


def count_single(tags_ids):
    from collections import Counter
    return Counter(tags_ids)


def count_word_tags(word_ids, tag_ids):
    # Ignore START END tags
    from collections import Counter
    return Counter(zip(word_ids, tag_ids[2:]))


def triplets(iterable):
    "s -> (s0,s1,s2), (s1,s2,s3), (s2, s3,s4), ..."
    from itertools import tee, izip
    a, b, c = tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return izip(a, b, c)


def progress_hook(count, total, last_percent_reported=None):
    import sys

    percent = int(count * 100 / total)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            if (percent == 100):
                sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

    last_percent_reported = percent
    return last_percent_reported
