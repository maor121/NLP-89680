import sys

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

def count_tags(train_data):
    from collections import Counter
    tags = train_data[1]
    return len(Counter(tags).keys())

def list_to_ids(L):
    from collections import Counter
    L2I = {t: i for i, t in enumerate(Counter(L).keys())}
    return L2I

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
    tags_count = count_tags(train_data)

    W2I = list_to_ids(train_data[0])
    T2I = list_to_ids(train_data[1])

