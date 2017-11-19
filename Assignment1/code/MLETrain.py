import sys

def read_input_file(input_filename):
    """Return a list of pairs, (word, tag)"""
    train_data = []
    try:
        with open(input_filename, 'rb') as f:
            for line in f:
                wordsAndTags = line.split()
                train_data += [tuple(w_t.split("\\")) for w_t in wordsAndTags]
        return train_data
    except Exception as e:
        raise


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

