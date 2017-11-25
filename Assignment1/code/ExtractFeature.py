import sys
import utils

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2:
        print "Wrong number of arguments. Use:\n" + \
                "python ExtractFeature.py corpus_file feature_filename"
        exit()

    corpus_filename = args[0]
    feature_filename = args[1]

    train_data = utils.read_input_file(corpus_filename, is_tagged=True)

    for words, tags in train_data:
        w_itr = iter(words)
        t_itr = iter(tags)

    print "Done"