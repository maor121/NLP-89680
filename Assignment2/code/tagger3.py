import numpy as np
from torch.utils.data import TensorDataset
import sys

from utils import StringCounter
from model_runner import ModelRunner
from model import load_dataset
import torch.utils.data

def parse_arg_bool(val):
    if val not in ("True", "False"):
        print "bool arg is either True or False. got {}".format(val)
        exit()
    return val == "True"

def parse_arg_eval_mode(val, allowed_values):
    if val not in allowed_values:
        print "eval_mode not in {}, got {}".format(allowed_values, val)
        exit()
    return val


if __name__ == '__main__':
    sys.argv = sys.argv[1:]
    wrongArgLenMsg = "Wrong number of arguments, Usage:\n"+\
        "tagger3.py is_pretrained(True\\False) is_cuda train_file test_file is_ner number_of_epoches eval_mode(blind\\everyepoch\\plot) [vocab_file] [wordVectors_file] [prediction_out_file]\n"+\
        "Note: vocab_file & wordVectors_file are needed only in pretrained mode."

    if len(sys.argv) not in (6,7,8,9):
        print wrongArgLenMsg
        exit()
    is_pretrained = parse_arg_bool(sys.argv[0])
    is_cuda = parse_arg_bool(sys.argv[1])
    train_filename = sys.argv[2]
    test_filename = sys.argv[3]
    is_ner = parse_arg_bool(sys.argv[4])  # Used for eval
    epoches = int(sys.argv[5])
    eval_mode = parse_arg_eval_mode(sys.argv[6], ["blind", "everyepoch", "plot"])
    if (is_pretrained and len(sys.argv) == 7) or (not is_pretrained and len(sys.argv) == 9):
        prediction_out_filename = sys.argv[-1] #always last
    else:
        prediction_out_filename = None


    UNK_WORD = "UUUNKKK"; START_WORD = "<s>"; END_WORD = "</s>"

    if is_pretrained:
        vocab_filename = sys.argv[7]
        word_vectors_filename = sys.argv[8]

        vocab = np.loadtxt(vocab_filename, dtype=object, comments=None)
        embeds = np.loadtxt(word_vectors_filename, dtype=np.float32)
        assert len(vocab) == len(embeds)
        embed_depth = embeds.shape[1]

        lower_case=True
        replace_numbers=False
        W2I = StringCounter(vocab, UNK_WORD)
    else:
        embed_depth = 50

        W2I = None
        lower_case = False
        replace_numbers = True

    window_size = 2
    learning_rate = 0.001
    batch_size = 500

    W2I, T2I, F2I, train_words, train_labels = load_dataset(train_filename, window_size, W2I=W2I,
                                                UNK_WORD=UNK_WORD, START_WORD=START_WORD, END_WORD=END_WORD,
                                                lower_case=lower_case, replace_numbers=replace_numbers,
                                                calc_sub_word=True)
    __, __, __, test_words, test_labels = load_dataset(test_filename, window_size, W2I=W2I, T2I=T2I, F2I=F2I,
                                                UNK_WORD=UNK_WORD, START_WORD=START_WORD, END_WORD=END_WORD,
                                                lower_case=lower_case, replace_numbers=replace_numbers,
                                                calc_sub_word=True)

    num_words = W2I.len()
    num_tags = T2I.len()
    num_features = F2I.len()
    omit_tag_id = T2I.get_id('O') if is_ner else None

    trainset = TensorDataset(train_words, train_labels)
    testset = TensorDataset(test_words, test_labels)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)

    del W2I, train_words, train_labels, test_words, test_labels, trainset, testset
    import gc
    gc.collect()

    runner = ModelRunner(window_size, learning_rate, is_cuda)
    if is_pretrained:
        runner.initialize_pretrained(num_tags, embeds, num_features)
    else:
        runner.initialize_random(num_words,num_tags,embed_depth,num_features)
    runner.train_and_eval(trainloader, epoches, testloader, omit_tag_id, eval_mode=eval_mode)

    if prediction_out_filename:
        # IMPORTANT: testloader must have shuffle false, to have same order as test_filename
        runner.write_prediction(testloader, T2I, test_filename, prediction_out_filename)

    print('Finished Training')