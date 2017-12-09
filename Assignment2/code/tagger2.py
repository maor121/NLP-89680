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
    if len(sys.argv)  < 8:
        print "Wrong number of arguments, usage:\n" +\
            "tagger2.py is_cuda(True\\False) train_file test_file is_ner(True\\False) number_of_epoches eval_mode(blind\\everyepoch\\plot) vocab_file, wordVectors_file [prediction_out_file]"
        exit()
    is_cuda = parse_arg_bool(sys.argv[0])
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    is_ner = parse_arg_bool(sys.argv[3])  # Used for eval
    epoches = int(sys.argv[4])
    eval_mode = parse_arg_eval_mode(sys.argv[5], ["blind", "everyepoch", "plot"])
    vocab_filename = sys.argv[6]
    word_vectors_filename = sys.argv[7]
    if len(sys.argv) >= 9:
        prediction_out_filename = sys.argv[8]

    vocab = np.loadtxt(vocab_filename, dtype=object, comments=None)
    embeds = np.loadtxt(word_vectors_filename, dtype=np.float32)
    assert len(vocab) == len(embeds)

    UNK_WORD = "UUUNKKK"; START_WORD = "<s>"; END_WORD = "</s>"
    embed_depth = embeds.shape[1]

    window_size = 2
    learning_rate = 0.001
    batch_size = 500

    W2I = StringCounter(vocab, UNK_WORD)

    __, T2I, train, train_labels = load_dataset(train_filename, window_size, W2I=W2I,
                                                UNK_WORD=UNK_WORD,
                                                START_WORD=START_WORD,
                                                END_WORD=END_WORD,
                                                lower_case=True, replace_numbers=False)
    __, __, test, test_labels = load_dataset(test_filename, window_size, W2I=W2I, T2I=T2I,
                                             UNK_WORD=UNK_WORD,
                                             START_WORD=START_WORD,
                                             END_WORD=END_WORD,
                                             lower_case=True, replace_numbers=False)

    num_words = W2I.len()
    num_tags = T2I.len()
    omit_tag_id = T2I.get_id('o') if is_ner else None

    trainset = TensorDataset(torch.LongTensor(train), torch.LongTensor(train_labels))
    testset = TensorDataset(torch.LongTensor(test), torch.LongTensor(test_labels))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=4)

    del W2I, T2I, train, train_labels, test, test_labels, trainset, testset
    import gc
    gc.collect()

    runner = ModelRunner(window_size, learning_rate, is_cuda)
    runner.initialize_pretrained(num_tags, embeds)
    runner.train_and_eval(trainloader, epoches, testloader, omit_tag_id, eval_mode=eval_mode)

    print('Finished Training')