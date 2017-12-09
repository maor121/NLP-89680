import sys
from torch.utils.data import TensorDataset
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
    if len(sys.argv)  < 6:
        print "Wrong number of arguments, usage:\n" +\
            "tagger1.py is_cuda(True\\False) train_file predict_file is_ner(True\\False) number_of_epoches eval_mode(blind\\everyepoch\\plot)"
        exit()
    else:
        is_cuda = parse_arg_bool(sys.argv[0])
        train_filename = sys.argv[1]
        test_filename = sys.argv[2]
        is_ner = parse_arg_bool(sys.argv[0])  # Used for eval
        epoches = int(sys.argv[4])
        eval_mode = parse_arg_eval_mode(sys.argv[5], ["blind", "everyepoch", "plot"])

    window_size = 2
    embedding_depth = 50
    learning_rate = 0.001
    batch_size = 500

    W2I, T2I, train, train_labels = load_dataset(train_filename, window_size)
    __, __, test, test_labels = load_dataset(test_filename, window_size, W2I=W2I, T2I=T2I)

    num_words = W2I.len()
    num_tags = T2I.len()
    omit_tag_id = T2I.get_id('O') if is_ner else None

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
    runner.initialize_random(num_words, num_tags, embedding_depth)
    runner.train_and_eval(trainloader, epoches, testloader, omit_tag_id, eval_mode=eval_mode)

    print('Finished Training')

