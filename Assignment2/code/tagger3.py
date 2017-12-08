import numpy as np
from torch.utils.data import TensorDataset

from utils import StringCounter
from model_runner import ModelRunner
from model import load_dataset
import torch.utils.data

if __name__ == '__main__':
    #sys.argv = sys.argv[1:]
    #if len(sys.argv) != 2:
    #    print("Wrong number of parameters, Usage:\n" + \
    #          "python tagger2.py vocab.txt wordVectors.txt")
    #vocab_filename = "../data/pretrained/vocab.txt"                 #sys.argv[0]
    #word_vectors_filename = "../data/pretrained/wordVectors.txt"    #sys.argv[1]

    #vocab = np.loadtxt(vocab_filename, dtype=object, comments=None)
    #embeds = np.loadtxt(word_vectors_filename, dtype=np.float32)
    #assert len(vocab) == len(embeds)

    #UNK_WORD = "UUUNKKK"; START_WORD = "<s>"; END_WORD = "</s>"
    embed_depth = 50 # embeds.shape[1]

    train_filename = "../data/pos/train"
    test_filename = "../data/pos/dev"
    is_ner = False #Used for eval

    is_cuda = True
    window_size = 2
    learning_rate = 0.001
    batch_size = 500
    epoches = 7

    #W2I = StringCounter(vocab, UNK_WORD)

    W2I, T2I, F2I, train_words, train_labels = load_dataset(train_filename, window_size,
                                                calc_sub_word=True)
    __, __, __, test_words, test_labels = load_dataset(test_filename, window_size, W2I=W2I, T2I=T2I, F2I=F2I,
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
                                             shuffle=True, num_workers=4)

    del W2I, T2I, train_words, train_labels, test_words, test_labels, trainset, testset
    import gc
    gc.collect()

    runner = ModelRunner(window_size, learning_rate, is_cuda)
    #runner.initialize_pretrained(num_tags, embeds, num_features)
    runner.initialize_random(num_words,num_tags,embed_depth,num_features)
    runner.train_and_eval(trainloader, epoches, testloader, omit_tag_id, eval_mode="plot")

    print('Finished Training')