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
    vocab_filename = "../data/pretrained/vocab.txt"                 #sys.argv[0]
    word_vectors_filename = "../data/pretrained/wordVectors.txt"    #sys.argv[1]

    vocab = np.loadtxt(vocab_filename, dtype=object, comments=None)
    embeds = np.loadtxt(word_vectors_filename, dtype=np.float32)
    assert len(vocab) == len(embeds)

    UNK_WORD = "UUUNKKK"
    START_WORD = END_WORD = "start/finish"
    embed_depth = embeds.shape[1]

    train_filename = "../data/pos/train"
    test_filename = "../data/pos/dev"
    is_ner = False #Used for eval

    is_cuda = False
    window_size = 2
    learning_rate = 0.001
    batch_size = 1000
    epoches = 4

    W2I = StringCounter(vocab, UNK_WORD)

    __, T2I, F2I, train_words, train_labels = load_dataset(train_filename, window_size, W2I=W2I,
                                                UNK_WORD=UNK_WORD,
                                                START_WORD=START_WORD,
                                                END_WORD=END_WORD,
                                                lower_case=True, replace_numbers=False,
                                                calc_sub_word=True)
    __, __, __, test_words, test_labels = load_dataset(test_filename, window_size, W2I=W2I, T2I=T2I, F2I=F2I,
                                             UNK_WORD=UNK_WORD,
                                             START_WORD=START_WORD,
                                             END_WORD=END_WORD,
                                             lower_case=True, replace_numbers=False,
                                                calc_sub_word=True)

    num_words = W2I.len()
    num_tags = T2I.len()
    num_features = F2I.len()
    omit_tag_id = T2I.get_id('o') if is_ner else None

    train = torch.cat([torch.LongTensor(train_features), torch.unsqueeze(torch.LongTensor(train_words), 2)], dim=2)
    test = torch.cat([torch.LongTensor(test_features), torch.unsqueeze(torch.LongTensor(test_words), 2)], dim=2)

    trainset = TensorDataset(train, torch.LongTensor(train_labels))
    testset = TensorDataset(test, torch.LongTensor(test_labels))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=4)

    del W2I, T2I, train, train_labels, test, test_labels, trainset, testset
    import gc
    gc.collect()

    runner = ModelRunner(window_size, learning_rate, is_cuda)
    runner.initialize_pretrained(num_tags, num_features, embeds)
    runner.train(trainloader, epoches)

    runner.eval(testloader, omit_tag_id)

    print('Finished Training')