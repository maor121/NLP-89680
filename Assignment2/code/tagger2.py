import numpy as np
from torch.utils.data import TensorDataset

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

    words = np.loadtxt(vocab_filename, dtype=object, comments=None)
    embeds = np.loadtxt(word_vectors_filename)
    assert len(words) == len(embeds)

    train_filename = "../data/pos/train"
    test_filename = "../data/pos/dev"
    is_ner = False #Used for eval

    is_cuda = True
    window_size = 2
    embedding_depth = 50
    learning_rate = 0.001
    batch_size = 1000
    epoches = 8

    W2I, T2I, train, train_labels = load_dataset(train_filename, window_size)
    __, __, test, test_labels = load_dataset(test_filename, window_size, is_train=False, W2I=W2I, T2I=T2I)

    num_words = W2I.len()
    num_tags = T2I.len()

    trainset = TensorDataset(torch.LongTensor(train), torch.LongTensor(train_labels))
    testset = TensorDataset(torch.LongTensor(test), torch.LongTensor(test_labels))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=4)

    runner = ModelRunner(window_size, learning_rate, is_cuda)
    runner.initialize_pretrained(num_tags, embeds)
    runner.train(trainloader, epoches)

    omit_tag_id = T2I['O'] if is_ner else None
    runner.eval(testloader, omit_tag_id)

    print('Finished Training')