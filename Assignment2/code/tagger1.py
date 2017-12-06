from torch.utils.data import TensorDataset

from model_runner import ModelRunner
from model import load_dataset

if __name__ == '__main__':
    import torch.utils.data

    train_filename = "../data/pos/train"
    test_filename = "../data/pos/dev"
    is_ner = False #Used for eval

    is_cuda = False
    window_size = 2
    embedding_depth = 50
    learning_rate = 0.001
    batch_size = 1000
    epoches = 4

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
    runner.train(trainloader, epoches)

    runner.eval(testloader, omit_tag_id)

    print('Finished Training')

