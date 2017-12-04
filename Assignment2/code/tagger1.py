import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset
import re
import time

UNK_WORD = "*UNK*"
START_WORD = "*START*"
END_WORD = "*END*"
RARE_WORDS_MAX_COUNT = 3
DIGIT_PATTERN = re.compile('\d')


def inverse_dict(dict):
    return {v: k for k, v in dict.iteritems()}

def windows_from_sentence(sentence_ids, window_size, w_start_id, w_end_id):
    sentence_ids = [w_start_id]*window_size + sentence_ids + [w_end_id]*window_size
    w_windows = []
    for window in list_to_tuples(sentence_ids, window_size * 2 + 1):
        w_windows.append(window)
    return w_windows

def load_dataset(path, window_size=2, is_train=True, W2I=None, T2I=None):
    if is_train:
        W2I = StringCounter([START_WORD, END_WORD, UNK_WORD])
        T2I = StringCounter([])

    w_start_id = W2I.get_id(START_WORD)
    w_end_id = W2I.get_id(END_WORD)

    sentence_ids = []
    words_ids = []
    tags_ids = []
    is_end_sentence = False
    with open(path) as data_file:
        for line in data_file:
            line = re.sub(DIGIT_PATTERN,'#', line.strip())
            if len(line) > 0:
                if is_end_sentence:
                    words_ids.extend(windows_from_sentence(sentence_ids, window_size, w_start_id, w_end_id))
                    sentence_ids = []
                    is_end_sentence = False
                w, t = line.split()
                if is_train:
                    w_id = W2I.get_id_and_update(w)
                    t_id = T2I.get_id_and_update(t)
                else:
                    w_id = W2I.get_id(w)
                    t_id = T2I.get_id(t)
                sentence_ids.append(w_id)
                tags_ids.append(t_id)
            else:
                is_end_sentence = True
    words_ids.extend(windows_from_sentence(sentence_ids, window_size, w_start_id, w_end_id))

    # Filter rare words from dataset
    if is_train:
        W2I.filter_rare_words(RARE_WORDS_MAX_COUNT+1)
        W2I_2 = StringCounter(W2I.S2I.keys())
        I2W = inverse_dict(W2I.S2I)
        words_ids_2 = []
        for window in words_ids:
            words_ids_2.append(tuple([W2I_2.S2I[I2W.get(w_id,UNK_WORD)] for w_id in window]))
        W2I = W2I_2
        words_ids = words_ids_2

    assert len(words_ids)==len(tags_ids)
    return W2I, T2I, words_ids, tags_ids


def list_to_tuples(L, tup_size):
    "s -> (s0,s1,s2), (s1,s2,s3), (s2, s3,s4), ..."
    from itertools import tee, izip
    tupItr = tee(L, tup_size)
    for i, itr in enumerate(tupItr):
        for j in range(i):
            next(itr, None)
    return izip(*tupItr)


class StringCounter:
    def __init__(self, strlist):
        from collections import Counter
        self.S2I = {}
        self.S2C = Counter()
        self.last_id = 0
        for s in strlist:
            self.get_id_and_update(s)

    def get_id_and_update(self, str):
        if not self.S2I.__contains__(str):
            self.S2I[str] = self.last_id
            self.last_id += 1
        self.S2C.update(str)
        return self.S2I[str]
    def get_id(self, str):
        if not self.S2I.__contains__(str):
            str = UNK_WORD
        return self.S2I[str]
    def filter_rare_words(self, min_count):
        w_to_filter = [k for k,v in self.S2C.iteritems() if v<min_count]
        for w in w_to_filter:
            self.S2C.pop(w)
            self.S2I.pop(w)


class Net(nn.Module):
    def __init__(self, W2I, T2I, embed_depth, window_size):
        super(Net, self).__init__()

        unk_id = W2I.S2I[UNK_WORD]
        num_words = len(W2I.S2I)
        num_tags = len(T2I.S2I)
        self.embed_depth=embed_depth
        self.window_size=window_size
        # an Embedding module containing 10 tensors of size 3
        self.embed1 = nn.Embedding(num_words, embed_depth, padding_idx=unk_id, sparse=False)
        self.fc1 = nn.Linear(embed_depth*(window_size*2+1), num_tags*2)
        self.fc2 = nn.Linear(num_tags*2, num_tags)

    def forward(self, x):
        x = self.embed1(x)
        #x = torch.cat(x, dim=0)
        x = x.view(-1, self.embed_depth*(self.window_size*2+1))
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    import torch.optim as optim
    import torch.utils.data

    window_size = 2
    embedding_depth = 50
    batch_size = 1000
    epoches = 8

    W2I, T2I, train, train_labels = load_dataset("../data/pos/train", window_size)
    __, __, test, test_labels = load_dataset("../data/pos/dev", window_size, is_train=False, W2I=W2I, T2I=T2I)

    net = Net(W2I, T2I, embedding_depth, window_size)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr = 0.001)

    trainset = TensorDataset(torch.LongTensor(train), torch.LongTensor(train_labels))
    testset = TensorDataset(torch.LongTensor(test), torch.LongTensor(test_labels))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=4)

    for epoch in range(epoches):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            start_t = time.time()

            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            end_t = time.time()

            # print statistics
            running_loss += loss.data[0]
            if i % 50 == 49:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f timer_per_batch: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50, (end_t-start_t)))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    for data in testloader:
        features, labels = data
        outputs = net(Variable(features))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the %d test words: %d %%' % (
        total, 100 * correct / total))
