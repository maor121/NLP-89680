import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

UNK_WORD = "*UNK*"

def load_dataset(path, window_size=2):
    W2I = StringCounter()
    T2I = StringCounter()
    words_ids = []
    tags_ids = []
    with open(path) as data_file:
        for line in data_file:
            line = line.strip()
            if len(line) > 0:
                w, t = line.split()
                w_id = W2I.get_id_and_update(w)
                t_id = T2I.get_id_and_update(t)
                words_ids.append(w_id)
                tags_ids.append(t_id)

    unk_id = W2I.get_id_and_update(UNK_WORD)
    words_ids = [unk_id]*window_size + words_ids + [unk_id]*window_size

    w_windows = []
    for window in list_to_tuples(words_ids, window_size * 2 + 1):
        w_windows.append(window)
    assert len(w_windows)==len(tags_ids)
    return W2I, T2I, w_windows, tags_ids


def list_to_tuples(L, tup_size):
    "s -> (s0,s1,s2), (s1,s2,s3), (s2, s3,s4), ..."
    from itertools import tee, izip
    tupItr = tee(L, tup_size)
    for i, itr in enumerate(tupItr):
        for j in range(i):
            next(itr, None)
    return izip(*tupItr)


class StringCounter:
    def __init__(self):
        self.S2I = {}
        self.last_id = 0

    def get_id_and_update(self, str):
        if not self.S2I.__contains__(str):
            self.S2I[str] = self.last_id
            self.last_id += 1
        return self.S2I[str]


class Net(nn.Module):
    def __init__(self, W2I, T2I, embed_depth, window_size):
        super(Net, self).__init__()

        unk_id = W2I.S2I[UNK_WORD]
        num_words = len(W2I.S2I)
        num_tags = len(T2I.S2I)
        self.embed_depth=embed_depth
        self.window_size=window_size
        # an Embedding module containing 10 tensors of size 3
        self.embed1 = nn.Embedding(num_words, embed_depth, padding_idx=unk_id, sparse=True)
        self.fc1 = nn.Linear(embed_depth*(window_size*2+1), num_tags*2)
        self.fc2 = nn.Linear(num_tags*2, num_tags)
        """
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)"""

    def forward(self, x):
        x = self.embed1(x)
        #x = torch.cat(x, dim=0)
        x = x.view(-1, self.embed_depth*(self.window_size*2+1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        """
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        """
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
    embedding_depth = 20
    batch_size = 10

    W2I, T2I, train, labels = load_dataset("../data/pos/train", window_size)

    net = Net(W2I, T2I, embedding_depth, window_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


    """train_size = len(train)
    for i in range(0, train_size, batch_size):
        train_batch = train[i:i + batch_size]

    batch = train[:10]
    input = Variable(torch.LongTensor(batch))
    out = net(input)
    print(out)

    net.zero_grad()
    out.backward(torch.randn(1, 10))
    """