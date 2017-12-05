from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import TensorDataset
import time
from model import Model, load_dataset

if __name__ == '__main__':
    import torch.optim as optim
    import torch.utils.data

    train_filename = "../data/pos/train"
    test_filename = "../data/pos/dev"
    is_ner = False #Used for eval

    is_cuda = True
    window_size = 2
    embedding_depth = 50
    batch_size = 1000
    epoches = 50

    W2I, T2I, train, train_labels = load_dataset(train_filename, window_size)
    __, __, test, test_labels = load_dataset(test_filename, window_size, is_train=False, W2I=W2I, T2I=T2I)

    net = Model(W2I, T2I, embedding_depth, window_size)
    if (is_cuda):
        #from torch.backends import cudnn
        #cudnn.benchmark = True
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr = 0.001)

    trainset = TensorDataset(torch.LongTensor(train), torch.LongTensor(train_labels))
    testset = TensorDataset(torch.LongTensor(test), torch.LongTensor(test_labels))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=4)

    for epoch in range(epoches):  # loop over the dataset multiple times

        start_e_t = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            start_b_t = time.time()

            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            if is_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            end_b_t = time.time()

            # print statistics
            running_loss += loss.data[0]
            if i % 50 == 49:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f timer_per_batch: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50, (end_b_t-start_b_t)))
                running_loss = 0.0
        end_e_t = time.time()
        print('epoch time: %.3f' % (end_e_t-start_e_t))
        correct = 0
        total = 0
        net.train(False) #Disable dropout during eval mode
        for data in testloader:
            features, labels = data
            input = Variable(features, volatile=True)
            if is_cuda:
                input, labels = input.cuda(), labels.cuda()
            outputs = net(input)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if is_ner:
                O_tag_id = T2I.get_id('O')
                diff_O_tag = sum([1 for p, l in zip(predicted, labels) if p == l and l == O_tag_id])
                correct += (predicted == labels).sum()
                correct -= diff_O_tag
                total -= diff_O_tag
            else:
                correct += (predicted == labels).sum()
        net.train(True) #Resume train mode
        print('Accuracy of the network on the %d test words: %d %%' % (
            total, 100 * correct / total))

    print('Finished Training')

