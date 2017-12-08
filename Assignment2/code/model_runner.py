from model import Model
from plot import PlotBatches
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch
import time

class ModelRunner:
    def __init__(self, window_size, learning_rate, is_cuda):
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.is_cuda = is_cuda
    def initialize_random(self, num_words, num_tags, embed_depth, num_features=0):
        net = Model(num_words, num_tags, embed_depth, self.window_size, num_features)
        self.__initialize(net)
    def initialize_pretrained(self, num_tags, embeddings, num_features=0):
        net = Model.pretrained(num_tags, self.window_size, embeddings, num_features)
        self.__initialize(net)
    def __initialize(self, net):
        if (self.is_cuda):
            # from torch.backends import cudnn
            # cudnn.benchmark = True
            net.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        self.net = net
    def train_and_eval(self, trainloader, epoches, testloader, omit_tag_id=None, eval_mode="everyepoch"):
        """Train model and print progress to console

        Attributes:
            eval_mode:
                        'everyepoch' will eval acc&loss on test set after each epoch, print to console.
                        'plot'       will eval acc&loss on test set after every 50 BATCHES, plot gui will show after train.
        """
        plotter = PlotBatches()
        plot_every_batch = eval_mode=="plot"
        updates_per_epoch = 0
        for epoch in range(epoches):  # loop over the dataset multiple times
            self.net.train(True)

            start_e_t = time.time()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                start_b_t = time.time()

                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

                if self.is_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                end_b_t = time.time()

                # print statistics
                running_loss += loss.data[0]
                if i % 50 == 49:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f timer_per_batch: %.3f' %
                          (epoch + 1, i + 1, running_loss / 50, (end_b_t - start_b_t)))
                    if plot_every_batch:
                        test_acc, test_loss = self.eval(testloader, omit_tag_id, to_print=False)
                        plotter.update(running_loss, test_loss, test_acc)
                        updates_per_epoch += 1
                        self.net.train(True)
                running_loss = 0.0
            end_e_t = time.time()
            print('epoch time: %.3f' % (end_e_t - start_e_t))
            self.eval(testloader, omit_tag_id)
        if plot_every_batch:
            updates_per_epoch /= epoches
            plotter.show(updates_per_epoch)

    def eval(self, testloader, omit_tag_id=None, to_print=True):
        self.net.train(False)  # Disable dropout during eval mode
        correct = 0
        total = 0
        for data in testloader:
            features, labels = data
            input = Variable(features)
            if self.is_cuda:
                input, labels = input.cuda(), labels.cuda()
            outputs = self.net(input)
            loss = self.criterion(outputs, Variable(labels))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if omit_tag_id is not None:
                O_tag_id = omit_tag_id
                diff_O_tag = sum([1 for p, l in zip(predicted, labels) if p == l and l == O_tag_id])
                correct += (predicted == labels).sum()
                correct -= diff_O_tag
                total -= diff_O_tag
            else:
                correct += (predicted == labels).sum()
        acc = 1.0 * correct / total
        loss = loss.data[0]
        if to_print:
            print('Accuracy of the network on the %d test words: %.5f %%, loss: %.3f' % (
                total, acc, loss))
        return acc, loss