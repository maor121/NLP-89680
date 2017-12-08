from IPython import display
from matplotlib import pyplot as plt
import numpy as np

class PlotBatches(object):
    def __init__(self):
        self.train_history = {"train_loss" : [], "test_loss" : [], "test_acc": []}

    def update(self, train_loss, test_loss, test_acc):
        self.train_history['train_loss'].append(train_loss)
        self.train_history['test_loss'].append(test_loss)
        self.train_history['test_acc'].append(test_acc)
    def show(self, updates_per_epoch):
        train_loss = np.array([i for i in self.train_history['train_loss']])
        test_loss = np.array([i for i in self.train_history['test_loss']])
        test_acc = np.array([i for i in self.train_history['test_acc']])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.gca().cla()
        plt.xlabel("epoches")
        plt.plot(train_loss, label="Train Loss")
        plt.plot(test_loss, label="Test Loss")
        plt.plot(test_acc, label="Test Accuracy")


        # scale
        ticks = plt.gca().get_xticks() * (1.0/updates_per_epoch)
        ticks = np.round(ticks, 2)
        plt.gca().set_xticklabels(ticks)

        # Annotate epoches points
        last_reported_epoch = None
        for i, y_val in enumerate(test_acc):
            epoch = (i+1) / updates_per_epoch
            if epoch != last_reported_epoch and epoch > 0:
                ax.annotate("%.3f" % test_acc[i], xy=(epoch*updates_per_epoch,test_acc[i]), textcoords='data')
                last_reported_epoch = epoch

        #plt.legend()
        #plt.draw()
        plt.grid()
        plt.show()
