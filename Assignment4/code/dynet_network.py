import dynet as dy
import numpy as np
import random

class Model(object):
    def __init__(self, vocab_size, arr_size, deps_size, params):
        dep_dropout, cons_dropout, embed_dim, cons_dim, deps_dim = params
        self.deps_dropout = dep_dropout
        self.cons_dropout = cons_dropout
        self.embed_dim = embed_dim
        self.cons_dim = cons_dim
        self.deps_dim = deps_dim

        self._model = dy.ParameterCollection()
        # lookups for the inputs
        # idea - have different lookup for the different sequence models
        self.word_lookup = self.model.add_lookup_parameters((vocab_size, embed_dim))
        self.arrow_lookup = self.model.add_lookup_parameters((arr_size, embed_dim))
        self.dep_lookup = self.model.add_lookup_parameters((deps_size, embed_dim))

        # sequence LSTM's
        self.cons_lstm = dy.LSTMBuilder(1, embed_dim, self.cons_dim, self.model)
        self.deps_lstm = dy.LSTMBuilder(1, embed_dim, self.deps_dim, self.model)

        # idea - add b's (biases vectors)
        dims = (128, 64)
        self.pW1 = self.model.add_parameters((dims[0], 4 + self.cons_dim + self.deps_dim))
        self.pW2 = self.model.add_parameters((dims[1], dims[0]))
        self.pW3 = self.model.add_parameters((3, dims[1]))

    @property
    def model(self):
        return self._model

    def cons_repr(self, cons_path):
        outvec = []
        for i, x in enumerate(cons_path):
            if i % 2 == 0:
                outvec.append(self.word_lookup[x])
            else:
                outvec.append(self.arrow_lookup[x])
        return outvec

    def cons_output(self, cons_path, train=False):
        cons_path = self.cons_repr(cons_path)
        if train:
            cons_path = [dy.dropout(x, self.cons_dropout) for x in cons_path]  # apply dropout
        cons_lstm = self.cons_lstm.initial_state()
        if len(cons_path) > 0:
            lstm_out = cons_lstm.transduce(cons_path)
            cons_path = lstm_out[-1]
        else:
            cons_path = dy.vecInput(self.cons_dim)
            cons_path.set(np.zeros(self.cons_dim))
        return cons_path

    def deps_repr(self, deps_path):
        outvec = []
        for i, x in enumerate(deps_path):
            if i % 3 == 0:
                outvec.append(self.word_lookup[x])
            elif i % 3 == 1:
                outvec.append(self.arrow_lookup[x])
            else:
                outvec.append(self.dep_lookup[x])

        return outvec

    def deps_output(self, deps_path, train=False):
        deps_path = self.deps_repr(deps_path)
        if train:
            deps_path = [dy.dropout(x, self.deps_dropout) for x in deps_path]  # apply dropout

        deps_lstm = self.deps_lstm.initial_state()
        if len(deps_path) > 0:
            lstm_out = deps_lstm.transduce(deps_path)
            deps_path = lstm_out[-1]
        else:
            deps_path = dy.vecInput(self.deps_dim)
            deps_path.set(np.zeros(self.deps_dim))
        return deps_path

    def ners_output(self, ners):
        ners_input = dy.vecInput(4)
        ners_input.set(ners)
        return ners_input

    def build_graph(self, inputs, train=False):
        ners, cons_path, deps_path = inputs

        dy.renew_cg()

        ners_vec = self.ners_output(ners)
        cons_vec = self.cons_output(cons_path, train)
        deps_vec = self.deps_output(deps_path, train)

        mlp_input = dy.concatenate([ners_vec, cons_vec, deps_vec])

        W1 = dy.parameter(self.pW1)
        W2 = dy.parameter(self.pW2)
        W3 = dy.parameter(self.pW3)

        output = dy.softmax(W3 * dy.tanh(W2 * dy.tanh(W1 * mlp_input)))

        return output

    def create_network_return_loss(self, inputs, output):
        out = self.build_graph(inputs, True)
        loss = -dy.log(dy.pick(out, output))
        return loss

    def create_network_return_best(self, inputs):
        out = self.build_graph(inputs)
        return np.argmax(out.npvalue())


def compute_acc(devY, goldY):
    from collections import Counter
    table = {}
    good = bad = 0.0
    for pred, gold in zip(devY, goldY):
        if gold not in table:
            table[gold] = Counter()
        table[gold].update([pred])
        if gold == pred:
            good += 1
        else:
            bad += 1

    acc = good / (good + bad)
    recall = {}
    prec = {}
    f1 = {}
    # move onto computing recall and precision
    for gold in table:
        tp = float(table[gold][gold])
        tpfn = sum(table[gold].values())
        recall[gold] = tp / tpfn

        sm = 0.0
        for r_gold in table:
            sm += table[r_gold][gold]
        prec[gold] = tp / sm
        f1[gold] = (2.0 * recall[gold] * prec[gold]) / (recall[gold] + prec[gold])

    return acc, recall, prec, f1


def run_network_print_result(trainX, trainY, devX, devY, vocab_size, arr_size, deps_size):
    assert len(trainX) == len(trainY)

    # dep_dropout, cons_dropout, embed_dim, cons_dim, deps_dim = params
    params = (0.3, 0.3, 1, 12, 12)

    print '=' * 30
    print 'TRAINING THE NETWORK'
    print '\tdep_dropout \t%f' % params[0]
    print '\tcons_dropout\t%f' % params[1]
    print '\tembed_dim   \t%d' % params[2]
    print '\tcons_dim    \t%d' % params[3]
    print '\tdeps_dim    \t%d' % params[4]
    network = Model(vocab_size, arr_size, deps_size, params)
    trainer = dy.AdamTrainer(network.model)

    train_data = zip(trainX, trainY)
    for epoch in xrange(25):
        random.shuffle(train_data)
        for inp, lbl in train_data:
            loss = network.create_network_return_loss(inp, lbl)
            loss_val = loss.value()  # run forward prop
            loss.backward()
            trainer.update()

        train_output = [network.create_network_return_best(inp) for inp in trainX]
        dev_output = [network.create_network_return_best(inp) for inp in devX]
        trainacc, _, _, _ = compute_acc(train_output, trainY)
        devacc, _, _, _ = compute_acc(dev_output, devY)
        # print '%d %f %f' % (epoch, trainacc, devacc)

    dev_output = [network.create_network_return_best(inp) for inp in devX]
    assert len(dev_output) == len(devY)
    # compute accuracies
    print 'RESULTS:'
    acc, recall, prec, f1 = compute_acc(dev_output, devY)
    print '\tacc:   ', acc
    print '\trecall:', recall
    print '\tprec:  ', prec
    print '\tf1:    ', f1
    print '=' * 30
