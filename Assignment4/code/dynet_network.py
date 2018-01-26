import dynet as dy
import numpy as np

embed_dim = 128
cons_path_out_dim = 128
deps_path_out_dim = 128


class Model(object):
    def __init__(self, model, vocab_size, arr_size, deps_size):

        # lookups for the inputs
        # idea - have different lookup for the different sequence models
        self.word_lookup = model.add_lookup_parameters((vocab_size, embed_dim))
        self.arrow_lookup = model.add_lookup_parameters((arr_size, embed_dim))
        self.dep_lookup = model.add_lookup_parameters((deps_size, embed_dim))

        # sequence LSTM's
        self.cons_lstm = dy.LSTMBuilder(1, embed_dim, cons_path_out_dim, model)
        self.deps_lstm = dy.LSTMBuilder(1, embed_dim, deps_path_out_dim, model)

        # idea - add b's (biases vectors)
        dims = (128, 64)
        self.pW1 = model.add_parameters((dims[0], 4 + cons_path_out_dim + deps_path_out_dim))
        self.pW2 = model.add_parameters((dims[1], dims[0]))
        self.pW3 = model.add_parameters((3, dims[1]))

    def cons_repr(self, cons_path):
        outvec = []
        for i, x in enumerate(cons_path):
            if i % 2 == 0:
                outvec.append(self.word_lookup[x])
            else:
                outvec.append(self.arrow_lookup[x])
        return outvec

    def cons_output(self, cons_path):
        cons_path = self.cons_repr(cons_path)
        cons_lstm = self.cons_lstm.initial_state()
        if len(cons_path) > 0:
            lstm_out = cons_lstm.transduce(cons_path)
            cons_path = lstm_out[-1]
        else:
            cons_path = dy.vecInput(cons_path_out_dim)
            cons_path.set(np.zeros(cons_path_out_dim))
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

    def deps_output(self, deps_path):
        deps_path = self.deps_repr(deps_path)
        deps_lstm = self.deps_lstm.initial_state()
        if len(deps_path) > 0:
            lstm_out = deps_lstm.transduce(deps_path)
            deps_path = lstm_out[-1]
        else:
            deps_path = dy.vecInput(deps_path_out_dim)
            deps_path.set(np.zeros(deps_path_out_dim))
        return deps_path

    def ners_output(self, ners):
        ners_input = dy.vecInput(4)
        ners_input.set(ners)
        return ners_input

    def build_graph(self, inputs):
        ners, cons_path, deps_path = inputs

        dy.renew_cg()

        ners_vec = self.ners_output(ners)
        cons_vec = self.cons_output(cons_path)
        deps_vec = self.deps_output(deps_path)

        mlp_input = dy.concatenate([ners_vec, cons_vec, deps_vec])

        W1 = dy.parameter(self.pW1)
        W2 = dy.parameter(self.pW2)
        W3 = dy.parameter(self.pW3)

        output = dy.softmax(W3 * dy.tanh(W2 * dy.tanh(W1 * mlp_input)))

        return output

    def create_network_return_loss(self, inputs, output):
        out = self.build_graph(inputs)
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

    m = dy.ParameterCollection()
    network = Model(m, vocab_size, arr_size, deps_size)
    trainer = dy.AdamTrainer(m)

    for epoch in xrange(20):
        for inp, lbl in zip(trainX, trainY):
            loss = network.create_network_return_loss(inp, lbl)
            loss_val = loss.value()  # run forward prop
            loss.backward()
            trainer.update()

        train_output = [network.create_network_return_best(inp) for inp in trainX]
        dev_output = [network.create_network_return_best(inp) for inp in devX]
        trainacc, _, _, _ = compute_acc(train_output, trainY)
        devacc, _, _, _ = compute_acc(dev_output, devY)
        print '%d %f %f' % (epoch, trainacc, devacc)

    dev_output = [network.create_network_return_best(inp) for inp in devX]
    assert len(dev_output) == len(devY)
    # compute accuracies
    acc, recall, prec, f1 = compute_acc(dev_output, devY)
    print 'Acc:', acc
    print 'Recall:', recall
    print 'prec:', prec
    print 'f1:', f1
