from extract import *


def score(f1):
    # return avg - delta, so values will be close
    return (f1[anno2i[WORK_FOR]] + f1[anno2i[LIVE_IN]]) - abs(f1[anno2i[WORK_FOR]] - f1[anno2i[LIVE_IN]])


def get_dicts(filename):
    a2i = {UNK: 0, UP: 1, DOWN: 2, LEFT: 3, RIGHT: 4}
    w2i = {UNK: 0}
    t2i = {UNK: 0, ROOT: 1}
    n2i = {UNK: 0}
    d2i = {UNK: 0}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            line = line.split()
            if len(line) == 0:
                continue

            word = line[1]
            tag = line[3]
            dep = line[6]

            if word not in w2i:
                w2i[word] = len(w2i)
            if tag not in t2i:
                t2i[tag] = len(t2i)

            if dep not in d2i:
                d2i[dep] = len(d2i)

            if len(line) > 8:
                ner = line[8]
                if ner not in n2i:
                    n2i[ner] = len(n2i)
    return w2i, t2i, n2i, d2i, a2i


def features_to_inputs(features_by_sent_id, anno_by_sent_id, feature_key_to_anno_key, dicts):
    X = []
    Y = []

    sent_ids = features_by_sent_id.keys()
    for sent_id in sent_ids:
        for f_key, features in features_by_sent_id[sent_id].items():

            input_vec = feat2vec(features, dicts)
            X.append(input_vec)

            if f_key not in feature_key_to_anno_key[sent_id]:
                Y.append(anno2i[UNK])
                continue

            anno_key = feature_key_to_anno_key[sent_id][f_key]
            if anno_key not in anno_by_sent_id[sent_id]:
                Y.append(anno2i[UNK])
                continue

            anno = anno_by_sent_id[sent_id][anno_key]
            if anno not in anno2i:
                Y.append(anno2i[UNK])
                continue

            # annontion is allowed, and we know its type.
            Y.append(anno2i[anno])
    return X, Y


def main(train_fname, train_anno, dev_fname, dev_anno):
    dicts = get_dicts(train_fname)
    features_by_sent_id = read_processed_file(train_fname)
    anno_by_sent_id = read_annotations_file(train_anno)
    feature_key_to_anno_key = compute_feature_key_to_anno_key(anno_by_sent_id, features_by_sent_id)
    X, Y = features_to_inputs(features_by_sent_id, anno_by_sent_id, feature_key_to_anno_key, dicts)

    features_by_sent_id = read_processed_file(dev_fname)
    anno_by_sent_id = read_annotations_file(dev_anno)
    feature_key_to_anno_key = compute_feature_key_to_anno_key(anno_by_sent_id, features_by_sent_id)
    DevX, DevY = features_to_inputs(features_by_sent_id, anno_by_sent_id, feature_key_to_anno_key, dicts)

    w2i, t2i, n2i, d2i, a2i = dicts
    from dynet_network import run_network_print_result
    network, (_, _, _, f1) = run_network_print_result(X, Y, DevX, DevY, len(w2i), len(a2i), len(d2i))

    # max network out of 5
    for i in xrange(0):
        net, (_, _, _, f1_new) = run_network_print_result(X, Y, DevX, DevY, len(w2i), len(a2i), len(d2i))
        if score(f1_new) > score(f1):
            f1 = f1_new
            network = net
    print 'Saving model with f1 score of:', f1
    network.model.save(MODEL_NAME)
    import pickle
    with open(DICTS_NAME, 'wb') as f:
        pickle.dump(dicts, f)


if __name__ == '__main__':
    import sys

    if len(sys.argv) <= 4:
        print 'Usage: <train.processed> <train.annotations> <dev.processed> <dev.annotation>'
        exit()
    main(*sys.argv[1:])
