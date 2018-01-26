import sys
from extract import read_annotations_file, compute_feature_key_to_anno_key
from collections import Counter


def compute_acc(anno_by_sent_gold, anno_by_sent_test):
    test_key_to_gold_key = compute_feature_key_to_anno_key(anno_by_sent_gold, anno_by_sent_test)

    sent_ids = list(set(anno_by_sent_gold.keys() + anno_by_sent_test.keys()))

    relevant_retrieved = Counter()
    relevant = Counter()
    retrieved = Counter()

    good = bad = 0.0
    for sent_id in sent_ids:
        for test_key, re_test in anno_by_sent_test[sent_id].items():
            if test_key in test_key_to_gold_key[sent_id]: # matched
                gold_key = test_key_to_gold_key[sent_id][test_key]
                re_gold = anno_by_sent_gold[sent_id][gold_key]
                if re_gold == re_test: # correct classification
                    good += 1
                    relevant_retrieved.update([re_gold])
                else:
                    bad += 1

        relevant.update([re_gold for re_gold in anno_by_sent_gold[sent_id].values()])
        retrieved.update([re_test for re_test in anno_by_sent_test[sent_id].values()])
    gold_labels = relevant.keys()

    acc = good / (good + bad)
    recall = {l: relevant_retrieved[l] / relevant[l] for l in gold_labels}
    prec = {l:relevant_retrieved[l] / retrieved[l] for l in gold_labels}
    f1 = {l:(2.0 * recall[l] * prec[l]) / (recall[l] + prec[l]) for l in gold_labels}

    return acc, recall, prec, f1


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'eval.py gold_file test_file\nWrong number of arguments given'
    gold_filename = sys.argv[1]
    test_filename = sys.argv[2]

    anno_by_sent_gold = read_annotations_file(gold_filename)
    anno_by_sent_test = read_annotations_file(test_filename)

    print '=' * 30
    print 'RESULTS:'

    acc, recall, prec, f1 = compute_acc(anno_by_sent_gold, anno_by_sent_test)

    print '\tacc:   ', acc
    print '\trecall:', recall
    print '\tprec:  ', prec
    print '\tf1:    ', f1
    print '=' * 30
