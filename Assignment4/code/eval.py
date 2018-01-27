import sys
from extract import read_annotations_file, compute_feature_key_to_anno_key
from collections import Counter
import utils
import numpy.random

def compute_acc(anno_by_sent_gold, anno_by_sent_test):
    test_key_to_gold_key = compute_feature_key_to_anno_key(anno_by_sent_gold, anno_by_sent_test)
    gold_key_to_test_key = {}

    sent_ids = list(set(anno_by_sent_gold.keys() + anno_by_sent_test.keys()))

    for sent_id in sent_ids:
        gold_key_to_test_key[sent_id] = utils.inverse_dict(test_key_to_gold_key.get(sent_id,{})) # 1 : 1

    relevant_retrieved = Counter()
    relevant = Counter()
    retrieved = Counter()

    gold_labels = set()
    for sent_id in sent_ids:
        gold_labels.update(anno_by_sent_gold[sent_id].values())

    good = bad = 0.0
    precision_errors = {l:[] for l in gold_labels}
    recall_errors = {l:[] for l in gold_labels}
    for sent_id in sent_ids:
        for test_key, re_test in anno_by_sent_test.get(sent_id,{}).items():
            if test_key in test_key_to_gold_key[sent_id]: # matched
                gold_key = test_key_to_gold_key[sent_id][test_key]
                re_gold = anno_by_sent_gold[sent_id][gold_key]
                if re_gold == re_test: # correct classification
                    good += 1
                    relevant_retrieved.update([re_gold])
                else:
                    # misclassification: precision error, recall error
                    bad += 1

                    # extracted by the system but absent in the gold
                    precision_errors[re_test].append(str(gold_key)+"-"+str(test_key)+"\t"+re_gold+"\t"+re_test)

                    # present in gold but absent from system
                    recall_errors[re_gold].append(str(gold_key)+"-"+str(test_key)+"\t"+re_gold+"\t"+re_test)
            else: # false-positive: precision error
                precision_errors[re_test].append("NO GOLD KEY-"+str(test_key) + "\tNO GOLD ANNO\t" + re_test)

        for gold_key, re_gold in anno_by_sent_gold[sent_id].items():
            if gold_key not in gold_key_to_test_key.get(sent_id,{}):  # no match
                recall_errors[re_gold].append(str(gold_key)+'-NO TEST KEY' + "\t"+re_gold+"\tNO TEST ANNO")

        relevant.update([re_gold for re_gold in anno_by_sent_gold.get(sent_id,{}).values()])
        retrieved.update([re_test for re_test in anno_by_sent_test.get(sent_id,{}).values()])

    acc = good / (good + bad)
    recall = {l: relevant_retrieved[l] / relevant[l] if relevant[l]>0 else 0 for l in gold_labels}
    prec = {l:relevant_retrieved[l] / retrieved[l] if retrieved[l]>0 else 0 for l in gold_labels}
    f1 = {l:(2.0 * recall[l] * prec[l]) / (recall[l] + prec[l]) if (recall[l] + prec[l])>0 else 0 for l in gold_labels}

    return acc, recall, prec, f1, precision_errors, recall_errors


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'eval.py gold_file test_file\nWrong number of arguments given'
    gold_filename = sys.argv[1]
    test_filename = sys.argv[2]

    print_errors_sample = True
    error_sample_size = 10

    anno_by_sent_gold = read_annotations_file(gold_filename)
    anno_by_sent_test = read_annotations_file(test_filename)

    print '=' * 30
    print 'RESULTS:'

    acc, recall, prec, f1, prec_errors, rec_errors = compute_acc(anno_by_sent_gold, anno_by_sent_test)

    print '\tacc:   ', acc
    print '\trecall:', recall
    print '\tprec:  ', prec
    print '\tf1:    ', f1
    print '=' * 30

    if print_errors_sample:
        prec_errors = {l:numpy.random.choice(prec_errors[l], size=min(error_sample_size,len(prec_errors[l]))) for l in prec_errors.keys() if len(prec_errors[l])>0}
        rec_errors = {l:numpy.random.choice(rec_errors[l], size=min(error_sample_size,len(rec_errors[l]))) for l in rec_errors.keys() if len(rec_errors[l])>0}

        print 'PRECISION ERROS(sample):'
        print '(in test annotations but not in gold)'
        print '\n'.join([l+'\n\t'+'\n\t'.join(list(prec_errors[l])) for l in prec_errors])
        print '=' * 30

        print 'RECALL ERROS(sample):'
        print '(in gold annotations but not in test)'
        print '\n'.join([l+'\n\t'+'\n\t'.join(list(rec_errors[l])) for l in rec_errors])
        print '=' * 30