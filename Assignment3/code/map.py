import numpy as np

def calc_precision_from_rel(rel_arr):
    sum = 0.0
    total_correct = 0
    for i, r in enumerate(rel_arr):
        if r == 1:
            total_correct += 1
        sum += float(total_correct) / (i+1)

    assert len(rel_arr) == 20
    return sum / len(rel_arr)


if __name__ == '__main__':
    rel_car_sentence = [int(s) for s in "1 1 0 1 1 1 0 0 1 0 0 0 0 0 1 0 0 0 1 0".split()]
    rel_car_window = [int(s) for s in "0 1 1 1 1 1 0 1 1 0 0 1 0 1 1 0 0 0 0 0".split()]
    rel_car_tree = [int(s) for s in "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1".split()]


    rel_piano_sentence = [int(s) for s in "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1".split()]
    rel_piano_window = [int(s) for s in "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1".split()]
    rel_piano_tree = [int(s) for s in "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0".split()]

    print("ap,car,sentence", calc_precision_from_rel(rel_car_sentence))
    print("ap,car,window", calc_precision_from_rel(rel_car_window))
    print("ap,car,tree", calc_precision_from_rel(rel_car_tree))
    print("ap,piano,sentence", calc_precision_from_rel(rel_piano_sentence))
    print("ap,piano,window", calc_precision_from_rel(rel_piano_window))
    print("ap,piano,tree", calc_precision_from_rel(rel_piano_tree))


    print("map,sentence", np.average([calc_precision_from_rel(rel) for rel in [rel_car_sentence, rel_piano_sentence]]))
    print("map,window", np.average([calc_precision_from_rel(rel) for rel in [rel_car_window, rel_piano_window]]))
    print("map,tree", np.average([calc_precision_from_rel(rel) for rel in [rel_car_tree, rel_piano_tree]]))