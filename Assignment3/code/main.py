def calc_cosine_distance(contexts):
    from itertools import tee
    import numpy as np

    sim = {}

    # 1) Calculate length of every vector u
    print("Part 1")
    lengths = {}
    for u, u_context in contexts.items():
        sum = 0.0
        for __, u_v_count in u_context.items():
            sum += np.log(u_v_count) ** 2
        lengths[u] = sum

    # 2) Calculate DT
    print("Part 2")
    DT = {}
    for u, u_context in contexts.items():
        for att, u_att_count in u_context.items():
            for v, v_att_count in contexts[att].items():
                DT[(u,v)] = DT.get((u,v), 0.0) + np.log(u_att_count) * np.log(v_att_count)

    # 3) Calculate cosine similarity
    print("Part 3")
    u_iter = iter(contexts)
    while True:
        try:
            u = u_iter.next()
            v_iter = tee(u_iter)
            while True:
                try:
                    v = v_iter.next()
                    sim[(u,v)] = sim[(v,u)] = DT[(u,v)] / np.sqrt(lengths[u] + lengths[v])
                except StopIteration:
                    break
        except StopIteration:
            break

    return sim


def inverse_dict(dict):
    return {v: k for k, v in dict.iteritems()}


if __name__ == '__main__':
    import time
    from preprocess import Preprocess

    is_tiny = False

    """
    if is_tiny:
        filename = "wikipedia.tinysample.trees.lemmatized"
    else:
        filename = "wikipedia.sample.trees.lemmatized"
    time_s = time.time()
    preprocess = Preprocess.from_input(filename)
    preprocess.save_to_file("../out/preprocess.pickle")
    time_e = time.time()
    print("Done. time: %.2f secs" % (time_e - time_s))
    """

    preprocess = Preprocess.load_from_file("../out/preprocess.pickle")

    I2W = inverse_dict(preprocess.W2I.S2I)

    sim = calc_cosine_distance(preprocess.contexts)
    print(sorted([(I2W[u],I2W[v],score) for (u,v),score in sim.items()],key=lambda (u,v,score) : score))

    """
    words_without_context = []
    words_with_context = []
    for w, w_id in preprocess.W2I.S2I.items():
        if w_id not in preprocess.contexts:
            words_without_context.append(w)
        else:
            words_with_context.append(w)

    print("Words without context:%d\n%s" % (len(words_without_context), words_without_context))
    print("Words with context:%d\n%s" % (len(words_with_context), words_with_context))
    """

    print(0)
