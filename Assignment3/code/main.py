def calc_cosine_distance(preprocess, target_words_ids):
    from itertools import tee
    import numpy as np

    sim = {}
    contexts = preprocess.contexts
    I2W = inverse_dict(preprocess.W2I.S2I)

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
    for u in target_words_ids:
        u_context = contexts[u]
        for att, u_att_count in u_context.items():
            for v, v_att_count in contexts[att].items():
                k = np.log(u_att_count) * np.log(v_att_count)
                DT[(u,v)] = DT.get((u,v), 0.0) + k

    # 3) Calculate cosine similarity
    print("Part 3")
    for u in target_words_ids:
        u_context = contexts[u]
        for v in u_context:
            if u != v:
                sim[(u,v)] = DT[(u,v)] / np.sqrt(lengths[u] * lengths[v])

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

    target_words = ["car" ,"bus" ,"hospital" ,"hotel" ,"gun" ,"bomb" ,"horse" ,"fox" ,"table", "bowl", "guitar" ,"piano"]
    target_words_ids = [preprocess.W2I.get_id(w) for w in target_words]
    sim = calc_cosine_distance(preprocess, target_words_ids)
    sortedSim = sorted([(I2W[u],I2W[v],score) for (u,v),score in sim.items()],key=lambda (u,v,score) : score, reverse=True)
    limit = 50
    for i, tup in enumerate(sortedSim):
        if i == limit:
            break
        print(tup)


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
