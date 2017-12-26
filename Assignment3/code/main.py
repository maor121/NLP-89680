import numpy as np

def calc_cosine_distance(contexts, target_words_ids):

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
    sim = {}
    for u in target_words_ids:
        u_context = contexts[u]
        sim[u] = {}
        for v in u_context:
            if u != v:
                sim[u][v] = DT[(u,v)] / np.sqrt(lengths[u] * lengths[v])

    return sim

def contexts_to_pmi_contexts(contexts):
    import copy
    u_freqs = {}
    contexts = copy.deepcopy(contexts)

    freq_total = 0
    for u,u_context in contexts.items():
        u_freq_total = 0
        for __, u_v_freq in u_context.items():
            u_freq_total += u_v_freq
        u_freqs[u] = u_freq_total
        freq_total += u_freq_total

    for u, u_context in contexts.items():
        p_u = float(u_freqs[u]) / freq_total
        for v, u_v_freq in u_context.items():
            p_v = float(u_freqs[v]) / freq_total
            p_u_v = float(u_v_freq) / freq_total
            u_context[v] = np.log(p_u_v) - (np.log(p_u) + np.log(p_v))

    return contexts

def inverse_dict(dict):
    return {v: k for k, v in dict.iteritems()}


if __name__ == '__main__':
    import utils
    import time
    from preprocess import Preprocess

    is_tiny = False


    if is_tiny:
        filename = "wikipedia.tinysample.trees.lemmatized"
    else:
        filename = "wikipedia.sample.trees.lemmatized"
    time_s = time.time()
    preprocess = Preprocess.from_input(filename, context_mode="window")
    preprocess.save_to_file("../out/window_context/preprocess.pickle")
    time_e = time.time()
    print("Done. time: %.2f secs" % (time_e - time_s))

    preprocess = Preprocess.load_from_file("../out/window_context/preprocess.pickle")

    I2W = inverse_dict(preprocess.W2I.S2I)

    target_words = ["car" ,"bus" ,"hospital" ,"hotel" ,"gun" ,"bomb" ,"horse" ,"fox" ,"table", "bowl", "guitar" ,"piano"]
    target_words_ids = [preprocess.W2I.get_id(w) for w in target_words]

    contexts = contexts_to_pmi_contexts(preprocess.contexts)
    sim = calc_cosine_distance(preprocess.contexts, target_words_ids)
    utils.save_obj(sim, "../out/window_context/sim_pmi.pickle")
    #sim = utils.load_obj("../out/window_context/sim_pmi.pickle")


    for u in target_words_ids:
        u_sim = sorted(list(sim[u].items()), key=lambda (v,score): score, reverse=True)
        u_sim_top_20 = [(I2W[v],"%.3f" % score) for i, (v,score) in enumerate(u_sim) if i < 20]
        print(I2W[u], u_sim_top_20)

    print(0)
