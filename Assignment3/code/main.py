"""Usage: main.py <INPUT_FILE> [-m mod_number]

-h --help    show this
-m mod_number    mod: 1=sentence, 2=window, 3=tree [default: 1]

"""
from docopt import docopt

import numpy as np

def calc_cosine_distance(contexts, target_words_ids):

    # 1) Calculate length of every vector u
    print("Cosine distance: length of vectors...")
    lengths = {}
    for u, u_context in contexts.items():
        sum = 0.0
        for __, u_v_count in u_context.items():
            sum += np.log(u_v_count) ** 2
        lengths[u] = sum

    # 2) Calculate DT
    print("Cosine distance: dot product...")
    DT = {}
    for u in target_words_ids:
        u_context = contexts[u]
        for att, u_att_count in u_context.items():
            for v, v_att_count in contexts[att].items():
                k = np.log(u_att_count) * np.log(v_att_count)
                DT[(u,v)] = DT.get((u,v), 0.0) + k

    # 3) Calculate cosine similarity
    print("Cosine distance: similarity...")
    sim = {}
    for u in target_words_ids: #4541
        u_context = contexts[u]
        sim[u] = {}
        for att in u_context: #18432
            for v in contexts[att]:
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

    sum_all_p = np.float64(0.0) # PMI sanity check
    sum_all_p_u = np.float64(0.0)

    to_filter = []
    for u, u_context in contexts.items():
        p_u = float(u_freqs[u]) / freq_total
        sum_all_p_u += p_u
        for v, u_v_freq in u_context.items():
            p_v = float(u_freqs[v]) / freq_total
            p_u_v = float(u_v_freq) / freq_total
            sum_all_p += p_u_v
            u_v_pmi = np.log(p_u_v) - (np.log(p_u) + np.log(p_v))
            if u_v_pmi < 0: # pmi<0, Not enough information
                # u_v_pmi = 0 # Can give exception later. Remove negative pmis instead
                to_filter.append((u,v))
            u_context[v] = u_v_pmi

    assert abs(sum_all_p_u -1) < 0.001 # sanity
    assert abs(sum_all_p - 1) < 0.001 # sanity

    # Filter negative pmis
    for (u,v) in to_filter:
        contexts[u].pop(v, None)
        contexts[v].pop(v, None)
    # Filter words that remained with no features
    to_filter = []
    for w_id, w_context in contexts.items():
        if len(w_context) == 0:
            to_filter.append(w_id)
    for w_id in to_filter:
        contexts.pop(w_id)


    return contexts


def weighted_jacard_matrix(pmi_contexts, target_words_ids):
    sim = {}
    for u in target_words_ids:
        u_context = pmi_contexts[u]
        sim[u] = {}
        sum_min = {} # mone, by v
        sum_max = {} # mechane, by v
        for att, u_att_pmi in u_context.items():
            for v, v_att_pmi in pmi_contexts[att].items():
                if u != v:
                    sum_min[v] = sum_min.get(v, 0.0) + min(u_att_pmi, v_att_pmi)
                    sum_max[v] = sum_max.get(v, 0.0) + max(u_att_pmi, v_att_pmi)
        for v in sum_min:
            sim[u][v] = sum_min[v] / sum_max[v]

    return sim


if __name__ == '__main__':
    import utils
    import time
    from preprocess import Preprocess

    arguments = docopt(__doc__, version='Naval Fate 2.0')
    filename = arguments['<INPUT_FILE>']
    mod_num = int(arguments['-m'])
    legal_modes = {1:"sentence", 2:"window", 3:"tree"}
    if mod_num not in legal_modes:
        print "Unknown mod number"
        exit()

    calc_preprocess = True
    calc_sim = True
    save_to_file = False

    mod = legal_modes[mod_num]
    out_dir = "../out/{}_context".format(mod)

    print("Mod: '"+mod+"'")

    print("Reading input file, preprocess stage...")
    if calc_preprocess:
        time_s = time.time()
        preprocess = Preprocess.from_input("../data/" + filename, context_mode=mod)
        if save_to_file:
            preprocess.save_to_file(out_dir+"/preprocess.pickle")
        time_e = time.time()
        print("Done. time: %.2f secs" % (time_e - time_s))
    else:
        preprocess = Preprocess.load_from_file(out_dir+"/preprocess.pickle")


    target_words = ["car" ,"bus" ,"hospital" ,"hotel" ,"gun" ,"bomb" ,"horse" ,"fox" ,"table", "bowl", "guitar" ,"piano"]
    target_words_ids = [preprocess.W2I.get_id(w) for w in target_words]

    W2I_TREE, contexts = preprocess.contexts

    I2W = utils.inverse_dict(preprocess.W2I.S2I)
    if mod == "tree":
        target_words_ids = [W2I_TREE.get_id(str(id)) for id in target_words_ids]
        I2W_TREE = utils.inverse_dict(W2I_TREE.S2I)
        inv_func = lambda u : " ".join([I2W[int(s)] if s.isdigit() else s for s in I2W_TREE[u].split()])
    else:
        inv_func = lambda u : I2W[u]

    print("Converting frequencies to pmis, calculating cosine distances for target words")
    if calc_sim:
        pmi_contexts = contexts_to_pmi_contexts(contexts)
        sim = calc_cosine_distance(pmi_contexts, target_words_ids)
        if save_to_file:
            utils.save_obj(sim, out_dir+"/sim_pmi.pickle")
    else:
        pmi_contexts = contexts_to_pmi_contexts(contexts)
        sim = utils.load_obj(out_dir+"/sim_pmi.pickle")

    # Print 1st order similarity
    print ("\n1st order:")
    for u in target_words_ids:
        u_sorted_pmi = sorted(list(pmi_contexts[u].items()), key=lambda (v, score): score, reverse=True)
        # u_pmi_top_20 = [(inv_func(v),"%.3f" % score) for i, (v,score) in enumerate(u_sim) if i < 20]
        u_pmi_top_20_no_score=[inv_func(v) for i, (v,score) in enumerate(u_sorted_pmi) if i < 20]
        print(inv_func(u), u_pmi_top_20_no_score)

    # Print 2nd order similarity
    print ("\n2nd order:")
    for u in target_words_ids:
        u_sim = set([])
        for u_att in pmi_contexts[u]:
            for v in pmi_contexts[u_att]:
                if u != v:
                    u_sim.add((v, sim[u][v]))
        u_sim = sorted(list(u_sim), key=lambda (v,score): score, reverse=True)
        u_sim_top_20_no_score=[inv_func(v) for i, (v,score) in enumerate(u_sim) if i < 20]
        print(inv_func(u), u_sim_top_20_no_score)

    print("\nDone")
