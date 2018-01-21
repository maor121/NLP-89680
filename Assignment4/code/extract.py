MAX_CLEN = 0


def extract_ners_from_sentence(sentence):
    ner_entities = []
    last_ner = []
    for i, word in enumerate(sentence):
        if word[5] == 'B':  # Begin NER entity
            if len(last_ner) > 0:
                ner_entities.append(last_ner)
            last_ner = [i]
        else:
            if word[5] == 'I':
                assert len(last_ner) != 0  # Illegal NER annotation, I without B
                last_ner.append(i)
            else:  # O
                if len(last_ner) > 0:
                    ner_entities.append(last_ner)
                    last_ner = []
    return ner_entities


def extract_ner_name(sentence, ner):
    name = " ".join([sentence[n][1] for n in ner])
    return name


def extract_ner_type(sentence, ner):
    ner_type = sentence[ner[0]][-1]
    return ner_type


def extract_constituent_path(sentence, ner1, ner2):
    global MAX_CLEN
    loc1 = ner1[-1]
    loc2 = ner2[-1]

    path_ids_up = []
    word = sentence[loc1]
    while word[3] != 0:  # not HEAD
        path_ids_up.append(word[0])  # ID
        word = sentence[word[3] - 1]
    path_ids_up.append(0)

    path_ids_down = []
    word = sentence[loc2]
    while word[3] != 0:  # not HEAD
        path_ids_down.append(word[0])  # ID
        word = sentence[word[3] - 1]
    path_ids_down.append(0)

    # Find lowest common ancestor
    ancestor = None
    ids_down_set = set(path_ids_down)
    for id in path_ids_up:
        if id in ids_down_set:
            ancestor = id
            break

    remove_after_index = path_ids_up.index(ancestor)
    path_ids_up = path_ids_up[:remove_after_index + 1]
    remove_after_index = path_ids_down.index(ancestor)
    path_ids_down = path_ids_down[:remove_after_index + 1]

    lens = len(path_ids_down) + len(path_ids_up)
    if lens > MAX_CLEN:
        MAX_CLEN = lens

    path_up_str = " ".join([(sentence[id - 1][2] if id != 0 else "S") + "_UP" for id in path_ids_up])  # POS UP
    path_down_str = " ".join([(sentence[id - 1][2] if id != 0 else "S") + "_DOWN" for id in reversed(path_ids_down)])

    return path_up_str + " " + path_down_str


def extract_typed_dependency_path(sentence, ner1, ner2):
    loc1 = ner1[-1]
    loc2 = ner2[-1]

    path_ids_right = []
    word = sentence[loc1]
    while word[3] != 0:  # not HEAD
        path_ids_right.append(word[0])  # ID
        word = sentence[word[3] - 1]

    path_idx_left = []
    word = sentence[loc2]
    while word[3] != 0:  # not HEAD
        path_idx_left.append(word[0])  # ID
        word = sentence[word[3] - 1]

    # Find lowest common ancestor
    ancestor = None
    ids_down_set = set(path_idx_left)
    for id in path_ids_right:
        if id in ids_down_set:
            ancestor = id
            break

    if ancestor is None:
        return ''

    remove_after_index = path_ids_right.index(ancestor)
    path_ids_right = path_ids_right[:remove_after_index][::-1]
    remove_after_index = path_idx_left.index(ancestor)
    path_idx_left = path_idx_left[:remove_after_index][::-1]

    path_left = ' '.join('%s L-%s' % (sentence[id - 1][1], sentence[id - 1][4]) for id in path_idx_left)
    path_right = ' '.join('%s R-%s' % (sentence[id - 1][1], sentence[id - 1][4]) for id in path_ids_right)
    common = sentence[ancestor - 1][1]
    return path_left + common + path_right


def print_ner_entities(sentence, ner_entities):
    for ner in ner_entities:
        name = extract_ner_name(sentence, ner)
        ner_type = extract_ner_type(sentence, ner)
        print(name + " " + ner_type)


def process_sentence(sentence):
    result = {}
    ner_entities = extract_ners_from_sentence(sentence)
    # print_ner_entities(sentence, ner_entities)
    # Some sentences have as many as 12 NERS in one sentence
    import itertools
    ner_pairs = list(itertools.combinations(ner_entities, 2))
    for tmpNer1, tmpNer2 in ner_pairs:
        for ner1, ner2 in [(tmpNer1, tmpNer2), (tmpNer2, tmpNer1)]:  # pair, reverse_pair
            name_1 = extract_ner_name(sentence, ner1)
            name_2 = extract_ner_name(sentence, ner2)
            entity_type_1 = extract_ner_type(sentence, ner1)
            entity_type_2 = extract_ner_type(sentence, ner2)
            type_concated = entity_type_1 + entity_type_2
            dep_path = extract_typed_dependency_path(sentence, ner1, ner2)
            constituent_path = extract_constituent_path(sentence, ner1, ner2)

            result[(name_1, name_2)] = (entity_type_1, entity_type_2, type_concated, constituent_path, dep_path)

    return result


def read_processed_file(filename):
    ID_IDX = 0
    WORD_IDX = 1
    POS_IDX = 3
    HEAD_IDX = 5
    TREE_IDX = 6
    NER_P_IDX = 7
    NER_IDX = 8

    features_by_sent_id = {}
    with open(filename) as inp_file:
        sentence = []
        saw_empty_line = True
        sent_id = None
        for line in inp_file:
            line = line.strip()
            if line.startswith("#"):
                if line.__contains__("#id"):
                    sent_id = line.split()[-1]
                continue  # Comment, skip
            if len(line) > 0:
                saw_empty_line = False
                arr = line.split()
                ner_type = arr[NER_IDX] if len(arr) > 8 else None
                word = (
                    int(arr[ID_IDX]), arr[WORD_IDX], arr[POS_IDX], int(arr[HEAD_IDX]), arr[TREE_IDX], arr[NER_P_IDX],
                    ner_type)
                sentence.append(word)
            else:
                if not saw_empty_line:
                    features = process_sentence(sentence)
                    features_by_sent_id[sent_id] = features
                    sentence = []
                saw_empty_line = True
    return features_by_sent_id


def read_annotations_file(filename):
    relation_by_sent_id = {}
    with open(filename) as anno_file:
        for line in anno_file:
            line = line.strip()
            arr = line.split('\t', 5)  # Last element in the whole sentence
            sent_id = arr[0]
            ner1 = arr[1]
            rel = arr[2]
            ner2 = arr[3]
            if sent_id not in relation_by_sent_id:
                relation_by_sent_id[sent_id] = {}
            relation_by_sent_id[sent_id][(ner1, ner2)] = rel
    return relation_by_sent_id


def compute_feature_key_to_anno_key(anno_by_sent_id, features_by_sent_id):
    # Number of sentences should be the same
    assert len(features_by_sent_id) == len(anno_by_sent_id)
    sent_ids = features_by_sent_id.keys()

    # For each annotation, find it's features from the input
    # Note: They are not always the same :_(
    # i.e "United States" in .annotations is "the United States" in .processed
    from difflib import SequenceMatcher
    feature_key_to_anno_key = {}
    SIM_THRESHOLD = 0.7
    removed_anno_count = 0
    added_anno_count = 0
    for sent_id in sent_ids:
        for anno_key in anno_by_sent_id[sent_id]:
            anno_ner1, anno_ner2 = anno_key
            found_f_key = None
            f_key_score = 0.0
            both_passed_threshold = False
            both_passed_shared_word = False
            for f_key in features_by_sent_id[sent_id]:
                f_ner1, f_ner2 = f_key
                ner1_sim = SequenceMatcher(None, anno_ner1, f_ner1).ratio()
                ner2_sim = SequenceMatcher(None, anno_ner2, f_ner2).ratio()
                if ner1_sim + ner2_sim > f_key_score:
                    f_key_score = ner1_sim + ner2_sim
                    found_f_key = f_key
                    both_passed_threshold = ner1_sim > SIM_THRESHOLD and ner2_sim > SIM_THRESHOLD
                    both_passed_shared_word = \
                        len(set(anno_ner1.replace(".", "").split()) & set(f_ner1.replace(".", "").split())) > 0 and \
                        len(set(anno_ner2.replace(".", "").split()) & set(f_ner2.replace(".", "").split())) > 0

            """ 
                Uncomment to see warnings of low-percent matching. I chose to remove those without at least one shared word
                I observed that if a NER exists in the possibilities it would choose it correctly, if it doesn't it chooses a bad one
                But both_passed_shared_word would be False on those occasions

            if not both_passed_threshold:
                print("WARNING: match for annotation key didn't pass threshold")
                print("Sentence id: "+sent_id)
                print("Selected match: "+str(anno_key)+" -> "+str(found_f_key))
                print("Possible matches: "+str(set([a for a,b in features_by_sent_id[sent_id].keys()])))
                if not both_passed_shared_word:
                    print("WARNING: extra low rating. Consider filtering out")
                print("\n")
            """
            if sent_id not in feature_key_to_anno_key:
                feature_key_to_anno_key[sent_id] = {}
            if both_passed_shared_word:
                if found_f_key in feature_key_to_anno_key[sent_id]:
                    print("Warning! double annotation for sentence: " + sent_id + " skipping.\n")
                else:
                    feature_key_to_anno_key[sent_id][found_f_key] = anno_key
                    added_anno_count += 1
            else:
                print("Sentence id: " + sent_id)
                print("Removed match: " + str(anno_key) + " -> " + str(found_f_key))
                print("")
                removed_anno_count += 1

    assert added_anno_count == sum([len(feature_key_to_anno_key[k]) for k in feature_key_to_anno_key])
    print("Found: {} annotations. Removed (because could not find match): {}"
          .format(added_anno_count, removed_anno_count))
    return feature_key_to_anno_key


def convert_features_to_numbers(features_by_sent_id, anno_by_sent_id, feature_key_to_anno_key, Counters=None):
    from utils import StringCounter
    import numpy as np
    sent_ids = features_by_sent_id.keys()
    features_dim_count = len(features_by_sent_id[sent_ids[0]].values()[0])

    allowed_anno = {"Work_For", "Live_In"}

    if Counters is None:
        Counters = np.ndarray(shape=(features_dim_count + 1), dtype=object)
        for i in range(features_dim_count + 1):
            Counters[i] = StringCounter()
        for sent_id in sent_ids:
            for f_key, features in features_by_sent_id[sent_id].items():
                [Counters[i].get_id_and_update(f) for i, f in enumerate(features)]
        # Filter rare features
        for i in range(features_dim_count):
            Counters[i].filter_rare_words(2)  # Min appearance: 2
            Counters[i] = StringCounter(Counters[i].S2I.keys(), "*UNK*")

    X = []
    Y = []
    YCounter = Counters[-1]
    removed_anno_count = 0
    for sent_id in sent_ids:
        for f_key, features in features_by_sent_id[sent_id].items():
            features_as_ids = tuple([Counters[i].get_id(f) for i, f in enumerate(features)])
            X.append(features_as_ids)
            anno_key = feature_key_to_anno_key[sent_id].get(f_key, None)
            anno = anno_by_sent_id[sent_id].get(anno_key, None)
            if anno == None:
                anno = "None"
            else:
                if anno not in allowed_anno:
                    anno = "None"
                    removed_anno_count += 1
            Y.append(YCounter.get_id_and_update(anno))

    print("Removed {} annotations, because they are not in {}".format(removed_anno_count, allowed_anno))

    return Counters, X, Y


if __name__ == '__main__':
    features_by_sent_id = read_processed_file("../data/Corpus.TRAIN.processed")
    anno_by_sent_id = read_annotations_file("../data/TRAIN.annotations")
    feature_key_to_anno_key = compute_feature_key_to_anno_key(anno_by_sent_id, features_by_sent_id)
    Counters, TrainX, TrainY = convert_features_to_numbers(features_by_sent_id, anno_by_sent_id,
                                                           feature_key_to_anno_key)

    features_by_sent_id = read_processed_file("../data/Corpus.DEV.processed")
    anno_by_sent_id = read_annotations_file("../data/DEV.annotations")
    feature_key_to_anno_key = compute_feature_key_to_anno_key(anno_by_sent_id, features_by_sent_id)
    Counters, DevX, DevY = convert_features_to_numbers(features_by_sent_id, anno_by_sent_id, feature_key_to_anno_key,
                                                       Counters=Counters)

    import utils

    class_dict = utils.inverse_dict(Counters[-1].S2I)

    import svm
    #svm.run_svm_print_result(TrainX, TrainY, DevX, DevY, class_dict)

    import mlp
    features_per_dim = [len(c.S2I) for c in Counters[:-1]]
    mlp.run_mlp_print_result(TrainX, TrainY, DevX, DevY, class_dict, features_per_dim)

    print("Done")