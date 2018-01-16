
def extract_ners_from_sentence(sentence):
    ner_entities = []
    last_ner = []
    for i, word in enumerate(sentence):
        if word[5] == 'B':  # Begin NER entity
            last_ner = [i]
        else:
            if word[5] == 'I':
                assert len(last_ner) != 0 #Illegal NER annotation, I without B
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
    return "TODO_const"


def extract_base_sysntactic_chunk_path(sentence, ner1, ner2):
    return "TODO_chunk"


def extract_typed_dependency_path(sentence, ner1, ner2):
    return "TODO_dep"


def print_ner_entities(sentence, ner_entities):
    for ner in ner_entities:
        name = extract_ner_name(sentence, ner)
        ner_type = extract_ner_type(sentence, ner)
        print(name +" "+ner_type)


def process_sentence(sentence):
    result = {}
    ner_entities = extract_ners_from_sentence(sentence)
    #print_ner_entities(sentence, ner_entities)
    # Some sentences have as many as 12 NERS in one sentence
    import itertools
    ner_pairs = list(itertools.combinations(ner_entities, 2))
    for tmpNer1, tmpNer2 in ner_pairs:
        for ner1, ner2 in [(tmpNer1, tmpNer2), (tmpNer2, tmpNer1)]: #pair, reverse_pair
            name_1 = extract_ner_name(sentence, ner1)
            name_2 = extract_ner_name(sentence, ner2)
            entity_type_1 = extract_ner_type(sentence, ner1)
            entity_type_2 = extract_ner_type(sentence, ner2)
            type_concated = entity_type_1+entity_type_2
            constituent_path = extract_constituent_path(sentence, ner1, ner2)
            chunk_path = extract_base_sysntactic_chunk_path(sentence, ner1, ner2)
            dep_path = extract_typed_dependency_path(sentence, ner1, ner2)
            result[(name_1, name_2)] = \
                (entity_type_1, entity_type_2, type_concated, constituent_path, chunk_path, dep_path)

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
                arr[ID_IDX], arr[WORD_IDX], arr[POS_IDX], arr[HEAD_IDX], arr[TREE_IDX], arr[NER_P_IDX], ner_type)
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
            arr = line.split('\t', 5) # Last element in the whole sentence
            sent_id = arr[0]
            ner1 = arr[1]
            rel = arr[2]
            ner2 = arr[3]
            if sent_id not in relation_by_sent_id:
                relation_by_sent_id[sent_id] = {}
            relation_by_sent_id[sent_id][(ner1,ner2)] = rel
    return relation_by_sent_id


if __name__ == '__main__':

    features_by_sent_id = read_processed_file("../data/Corpus.TRAIN.processed")
    anno_by_sent_id = read_annotations_file("../data/TRAIN.annotations")

    print(features_by_sent_id)
    print(anno_by_sent_id)