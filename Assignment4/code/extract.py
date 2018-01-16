
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
    pass


def extract_base_sysntactic_chunk_path(sentence, ner1, ner2):
    pass


def extract_typed_dependency_path(sentence, ner1, ner2):
    pass


def print_ner_entities(sentence, ner_entities):
    for ner in ner_entities:
        name = extract_ner_name(sentence, ner)
        ner_type = extract_ner_type(sentence, ner)
        print(name +" "+ner_type)


def process_sentence(sentence):
    ner_entities = extract_ners_from_sentence(sentence)
    #print_ner_entities(sentence, ner_entities)
    # Some sentences have as many as 12 NERS in one sentence
    import itertools
    ner_pairs = list(itertools.combinations(ner_entities, 2))
    for ner1, ner2 in ner_pairs:
        entity_type_1 = extract_ner_type(sentence, ner1)
        entity_type_2 = extract_ner_type(sentence, ner2)
        type_concated = entity_type_1+entity_type_2
        constituent_path = extract_constituent_path(sentence, ner1, ner2)
        chunk_path = extract_base_sysntactic_chunk_path(sentence, ner1, ner2)


if __name__ == '__main__':
    ID_IDX = 0
    LEMMA_IDX = 2
    POS_IDX = 3
    HEAD_IDX = 5
    TREE_IDX = 6
    NER_P_IDX = 7
    NER_IDX = 8

    with open("../data/Corpus.TRAIN.processed") as inp_file:
        sentence = []
        saw_empty_line = True
        for line in inp_file:
            line = line.strip()
            if line.startswith("#"):
                continue # Comment, skip
            if len(line) > 0:
                saw_empty_line = False
                arr = line.split()
                ner_type = arr[NER_IDX] if len(arr) > 8 else None
                word = (arr[ID_IDX], arr[LEMMA_IDX], arr[POS_IDX], arr[HEAD_IDX], arr[TREE_IDX], arr[NER_P_IDX], ner_type)
                sentence.append(word)
            else:
                if not saw_empty_line:
                    process_sentence(sentence)
                    sentence = []
                saw_empty_line = True