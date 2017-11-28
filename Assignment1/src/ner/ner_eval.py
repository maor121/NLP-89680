import sys
import codecs
gold_file=sys.argv[1]
pred_file=sys.argv[2]

def read_data(fname):
    for line in codecs.open(fname):
        line = line.strip().split()
        tagged = [x.rsplit("/",1) for x in line]
        yield tagged


def normalize_bio(tagged_sent, is_spanned):
    last_bio, last_type = "O","O"
    normalized = []
    if is_spanned:
        for word, tag in tagged_sent:
            if tag == "O": tag = "O-O"
            bio,typ = tag.split("-",1)
            if bio=="I" and last_bio=="O": bio="B"
            if bio=="I" and last_type!=typ: bio="B"
            normalized.append((word,(bio,typ)))
            last_bio,last_type=bio,typ
    else:
        for word, tag in tagged_sent:
            normalized.append((word, tag))
    return normalized

def compare_accuracy(gold, pred):
    assert(len(gold_data)==len(pred_data))
    correct = 0.0
    total = 0.0
    for gold_sent, pred_sent in zip(gold, pred):
        assert(len(gold_sent)==len(pred_sent))
        gws = [w for w,t in gold_sent]
        pws = [w for w,t in pred_sent]
        assert(gws==pws)
        gtags = [t for w,t in gold_sent]
        ptags = [t for w,t in pred_sent]
        correct += sum([1 if g==p else 0 for g,p in zip(gold_sent, pred_sent)])
        total += len(gold_sent)
    return correct/total

def get_entities(sent):
    ent=[]
    for i,(word,tag) in enumerate(sent):
        bio,typ=tag
        if bio=="B":
            if ent: yield tuple(ent)
            ent=[]
            ent.append(i)
            ent.append(typ)
            ent.append(word)
        if bio=="I":
            ent.append(word)
        if bio=="O":
            if ent: yield tuple(ent)
            ent=[]
    if ent: yield tuple(ent)

if __name__=='__main__':

    is_spans=True #The original intention of the file
    sys.argv=sys.argv[1:]
    if len(sys.argv)>2:
        is_spans=sys.argv[2] in (1, 'True','y')

    gold_data = [normalize_bio(tagged, is_spans) for tagged in read_data(gold_file)]
    pred_data = [normalize_bio(tagged, is_spans) for tagged in read_data(pred_file)]

    assert(len(gold_data)==len(pred_data))

    acc = compare_accuracy(gold_data, pred_data)
    print "Accuracy: %.3f" % acc
    print

    if is_spans:
        gold_entities = set()
        for i,sent in enumerate(gold_data):
            for entity in get_entities(sent):
                gold_entities.add((i,entity))

        pred_entities = set()
        for i,sent in enumerate(pred_data):
            for entity in get_entities(sent):
                pred_entities.add((i,entity))

        print

        prec = len(gold_entities.intersection(pred_entities)) / float(len(pred_entities))
        rec  = len(gold_entities.intersection(pred_entities)) / float(len(gold_entities))
        print "All-types \tPrec:%.3f Rec:%.3f" % (prec, rec)

        types = set([e[1][1] for e in gold_entities]) - set(["O"])
        for t in types:
            gents = set([e for e in gold_entities if e[1][1]==t])
            pents = set([e for e in pred_entities if e[1][1]==t])
            prec = len(gents.intersection(pents)) / float(len(pents))
            rec  = len(gents.intersection(pents)) / float(len(gents))
            f1 = prec * rec / (prec + rec)
            print "%10s \tPrec:%.3f Rec:%.3f F1:%.3f" % (t, prec, rec,f1)




