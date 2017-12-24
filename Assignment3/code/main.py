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

    words_without_context = []
    words_with_context = []
    for w, w_id in preprocess.W2I.S2I.items():
        if w_id not in preprocess.contexts:
            words_without_context.append(w)
        else:
            words_with_context.append(w)

    print("Words without context:%d\n%s" % (len(words_without_context), words_without_context))
    print("Words with context:%d\n%s" % (len(words_with_context), words_with_context))

    print(0)
