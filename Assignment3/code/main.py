if __name__ == '__main__':
    import time
    from preprocess import Preprocess

    is_tiny = False

    if is_tiny:
        filename = "wikipedia.tinysample.trees.lemmatized"
    else:
        filename = "wikipedia.sample.trees.lemmatized"
    time_s = time.time()
    preprocess = Preprocess.from_input(filename)
    preprocess.save_to_file("../out/preprocess.pickle")
    time_e = time.time()
    print("Done. time: %.2f secs" % (time_e - time_s))

    preprocess = Preprocess.load_from_file("../out/preprocess.pickle")

    print(0)
