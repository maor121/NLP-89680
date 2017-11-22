from MLETrain import MLETrain
import sys

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) not in (4, 5):
        print "Wrong number of arguments. Use:\n" + \
                "python GreedyTag.py input_file q_mle_file e_mle_file, out_file, [extra_file]"
        exit()

    input_filename = args[0]
    q_mle_filename = args[1]
    e_mle_filename = args[2]
    out_filename = args[3]
    extra_filename = args[4] if len(args) >= 5 else None

    MLETrain.createModelFilesFromInput(input_filename, q_mle_filename, e_mle_filename)
    model = MLETrain(q_mle_filename, e_mle_filename)
