import sys

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2:
        print "Wrong number of arguments. Use:\n" + \
                "python MLETrain.py input_file_name q_mle_filename e_mle_filename"
        exit()
